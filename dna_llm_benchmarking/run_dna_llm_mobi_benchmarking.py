import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from datasets import load_dataset

from benchmarking_utils.answering import MAX_TOKENS, TEMPERATURE, get_keys
from benchmarking_utils.instantiation import instantiate_models


DEFAULT_DATA_PATH = "manufernandezbur/mobiplant"
DEFAULT_CONFIG = "dna_llm"
DEFAULT_STYLE = "CoT"
DEFAULT_CHECKPOINT_EVERY = 50
DEFAULT_MAX_API_WORKERS = 30


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH,
		help="Local JSON file path, or a Hugging Face dataset repo id loaded via load_dataset")
	parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
		help="Hugging Face dataset config/subset name (ignored when --data_path is a local file)")
	parser.add_argument("--model", type=str, required=True)
	parser.add_argument("--results_dataset_path", type=str, required=True)
	parser.add_argument("--evaluation_style", type=str, default=DEFAULT_STYLE)
	parser.add_argument("--num_samples", type=str, default="all")
	parser.add_argument("--with_search", action="store_true")
	parser.add_argument("--max_api_workers", type=int, default=DEFAULT_MAX_API_WORKERS)
	parser.add_argument("--checkpoint_every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
	return parser.parse_args()


def infer_task_type(answer_value):
	if isinstance(answer_value, str):
		lowered = answer_value.strip().lower()
		if lowered in {"yes", "no"}:
			return "classification"
	try:
		float(answer_value)
		return "regression"
	except (TypeError, ValueError):
		return "unknown"


def normalize_ground_truth(answer_value):
	task_type = infer_task_type(answer_value)
	if task_type == "classification":
		return str(answer_value).strip().lower()
	if task_type == "regression":
		return float(answer_value)
	return answer_value


def sample_key(row: pd.Series) -> str:
	answer_norm = normalize_ground_truth(row["answer"])
	payload = f"{row['question']}||{row['reference_dataset']}||{answer_norm}"
	return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_answer_from_tags(model_output: str, task_type: str):
	if not isinstance(model_output, str):
		return None, "invalid_output_type"

	match = re.search(r"<\s*ANSWER\s*>\s*(.*?)\s*<\s*/\s*ANSWER\s*>", model_output, flags=re.IGNORECASE | re.DOTALL)
	if not match:
		return None, "missing_answer_tag"

	extracted_value = match.group(1).strip()
	if task_type == "classification":
		label = extracted_value.lower()
		if label in {"yes", "no"}:
			return label, "ok"
		return None, "invalid_classification_label"

	if task_type == "regression":
		try:
			value = float(extracted_value)
		except ValueError:
			return None, "invalid_regression_value"
		if -10.0 <= value <= 10.0:
			return value, "ok"
		return None, "regression_out_of_range"

	return None, "unknown_task_type"


def extract_answer_without_tags(model_output: str, task_type: str):
	"""Fallback parser for responses that omit <ANSWER> tags."""
	if not isinstance(model_output, str):
		return None, "invalid_output_type"

	text = model_output.strip()
	if not text:
		return None, "empty_output"

	if task_type == "classification":
		guided = re.search(r"(?:final\s+answer|answer|prediction)\s*[:\-]?\s*(yes|no)\b", text, flags=re.IGNORECASE)
		if guided:
			return guided.group(1).lower(), "ok_without_tags"

		labels = re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
		unique_labels = {label.lower() for label in labels}
		if len(unique_labels) == 1:
			return unique_labels.pop(), "ok_without_tags_single_label"
		return None, "invalid_classification_label"

	if task_type == "regression":
		guided = re.search(r"(?:final\s+answer|answer|prediction)\s*[:\-]?\s*(-?\d+(?:\.\d+)?)\b", text, flags=re.IGNORECASE)
		value_text = guided.group(1) if guided else None
		if value_text is None:
			all_numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
			if len(all_numbers) == 1:
				value_text = all_numbers[0]

		if value_text is None:
			return None, "invalid_regression_value"

		try:
			value = float(value_text)
		except ValueError:
			return None, "invalid_regression_value"

		if -10.0 <= value <= 10.0:
			return value, "ok_without_tags"
		return None, "regression_out_of_range"

	return None, "unknown_task_type"


def is_valid_prediction(prediction_value, task_type: str) -> bool:
	if task_type == "classification":
		return isinstance(prediction_value, str) and prediction_value.lower() in {"yes", "no"}
	if task_type == "regression":
		try:
			value = float(prediction_value)
			return -10.0 <= value <= 10.0
		except (TypeError, ValueError):
			return False
	return False


def load_input_data(data_path: str, num_samples: str, config: str | None = None) -> pd.DataFrame:
	input_path = Path(data_path)
	if input_path.exists():
		# Local JSON file
		with input_path.open("r", encoding="utf-8") as f:
			rows = json.load(f)
		df = pd.DataFrame(rows)
	else:
		# Hugging Face dataset repo id (optionally a named config/subset)
		dataset = load_dataset(data_path, name=config, split="train")
		df = dataset.to_pandas()

	required_columns = {"question", "answer", "reference_dataset"}
	if not required_columns.issubset(df.columns):
		missing = sorted(required_columns.difference(set(df.columns)))
		raise ValueError(f"Input data is missing required columns: {missing}")

	if num_samples != "all":
		count = int(num_samples)
		df = df.head(count).copy()

	df["answer"] = df["answer"].apply(normalize_ground_truth)
	df["task_type"] = df["answer"].apply(infer_task_type)
	df["sample_key"] = df.apply(sample_key, axis=1)
	return df


def output_paths(results_root: Path, style: str, model: str):
	temp_dir = results_root / "temp"
	inference_dir = results_root / "inference"
	temp_dir.mkdir(parents=True, exist_ok=True)
	inference_dir.mkdir(parents=True, exist_ok=True)

	temp_file = temp_dir / f"dna-temp-{style}_{model}_results.json"
	final_file = inference_dir / f"dna-benchmarking_{style}_{model}_results.json"
	return temp_file, final_file


def discover_existing_result_files(results_root: Path, style: str, model: str):
	pattern = f"*{style}*{model}*_results.json"
	candidates = list((results_root / "temp").glob(pattern)) + list((results_root / "inference").glob(pattern))
	return [path for path in candidates if path.is_file()]


def load_existing_completed_rows(
	results_root: Path,
	style: str,
	model: str,
	prediction_col: str,
	reasoning_col: str,
	status_col: str,
) -> dict:
	completed_by_key = {}
	files = discover_existing_result_files(results_root, style, model)
	files.sort(key=lambda p: p.stat().st_mtime)

	for file_path in files:
		try:
			df = pd.read_json(file_path)
		except ValueError:
			continue
		if "sample_key" not in df.columns:
			if {"question", "answer", "reference_dataset"}.issubset(df.columns):
				df = df.copy()
				df["answer"] = df["answer"].apply(normalize_ground_truth)
				df["sample_key"] = df.apply(sample_key, axis=1)
			else:
				continue

		for _, row in df.iterrows():
			task_type = infer_task_type(row.get("answer"))
			prediction_value = row.get(prediction_col)
			if is_valid_prediction(prediction_value, task_type):
				reasoning_value = row.get(reasoning_col) if reasoning_col in df.columns else pd.NA
				status_value = row.get(status_col) if status_col in df.columns else "ok_from_checkpoint"
				completed_by_key[row["sample_key"]] = {
					prediction_col: prediction_value,
					reasoning_col: reasoning_value,
					status_col: status_value,
					"task_type": task_type,
				}
	return completed_by_key


def save_json_records(df: pd.DataFrame, output_file: Path):
	serializable = df.copy()
	serializable.to_json(output_file, orient="records", indent=2)
	print(f"Saved results to {output_file}")


def build_prompt(question: str, task_type: str, with_search: bool):
	if task_type == "classification":
		task_instruction = "This is a binary classification task. Output only 'yes' or 'no' inside <ANSWER> tags."
	elif task_type == "regression":
		task_instruction = "This is a regression task. Output one float in [-10, 10] inside <ANSWER> tags."
	else:
		task_instruction = "Output the final answer inside <ANSWER> tags."

	system_message = (
		"You are solving DNA sequence tasks. Think step by step and provide the final answer in tags <ANSWER> X </ANSWER> where X is ONLY the correct answer. "
		"Do not include any text inside tags except the final answer value."
	)
	if with_search:
		system_message = system_message + "You have web access."
	user_message = f"{task_instruction}\n\nQuestion:\n{question}"
	return system_message, user_message


def format_reasoning_response(response_obj, parser: StrOutputParser):
	if hasattr(response_obj, "content_blocks"):
		chunks = []
		for block in response_obj.content_blocks:
			block_type = block.get("type")
			if block_type == "reasoning":
				chunks.append(f"<REASONING>{block.get('reasoning', '')}</REASONING>")
			if block_type == "server_tool_result":
				output = block.get("output")
				if isinstance(output, list):
					urls = [entry.get("url") for entry in output if isinstance(entry, dict) and entry.get("url")]
					chunks.extend([f"<URL>{url}</URL>" for url in urls])
			if block_type == "text":
				text_value = block.get("text", "")
				if text_value:
					chunks.append(text_value)
		parsed_text = parser.invoke(response_obj)
		if not parsed_text:
			content_attr = getattr(response_obj, "content", None)
			if isinstance(content_attr, str):
				parsed_text = content_attr
			elif isinstance(content_attr, list):
				list_text_parts = []
				for item in content_attr:
					if isinstance(item, str):
						list_text_parts.append(item)
					elif isinstance(item, dict):
						text_value = item.get("text")
						if isinstance(text_value, str) and text_value:
							list_text_parts.append(text_value)
				parsed_text = "\n".join(list_text_parts)
		if parsed_text and (not chunks or parsed_text not in chunks):
			chunks.append(parsed_text)
		return "\n".join(chunks)

	return parser.invoke(response_obj)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def invoke_with_retry(chat_model, prompt_obj):
	return chat_model.invoke(prompt_obj)


def run_row_inference(row, model_name: str, chat_model, parser: StrOutputParser, with_search:bool):
	task_type = row["task_type"]
	system_message, user_message = build_prompt(row["question"], task_type, with_search)
	if model_name in {"o1-mini", "r1"}:
		prompt_obj = system_message + "\n" + user_message
	else:
		prompt_obj = [
			{"role": "system", "content": system_message},
			{"role": "user", "content": user_message},
		]

	response_obj = invoke_with_retry(chat_model, prompt_obj)
	reasoning_text = format_reasoning_response(response_obj, parser)
	extracted, status = extract_answer_from_tags(reasoning_text, task_type)
	if status == "missing_answer_tag":
		fallback_extracted, fallback_status = extract_answer_without_tags(reasoning_text, task_type)
		if fallback_status.startswith("ok_"):
			extracted, status = fallback_extracted, fallback_status
		else:
			stop_reason = (getattr(response_obj, "response_metadata", {}) or {}).get("stop_reason")
			if stop_reason == "refusal" and not str(reasoning_text).strip():
				status = "refusal_empty_response"
	return extracted, reasoning_text, status


def run_answer_prediction(args: argparse.Namespace):
	if args.checkpoint_every <= 0:
		raise ValueError("--checkpoint_every must be > 0")
	if args.max_api_workers <= 0:
		raise ValueError("--max_api_workers must be > 0")

	style = args.evaluation_style
	model = args.model
	results_root = Path(args.results_dataset_path)
	results_root.mkdir(parents=True, exist_ok=True)

	prediction_col = f"{style}_extracted_by_{model}"
	reasoning_col = f"{style}_reasoning_by_{model}"
	status_col = f"{style}_status_by_{model}"

	df = load_input_data(args.data_path, args.num_samples, args.config)
	df[prediction_col] = pd.NA
	df[reasoning_col] = pd.NA
	df[status_col] = pd.NA

	completed = load_existing_completed_rows(
		results_root,
		style,
		model,
		prediction_col,
		reasoning_col,
		status_col,
	)
	for idx, row in df.iterrows():
		entry = completed.get(row["sample_key"])
		if entry is not None:
			df.at[idx, prediction_col] = entry[prediction_col]
			df.at[idx, reasoning_col] = entry.get(reasoning_col, pd.NA)
			df.at[idx, status_col] = entry.get(status_col, "ok_from_checkpoint")

	missing_indices = []
	for idx, row in df.iterrows():
		if not is_valid_prediction(df.at[idx, prediction_col], row["task_type"]):
			missing_indices.append(idx)

	print(f"Detected {len(df) - len(missing_indices)} completed samples and {len(missing_indices)} missing samples.")

	temp_file, final_file = output_paths(results_root, style, model)
	if not missing_indices:
		print(f"All samples where completed with this model in {results_root}.")
		return

	chat_models = instantiate_models(
		keys=get_keys(),
		models=[model],
		max_tokens=MAX_TOKENS,
		temperature=TEMPERATURE,
		with_search=args.with_search,
	)
	chat_model = chat_models[model]
	parser = StrOutputParser()

	processed = 0
	total_missing = len(missing_indices)
	with ThreadPoolExecutor(max_workers=args.max_api_workers) as executor:
		future_map = {
			executor.submit(run_row_inference, df.loc[idx], model, chat_model, parser, args.with_search): idx
			for idx in missing_indices
		}

		for future in tqdm(as_completed(future_map), total=total_missing):
			idx = future_map[future]
			try:
				extracted, reasoning_text, status = future.result()
				df.at[idx, prediction_col] = extracted
				df.at[idx, reasoning_col] = reasoning_text
				df.at[idx, status_col] = status
			except Exception as exc:
				df.at[idx, prediction_col] = pd.NA
				df.at[idx, reasoning_col] = f"ERROR: {str(exc)}"
				df.at[idx, status_col] = "error"

			processed += 1
			if processed % args.checkpoint_every == 0 or processed == total_missing:
				save_json_records(df, temp_file)

	save_json_records(df, final_file)


def main():
	args = parse_args()
	run_answer_prediction(args)


if __name__ == "__main__":
	main()
