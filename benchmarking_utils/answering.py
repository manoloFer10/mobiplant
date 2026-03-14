import json
import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor # Added import
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm
from benchmarking_utils.instantiation import instantiate_models
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential


TEMPERATURE = 0.7
MAX_TOKENS = 8192
MAX_API_WORKERS = 30 # Added max workers


def get_keys():
    keys_path = Path('tokens.json')
    if keys_path.exists():
        with keys_path.open('r', encoding='utf-8') as f:
            keys = json.load(f)
    else:
        # try environment keys
        openai_key = os.environ.get('OPENAI_KEY')
        anthropic_key = os.environ.get('ANTHROPIC_KEY')
        keys = {
            'gpt-5.2': openai_key,
            'sonnet-4.6': anthropic_key
        }
    return keys


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_answer_with_retry(chat_model, prompt_):
    return chat_model.invoke(prompt_)


def save_results_to_json(data: pd.DataFrame, setting, models, output_folder):

    output_folder = Path(output_folder / 'inference')
    output_folder.mkdir(parents=True, exist_ok=True)
    
    models_str = "_".join(models)
    output_file = output_folder / f"{setting}_{models_str}_results.json"
    data.to_json(output_file, indent=2, orient='records')
    print(f"Saved results to {output_file}")



def generateAnswers_long(df: pd.DataFrame, 
                         models: list[str], 
                         restart_from:int, 
                         output_folder: str
                         ):
    output_path = Path(output_folder)
    parser = StrOutputParser()

    chat_models = instantiate_models(
        get_keys(), 
        models, 
        max_tokens=MAX_TOKENS, 
        temperature=TEMPERATURE
    )

    system_message = "You are given a plant molecular biology question to answer. Respond concisely in one paragraph and provide a source (document title and link) for your answer."

    def _task_long_form(chat_model_instance, current_prompt_obj, parser_instance, model_name_log, index_log):
        raw_model_answer_content = "N/A"
        try:
            raw_model_response_obj = generate_answer_with_retry(chat_model_instance, current_prompt_obj)
            if hasattr(raw_model_response_obj, 'content_blocks'):
                model_answer = ""
                for block in raw_model_response_obj.content_blocks:
                    if block["type"] == "reasoning":
                        model_answer+=f"REASONING: {block['reasoning']}\n"
                    elif block["type"] == "web_search_call":
                        model_answer+=f"SEARCHED: {block['query']}\n"
                    elif block["type"] == "web_search_result":
                        model_answer+=f"SOURCES: {block['urls']}\n"
                    elif block["type"] == "text":
                        model_answer+=f"TEXT: {block['text']}\n"
            elif hasattr(raw_model_response_obj, 'content'):
                raw_model_answer_content = raw_model_response_obj.content
                model_answer = parser_instance.invoke(raw_model_response_obj)
            return model_answer, None, model_name_log 
        except Exception as e:
            return raw_model_answer_content, e, model_name_log

    with ThreadPoolExecutor(max_workers=MAX_API_WORKERS) as executor:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            if index < restart_from:
                continue
            
            question = row['question']
            base_prompt_list = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question}
            ]

            futures = []
            for model_name, chat_model in chat_models.items():
                prompt_ = base_prompt_list
                if model_name == 'o1-mini': 
                    prompt_ = system_message + '\n' + base_prompt_list[1]['content']
                
                futures.append(executor.submit(_task_long_form, chat_model, prompt_, parser, model_name, index))

            for future in futures:
                model_answer_or_raw_on_error, error, model_name_res = future.result()
                if error:
                    print(f'Error with {model_name_res} for long-form on row {index}: {str(error)}')
                    print(f"  Raw response content on error: {model_answer_or_raw_on_error}")
                    df.at[index, 'long_answer_by_' + model_name_res] = f"ERROR: {str(error)} - Raw: {model_answer_or_raw_on_error}"
                else:
                    df.at[index, 'long_answer_by_' + model_name_res] = model_answer_or_raw_on_error

            if index % 50 == 0 or index == len(df) - 1 : # Save more frequently or at the end
                save_results_to_json(
                    data=df,
                    setting='long-temp',
                    models=models, # This should be list of model names processed so far or all models
                    output_folder=output_path / 'temp'
                )
                print(f'Saved long-form temp results up to index: {index}')
    return df


def _format_block(block):
    block_type = block.get('type')
    if block_type == 'reasoning':
        string = f'<REASONING>{block.get("reasoning")}</REASONING>'
    if block_type == 'server_tool_result':
        outputs=block.get('output')
        if isinstance(outputs, list):
            urls = [output.get('url') for output in outputs]
            string = "\n".join([f'<URL>{url}</URL>' for url in urls])
        else:
            urls = []
            string = "\n"
    if block_type == 'text':
        annotations = block.get('annotations')
        if isinstance(annotations,list):
            urls = [annotation.get('url') for annotation in annotations]
            string = "\n".join([f'<URL>{url}</URL>' for url in urls])
        else:
            urls = []
            string = "\n"
    return string


def generateAnswers_mcq(df: pd.DataFrame, 
                        models: list[str], 
                        max_tokens: int, 
                        system_message: str, 
                        style: str,
                        restart_from: int,
                        answer_extractor,
                        output_folder,
                        with_search):

    output_path = Path(output_folder)

    def generate_options_format(options: list[str]):
        formatted_options = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)]) 
        return formatted_options

    parser = StrOutputParser()

    chat_models = instantiate_models(
        get_keys(), 
        models, 
        max_tokens=max_tokens, 
        temperature=TEMPERATURE,
        with_search=with_search
    )

    BASE_PROMPT = """Question:\n{question}\n\nOptions:\n{options}\nAnswer:"""

    def _task_mcq(chat_model_instance, current_api_prompt_obj, parser_instance, answer_extractor_func, model_name_log, index_log, style_log):
        raw_response_content_for_error = "N/A"
        try:
            response_obj = chat_model_instance.invoke(current_api_prompt_obj)
            if hasattr(response_obj, 'content_blocks'):
                CoT_context = "\n".join(
                    _format_block(block)
                    for block in response_obj.content_blocks
                    if block.get("type") in('server_tool_result', 'reasoning', 'text')
                )  # reasoning usage and web searches
                model_answer_text = parser_instance.invoke(response_obj)
            else:
                model_answer_text = parser_instance.invoke(response_obj)
                CoT_context = model_answer_text

            extracted_answer = answer_extractor_func(model_answer_text)
            return extracted_answer, CoT_context, None, model_name_log 
        except Exception as e:
            return None, raw_response_content_for_error, e, model_name_log

    with ThreadPoolExecutor(max_workers=MAX_API_WORKERS) as executor:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            if index < restart_from:
                continue

            question_text = row['question']
            options_text = generate_options_format(row['options'])
            content = BASE_PROMPT.format(question=question_text, options=options_text)
            
            futures = []
            for model_name, chat_model in chat_models.items():
                current_api_prompt = ""
                if model_name in ['o1-mini', 'r1']: # due to reasoning models prompting constraints
                    current_api_prompt = system_message + '\n' + content
                else:
                    current_api_prompt = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": content}
                    ]
                
                futures.append(executor.submit(_task_mcq, chat_model, current_api_prompt, parser, answer_extractor, model_name, index, style))

            for future in futures:
                extracted_answer, model_answer_text_or_raw_on_error, error, model_name_res = future.result()
                if error:
                    print(f'Error with {model_name_res} for MCQ ({style}) on row {index}: {str(error)}')
                    print(f"  Raw response content on error: {model_answer_text_or_raw_on_error}")
                    df.at[index, f'{style}_election_by_' + model_name_res] = "ERROR"
                    df.at[index, f'{style}_reasoning_by_' + model_name_res] = f"ERROR: {str(error)} - Raw: {model_answer_text_or_raw_on_error}"
                else:
                    df.at[index, f'{style}_election_by_' + model_name_res] = extracted_answer
                    reasoning_value = model_answer_text_or_raw_on_error
                    if isinstance(reasoning_value, list):
                        reasoning_value = json.dumps(reasoning_value)
                    df.at[index, f'{style}_reasoning_by_' + model_name_res] = reasoning_value
        
            if index % 50 == 0 or index == len(df) -1 :
                save_results_to_json(
                    data=df,
                    setting=f'mcq-temp-{style}',
                    models=models, # This should be list of model names
                    output_folder=output_path / 'temp'
                )
                print(f'Saved MCQ temp results ({style}) until index: {index}')
    return df