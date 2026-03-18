import argparse
import re
import pandas as pd
from pathlib import Path
from datasets import load_dataset

from benchmarking_utils.answering import (
    MAX_TOKENS,
    generateAnswers_long, generateAnswers_mcq, save_results_to_json
)

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_path",
        type=str, 
        required=True,
        help="path of the original data",
    )
    parser.add_argument(
        "--models",
        nargs="+",  # Accepts one or more values as a list
        required=True,
        help="names of models to run inference (provide as space-separated values)",
    )
    
    # For --setting (string)
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        help="[mcq-answering, long-form-answering]",
    )
    
    # For --results_dataset_path (list of strings)
    parser.add_argument(
        "--results_dataset_path",
        type=str,  # Accept as a JSON string
        required=True,
        help="list of paths to save results (provide as a JSON string)",
    )
    parser.add_argument(
        "--evaluation_style",
        type=str,  
        help="evaluation styles: direct, CoT",
    )
    parser.add_argument(
        "--num_samples",
        type=str,  
        required=True,
        default='all',
        help="samples to evaluate",
    )
    parser.add_argument(
        "--restart_from",
        type=str,  
        default=None,
        help="from where to restart"
    )
    parser.add_argument(
        "--restart_path",
        type=str,
        default= None,  
        help="from where to restart",
    )
    parser.add_argument(
        "--subset",
        type=str,
        help = "whether to subset by expert or not"
    )
    parser.add_argument(
        "--with_search",
        action="store_true",
        required=True,
        help='whether to use web search in mcq responses'
    )
    args = parser.parse_args()
    return args



def format_answer_direct(answer: str):
    """
    Returns: A zero-indexed integer corresponding to the answer.
    """
    if not isinstance(answer, str):
        raise ValueError(f"Invalid input: '{answer}'.")
    if "A" <= answer <= "C":
        # Convert letter to zero-indexed number
        return ord(answer) - ord("A")
    else:
        return f"Invalid answer: '{answer}'. Must be a letter (A-C)"
    
def extract_answer_from_tags(answer: str):
    """
    Searchs for the answer between tags <Answer>.

    Returns: A zero-indexed integer corresponding to the answer.
    """
    pattern = r"<ANSWER>\s*([A-Za-z])\s*</ANSWER>"
    match = re.search(pattern, answer, re.IGNORECASE)

    if match:
        # Extract and convert answer letter
        letter = match.group(1).upper()
        election = ord(letter) - ord("A")

    elif len(answer) == 1:
        if "A" <= answer <= "Z":
            # Convert letter to zero-indexed number
            election = ord(answer) - ord("A")
        elif "1" <= answer <= "9":
            # Convert digit to zero-indexed number
            election = int(answer) - 1
        else:
            election = answer
    else:
        # Error handling cases
        election = "No valid answer tag found"
        if re.search(r"<ANSWER>.*?</ANSWER>", answer):
            election = "Answer tag exists but contains invalid format"
        
    return election



def filter_data_by_settings(data_path, 
                            num_samples, 
                            restart_from, 
                            restart_path,
                            subset):
    data = load_dataset(data_path)['train'].to_pandas() 
    
    if num_samples != 'all':
        data = data.sample(n=int(num_samples))

    if subset:
        if 'is_expert' in data.columns:
            if subset == 'expert':
                data = data[data['is_expert']==True]
            if subset == 'synthetic':
                data = data[data['is_expert']==False]

    if restart_from:
        restart_from = int(restart_from)
        if restart_path is None:
            raise ValueError("restart_path must be provided when restart_from is provided")

        previous_results = pd.read_json(restart_path)
        previous_results = previous_results.iloc[:restart_from]
        remaining_data = data.iloc[restart_from:]
        combined_data = pd.concat([previous_results, remaining_data], ignore_index=True)

        return combined_data
    
    else:
        return data
    

def run_answer_prediction(args):

    data = filter_data_by_settings(
        data_path=args.data_path,
        num_samples=args.num_samples,
        restart_from=args.restart_from,
        restart_path=args.restart_path,
        subset=args.subset
    )
    
    results_path = Path(args.results_dataset_path)
    results_path.mkdir(parents=True, exist_ok=True)

    if args.restart_from:
        restart_from=int(args.restart_from)
    else:
        restart_from=0
    print(data)
    
    if args.setting == 'mcq-answering':
        if args.evaluation_style == 'direct':
            direct_system_message = "You are given a multiple-choice question to answer. You must respond ONLY with the letter corresponding to the correct answer, without any additional text or explanation. Answer with the letter directly, do not include any other word. You can answer 'A','B or 'C'."
            answers = generateAnswers_mcq(data,  
                                          args.models, 
                                          max_tokens=1, 
                                          system_message = direct_system_message, 
                                          style = args.evaluation_style,
                                          restart_from=restart_from,
                                          answer_extractor= format_answer_direct,
                                          output_folder=args.results_dataset_path,
                                          with_search=args.with_search)
        elif args.evaluation_style == 'CoT':
            cot_system_message = 'The following is a multiple-choice question. Think step by step and then provide your FINAL answer between the tags <ANSWER> X </ANSWER> where X is ONLY the correct letter of your choice. Do not write additional text between the tags.'
            answers = generateAnswers_mcq(data,  
                                          args.models, 
                                          max_tokens= MAX_TOKENS, 
                                          system_message = cot_system_message, 
                                          style = args.evaluation_style,
                                          restart_from=restart_from,
                                          answer_extractor= extract_answer_from_tags,
                                          output_folder=args.results_dataset_path,
                                          with_search=args.with_search)
        else: 
            raise ValueError(f'MCQ: Unexpected evaluation_style value: {args.evaluation_style}')
    elif args.setting == 'long-form-answering':
        answers = generateAnswers_long(data, args.models, restart_from, Path(args.results_dataset_path)) # Pass Path for output_folder   
    else:
        raise ValueError(f'unexpected value {args.setting} for an inference setting')

    #save answers
    save_results_to_json(answers, args.setting + f'_{args.evaluation_style}', args.models, results_path)


def main():
    args = parse_args()
    run_answer_prediction(args)


if __name__ == '__main__':
    main()
