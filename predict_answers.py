import argparse
import json
import re
import os
import pandas as pd
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

SUPPORTED_MODELS = ['llama', 'chatgpt', 'o1-mini', 'gemini', 'claude', 'r1', 'v3']

TEMPERATURE = 0.7
MAX_TOKENS = 4096

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
    
    # For --whos (list of strings)
    parser.add_argument(
        "--whos",
        nargs="+",  # Accepts one or more values as a list
        required=True,
        help="list of emails (provide as space-separated values)",
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
    args = parser.parse_args()
    return args

def get_keys():
    keys_path = Path('tokens.json')
    with keys_path.open('r', encoding='utf-8') as f:
        keys = json.load(f)
    return keys

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def instantiate_models(keys: dict, 
                       models: list[str], 
                       max_tokens: int, 
                       temperature: float) -> dict:
    if max_tokens is None:
        max_tokens = 256
    
    chat_models = {}
    for model in models:
        if model == 'gemini':
            replier = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=None,
                max_retries=2,
                google_api_key=keys[model]
            )

        if model == 'chatgpt':
            replier = ChatOpenAI(
                model='gpt-4o',
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=None,
                max_retries=2,
                api_key=keys[model]
            )
        if model == 'o1-mini':
            replier = ChatOpenAI(
                model="o1-mini",
                api_key=keys[model],
                temperature=1, # only param supported
                max_tokens=max_tokens,
                timeout=None,
                max_retries=2
            )
        if model == 'claude':
            replier = ChatAnthropic(
                model='claude-3-5-sonnet-latest',
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=2,
                timeout=None,
                api_key=keys[model]
            )
        if model == 'llama':
            replier = ChatOpenAI(
                model="llama3.1-405b",
                api_key=keys[model],
                base_url='https://api.llama-api.com',
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=None,
                max_retries=2
            )
        if model == 'v3':
            replier = ChatOpenAI(
                model="deepseek-chat",
                api_key=keys[model],
                base_url='https://api.deepseek.com', 
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=None,
                max_retries=2
            )
        if model == 'r1':
            replier = ChatOpenAI(
                model="deepseek-reasoner",
                api_key=keys[model],
                base_url='https://api.deepseek.com', 
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=None,
                max_retries=2,
                reasoning_effort ='low'
            )
        if model not in SUPPORTED_MODELS:
            raise NameError(f'Model:{model} not implemented for answering DB.')
        chat_models[model] = replier

    return chat_models

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


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_answer_with_retry(chat_model, prompt_):
    return chat_model.invoke(prompt_)

def generateAnswers_long(df: pd.DataFrame, 
                         models: list[str], 
                         restart_from:int, 
                         output_folder: str
                         ):

    output_path = Path(output_folder)

    parser = StrOutputParser()

    chat_models= instantiate_models(
        get_keys(), 
        models, 
        max_tokens=MAX_TOKENS, 
        temperature=TEMPERATURE
    )

    system_message = "You are given a plant molecular biology question to answer. Respond concisely in one paragraph and provide a source (document title and link) for your answer."

    for index, row in tqdm(df.iterrows(), total= len(df)):

        if index < restart_from: continue
        
        question = row['question']
        prompt = [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": question 
            }
        ]
        for model_name, chat_model in chat_models.items():
            prompt_ = prompt
            if model_name == 'o1-mini': 
                prompt_ = 'Respond concisely in one paragraph and provide a source (document title and link) for your answer.' + prompt[1]['content'] #due to o1 settings

            model_answer = parser.invoke(generate_answer_with_retry(chat_model, prompt_))
            df.at[index, 'long_answer_by_' + model_name] = model_answer

        #save temp results on every iteration
        save_results_to_csv(
            data=df,
            setting='long-temp',
            models=models,
            output_folder=output_path / 'temp'
        )

    return df


def generateAnswers_mcq(df: pd.DataFrame, 
                        models: list[str], 
                        max_tokens: int, 
                        system_message: str, 
                        style: str,
                        restart_from: int,
                        answer_extractor,
                        output_folder):

    output_path = Path(output_folder)

    def generate_options_format(options: list[str]):
        formatted_options = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)]) 
        return formatted_options

    parser = StrOutputParser()

    chat_models = instantiate_models(
        get_keys(), 
        models, 
        max_tokens=max_tokens, 
        temperature=TEMPERATURE
    )

    BASE_PROMPT = """Question:\n{question}\n\nOptions:\n{options}\nAnswer:"""

    for index, row in tqdm(df.iterrows(), total= len(df)):

        if index < restart_from: continue

        question_text =  row['question']
        for model_name, chat_model in chat_models.items():
            try:
                options_text = generate_options_format(row['options'])
                content = BASE_PROMPT.format(question=question_text, options=options_text)
               
                if model_name in ['o1-mini', 'r1']: 
                    prompt = system_message + '\n' + content #due to reasoning models requirements
                else:
                    prompt = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": content}
                    ]

                response = chat_model.invoke(prompt)
                
                model_answer = parser.invoke(response)
                answer = answer_extractor(model_answer)
                df.at[index, f'{style}_election_by_' + model_name] = answer
                df.at[index, f'{style}_reasoning_by_' + model_name] = model_answer

            except Exception as e:
                print(f'Error with {model_name}: {str(e)}')
                print(f"Raw response: {response.content if 'response' in locals() else ''}")
        
        save_results_to_csv(
            data=df,
            setting=f'mcq-temp-{style}',
            models=models,
            output_folder=output_path / 'temp'
        )
        print(f'Saved until index: {index}, restart_from should be {index+1}')
    return df

def filter_data_by_settings(data_path, 
                            email_list, 
                            num_samples, 
                            restart_from, 
                            restart_path):
    data_path = Path(data_path)

    try:
        data = pd.read_json(data_path)
    except:
        data = pd.read_json(data_path, lines=True) 
    
    if num_samples != 'all':
        data = data.sample(n=int(num_samples))
    
    if not (len(email_list) == 1 and email_list[0] == 'everyone'):
        data = data[data['email'].isin(email_list)]
    
    if restart_from:
        restart_from = int(restart_from)
        if restart_path is None:
            raise ValueError("restart_path must be provided when restart_from is provided")

        previous_results = pd.read_csv(restart_path)
        previous_results = previous_results.iloc[:restart_from]
        remaining_data = data.iloc[restart_from:]
        combined_data = pd.concat([previous_results, remaining_data], ignore_index=True)

        return combined_data
    
    else:
        return data
    
def save_results_to_csv(data: pd.DataFrame, setting, models, output_folder):

    output_folder = Path(output_folder)
    ensure_dir(output_folder)
    
    models_str = "_".join(models)
    output_file = output_folder / 'inference' / f"{setting}_{models_str}_results.csv"
    data.to_csv(output_file)
    print(f"Saved results to {output_file}")


def run_answer_prediction(args):

    data = filter_data_by_settings(
        args.data_path,
        args.whos,
        args.num_samples,
        args.restart_from,
        args.restart_path
    )
    
    results_path = Path(args.results_dataset_path)
    ensure_dir(results_path)

    if args.restart_from:
        restart_from=int(args.restart_from)
    else:
        restart_from=0
    print(data)
    
    if args.setting == 'mcq-answering':
        if args.evaluation_style == 'direct':
            direct_system_message = "You are given a multiple-choice question to answer. You must respond ONLY with the nulettermber corresponding to the correct answer, without any additional text or explanation. Answer with the letter directly, do not include any other word. You can answer 'A','B or 'C'."
            answers = generateAnswers_mcq(data,  
                                          args.models, 
                                          max_tokens=1, 
                                          system_message = direct_system_message, 
                                          style = args.evaluation_style,
                                          restart_from=restart_from,
                                          answer_extractor= format_answer_direct,
                                          output_folder=args.results_dataset_path)
        elif args.evaluation_style == 'CoT':
            cot_system_message = 'The following is a multiple-choice question. Think step by step and then provide your FINAL answer between the tags <ANSWER> X </ANSWER> where X is ONLY the correct letter of your choice. Do not write additional text between the tags.'
            answers = generateAnswers_mcq(data,  
                                          args.models, 
                                          max_tokens= MAX_TOKENS, 
                                          system_message = cot_system_message, 
                                          style = args.evaluation_style,
                                          restart_from=restart_from,
                                          answer_extractor= extract_answer_from_tags,
                                          output_folder=args.results_dataset_path)
        else: 
            raise ValueError(f'MCQ: Unexpected evaluation_style value: {args.evaluation_style}')
    elif args.setting == 'long-form-answering':
        answers = generateAnswers_long(data, args.models, restart_from)    
    else:
        raise ValueError(f'unexpected value {args.setting} for an inference setting')

    #save answers
    save_results_to_csv(answers, args.setting + f'_{args.evaluation_style}', args.models, args.results_dataset_path)


def main():
    args = parse_args()
    run_answer_prediction(args)


if __name__ == '__main__':
    main()
