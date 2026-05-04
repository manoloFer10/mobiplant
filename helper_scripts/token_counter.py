import tiktoken
import json
import argparse

def count_tokens(text, model="gpt-4o"):
    """
    Count the number of tokens in a given text for a specified model.

    Args:
        text (str): The input text to tokenize.
        model (str): The model to use for determining the tokenizer. Defaults to "gpt-3.5-turbo".

    Returns:
        int: The number of tokens in the text.
    """
    try:
        # Get the tokenizer for the specified model
        encoding = tiktoken.encoding_for_model(model)

        # Encode the text and count the tokens
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--is_reasoning_model",
        type=int, 
        required=True,
        help="path of the original data",
    )
    parser.add_argument(
        "--input_cost",
        type=float,
        required=True,
        help='input token cost in $ per M tokens'
    )
    parser.add_argument(
        "--output_cost",
        type=float,
        required=True,
        help='output token cost in $ per M tokens'
    )
    args = parser.parse_args()
    USE_REASONING_MODEL = bool(args.is_reasoning_model)
    REASONING_MULTIPLIER = 5
    # Estimated visible (non-reasoning) output tokens per question when the
    # model is prompted to think step-by-step (CoT).  A short MCQ rationale
    # is typically 150-300 tokens; adjust to match your observed outputs.
    COT_TOKENS_PER_ANSWER = 250

    tokens = 0
    answer_tokens = 0
    question_tokens = 0
    input_tokens = 0
    system_message = "You are a helpful assistant that answers plant biology questions. Answer concisely in one paragraph. Return ONLY the number of the correct option (Should be an integer in {1,2,3})"
    with open('data\expert_mobi_A.json', 'r', encoding='utf-8') as f:
        question_json = json.load(f)
        length_db = len(question_json)
        for question_metadata in question_json:
            question = question_metadata['question']
            c_a = question_metadata['options'][0]
            i_a1 = question_metadata['options'][2]
            i_a2 = question_metadata['options'][1]

            text_for_counting = system_message + '' + question + '' + c_a +'' + i_a1 + '' + i_a2 + '/n'
            # CoT budget: model is asked to reason, so output is much larger
            # than the bare answer option text.
            answer_tokens += COT_TOKENS_PER_ANSWER
            question_tokens += count_tokens(system_message+'' +question)
            tokens+=count_tokens(text_for_counting)

    input_cost_per_token = args.input_cost/1000000
    output_cost_per_token = args.output_cost/1000000

    print(f"Total number of questions: {length_db}")
    print()
    print("── MCQ answering ────────────────────────────────────────")
    print(f"  Input tokens  : {tokens}")
    if USE_REASONING_MODEL:
        final_answer_tokens = answer_tokens * REASONING_MULTIPLIER
        total_output_tokens = answer_tokens + final_answer_tokens
    else: 
        final_answer_tokens = answer_tokens
        total_output_tokens = answer_tokens
    print(f"  Output tokens      : {total_output_tokens:.0f}")
    print(f"COSTS: Input = ${tokens*input_cost_per_token} USD, Output = ${total_output_tokens*output_cost_per_token}USD")
    print(f'Total cost: = ${tokens*input_cost_per_token + total_output_tokens*output_cost_per_token} USD')
    print()
    print("── Open-ended answering ─────────────────────────────────")
    print(f"  Input tokens  : {input_tokens}")
    print(f"  Output tokens (CoT completion, ~{COT_TOKENS_PER_ANSWER} tok/q): {answer_tokens:.0f}")
    if USE_REASONING_MODEL:
        reasoning_tokens_oe = answer_tokens * REASONING_MULTIPLIER
        total_output_tokens_oe = answer_tokens + reasoning_tokens_oe
        print(f"  Output tokens (reasoning, ×{REASONING_MULTIPLIER} multiplier): {reasoning_tokens_oe:.0f}")
        print(f"  Output tokens (total)     : {total_output_tokens_oe:.0f}")

