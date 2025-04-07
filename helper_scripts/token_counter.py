import tiktoken
import json

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

# Example usage
if __name__ == "__main__":
    tokens = 0
    answer_tokens = 0
    question_tokens = 0
    input_tokens = 0
    system_message = "You are a helpful assistant that answers plant biology questions. Answer concisely in one paragraph. Return ONLY the number of the correct option (Should be an integer in {1,2,3})"
    with open('data\questionsMCQ.json', 'r', encoding='utf-8') as f:
        question_json = json.load(f)
        length_db = len(question_json)
        for question_metadata in question_json:
            question = question_metadata['question']
            c_a = question_metadata['options'][0]
            i_a1 = question_metadata['options'][2]
            i_a2 = question_metadata['options'][1]

            text_for_counting = system_message + '' + question + '' + c_a +'' + i_a1 + '' + i_a2 + '/n'
            answer_tokens += (count_tokens(c_a)+ count_tokens(i_a1)+ count_tokens(i_a2))/3
            question_tokens += count_tokens(system_message+'' +question)
            tokens+=count_tokens(text_for_counting)


    print(f"Total number of input tokens for MCQ answering: {tokens}")
    print(f"Total number of input tokens for MCQ answering: {length_db}")

    print(f"Estimated total number of input tokens for answering open-ended question: {input_tokens}")
    print(f'Estimated total number of output tokens for answering open-ended question: {answer_tokens}')

