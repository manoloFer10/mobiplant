import pandas as pd

results=[
    'results_shuffled1\inference\mcq-answering_CoT_chatgpt_results.csv',
    'results_shuffled1\inference\mcq-answering_CoT_claude_results.csv',
    'results_shuffled1\inference\mcq-answering_CoT_gemini_results.csv',
    'results_shuffled1\inference\mcq-answering_CoT_v3_results.csv',
    'results_shuffled1\inference\mcq-answering_CoT_llama_results.csv',
    'results_shuffled1\inference\mcq-answering_CoT_o1-mini_results.csv',
    'results_shuffled1\inference\mcq-answering_CoT_r1_results.csv'
]

df = pd.read_csv(results[0])

if len(results[0:]) > 0:
    for result in results[0:]:
        result_df = pd.read_csv(result)
        for col in result_df.columns:
            if col.startswith('CoT_reasoning_by_'): model_reasoning_column = col
            if col.startswith('CoT_election_by_'): model_election_column = col

        df[model_reasoning_column] = result_df[model_reasoning_column]
        df[model_election_column] = result_df[model_election_column]

df.to_csv('results_shuffled1/Full_results_cot.csv', index=True)