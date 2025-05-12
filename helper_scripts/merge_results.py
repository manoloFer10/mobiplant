import pandas as pd

results=[
    'synth_results\inference\mcq-answering_CoT_chatgpt_results.json',
    'synth_results\inference\mcq-answering_CoT_claude_results.json',
    'synth_results\inference\mcq-answering_CoT_gemini_results.json',
    'synth_results\inference\mcq-answering_CoT_v3_results.json',
    'synth_results\inference\mcq-answering_CoT_llama_results.json',
    'synth_results\inference\mcq-answering_CoT_o1-mini_results.json',
    'synth_results\inference\mcq-answering_CoT_r1_results.json'
]

df = pd.read_json(results[0])

if len(results[0:]) > 0:
    for result in results[0:]:
        result_df = pd.read_json(result)
        for col in result_df.columns:
            if col.startswith('CoT_reasoning_by_'): model_reasoning_column = col
            if col.startswith('CoT_election_by_'): model_election_column = col

        df[model_reasoning_column] = result_df[model_reasoning_column]
        df[model_election_column] = result_df[model_election_column]

df.to_json('synth_results/full_results_cot.json', indent=2, orient='records')