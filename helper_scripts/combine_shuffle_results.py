import pandas as pd

shuffle_results=[
    ('results\inference\Full_results_cot(1).csv', 'first'),
    ('results_shuffled1\Full_results_cot.csv', 'second'),
    ('results_shuffled2\inference\Full_results_cot.csv', 'third')
]

df = pd.read_csv(shuffle_results[0][0])
shuffle_name = shuffle_results[0][1]
for col in df.columns:
    if col.startswith('CoT_reasoning_by_') or col.startswith('CoT_election_by_') or col == 'answer':
        df[f'{col}_{shuffle_name}'] = df[col]
        df = df.drop(columns=[col], axis=1)

if len(shuffle_results[0:]) > 0:
    for result, shuffle_name in shuffle_results[0:]:
        result_df = pd.read_csv(result)
        for col in result_df.columns:
            if col.startswith('CoT_reasoning_by_') or col.startswith('CoT_election_by_') or col == 'answer':
                df[f'{col}_{shuffle_name}'] = result_df[col]


df.to_csv('data/mcq_results_all_shuffles.csv', index=True)