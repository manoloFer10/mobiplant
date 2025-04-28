import pandas as pd

# Load the CSV file
df = pd.read_csv("unfolded_results.csv")

# Ensure there are exactly 9 unique journals
assert df['source_journal'].nunique() == 9, "There are not 9 unique journals."

# Step 1: Stratified sampling to get 1 question per journal
stratified_sample = df.groupby('source_journal', group_keys=False).apply(lambda x: x.sample(n=1))

# Step 2: Randomly sample remaining questions (excluding the 9 already selected)
remaining_sample = df.drop(stratified_sample.index).groupby('source_journal', group_keys=False).apply(lambda x: x.sample(n=1))

last_sample = df.drop(remaining_sample.index).sample(n=2)

# Combine and shuffle the final sample
final_sample = pd.concat([stratified_sample, remaining_sample,last_sample]).sample(frac=1).reset_index(drop=True)

final_sample.to_csv("sampled_questions.csv")