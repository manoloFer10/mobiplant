import pandas as pd
import random

def shuffle_options_in_row(row: pd.Series) -> pd.Series:
    """Shuffle options and update answer index while preserving other columns"""
    new_row = row.copy()
    
    original_options = new_row['options']
    correct_answer_value = original_options[new_row['answer']]
    
    shuffled_options = original_options.copy()
    random.shuffle(shuffled_options)
    
    new_answer_index = shuffled_options.index(correct_answer_value)
    
    new_row['options'] = shuffled_options
    new_row['answer'] = new_answer_index
    
    return new_row


input_path = 'data\questionsMCQ_named.json'
original_df = pd.read_json(input_path)

#First shuffle
shuffled_df = original_df.apply(shuffle_options_in_row, axis=1)
output_path = input_path.replace('.json', '_shuffled1.json')
shuffled_df.to_json(output_path, orient='records', indent=2)
print(f"First shuffle 1 saved to {output_path}")

#Second shuffle
shuffled_df = original_df.apply(shuffle_options_in_row, axis=1)
output_path = input_path.replace('.json', '_shuffled2.json')
shuffled_df.to_json(output_path, orient='records', indent=2)
print(f"Second shuffle data 1 saved to {output_path}")
