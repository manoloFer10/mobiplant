import pandas as pd
import random
import ast
import math

# Load the CSV
df = pd.read_csv("long_answer_results_everyone/merged_model_answers.csv", encoding='utf-8', engine='python')

# Filter if necessary
df = df[df['name'] == 'Anne Krapp']

# Define model columns
model_columns = ["long_answer_by_llama", 
                 "long_answer_by_gemini", 
                 "long_answer_by_claude", 
                 "long_answer_by_chatgpt",
                 "long_answer_by_o1-mini",
                 "long_answer_by_v3",
                 "long_answer_by_r1"
]

idx2word = {
    0: 'first',
    1: 'second',
    2: 'third',
    3: 'fourth',
    4: 'fifth'
}

# Create a new transformed structure
transformed_data = []

for email, group in df.groupby("email"):
    group = group.reset_index(drop=True)
    name = group.loc[0, "name"]
    
    num_questions = len(group)
    num_packs = math.ceil(num_questions / 5)  # how many packs of 5 we have
    
    # Process each pack of 5 questions
    for pack_idx in range(num_packs):
        # Create a new record for each pack
        user_data = {"email": email, "name": name, "pack": pack_idx + 1}
        
        # Determine the slice for this pack
        start_index = pack_idx * 5
        end_index = start_index + 5
        pack_questions = group.iloc[start_index:end_index].reset_index(drop=True)
        

        # Process questions in the current pack
        for i, row in pack_questions.iterrows():
            question_key = idx2word[i]  # first, second, etc.
            
            # Assign question text
            user_data[f"{question_key}_question"] = row.question
            
            # Convert options string to a list (if needed)
            options = ast.literal_eval(row.options) if isinstance(row.options, str) else row.options
            
            # Get the correct answer using the answer index
            correct_answer = str(options[int(row.answer)])
            user_data[f"{question_key}_question_answer"] = correct_answer
            
            # Shuffle model answers
            model_answers = [getattr(row, model) for model in model_columns]
            random.shuffle(model_answers)
            for j, answer in enumerate(model_answers, start=1):
                user_data[f"{question_key}_question_model{j}_answer"] = answer
        
        # Append the record for this pack to our data list
        transformed_data.append(user_data)

# Convert dictionary to DataFrame
final_df = pd.DataFrame(transformed_data)

# Save to CSV
final_df.to_csv("AnneKrapp_formatted_questions.csv", index=False)

print("Formatted CSV saved!")