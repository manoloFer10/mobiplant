import pandas as pd

question_numbers = ['first_question', 'second_question', 'third_question', 'fourth_question', 'fifth_question']
models_numbers = ['model1', 'model2', 'model3', 'model4', 'model5', 'model6', 'model7']

evaluation_df = pd.read_csv('superannotate_work/annotations/annotations.csv')
print(f'Eval df len: {len(evaluation_df)}')
original_df  = pd.read_csv('long_answer_results_everyone\merged_model_answers.csv')
print(f'Pre-eval df len: {len(original_df)}')


# Revised criteria mapping: keys match the suffix used in the evaluation CSV.
criteria = {
    'hallucinated_content': 'hallucinated_content',
    'irrelevant_content': 'irrelevant_content',
    'species_bias': 'species_bias',
    'reading_comprehension': 'reading_comprehension',
    'source': 'source',
    'reasoning': 'correct_reasoning',
    'limitations': 'limitations',
    'omission_information': 'omission_information',
    'consensus': 'alignment_consensus'
}

# Question slots (as they appear in the evaluation CSV)
question_slots = ['first_question', 'second_question', 'third_question', 'fourth_question', 'fifth_question']

# Model slots in the evaluation CSV (should be: model1, model2, ..., model7)
model_slots = ['model1', 'model2', 'model3', 'model4', 'model5', 'model6', 'model7']

# Original model columns in the original CSV.
original_model_columns = [
    "long_answer_by_llama", 
    "long_answer_by_gemini", 
    "long_answer_by_claude", 
    "long_answer_by_chatgpt",
    "long_answer_by_o1-mini",
    "long_answer_by_v3",
    "long_answer_by_r1"
]

# For each original model and each evaluation criterion, add new columns in original_df.
for model in original_model_columns:
    for crit in criteria.values():
        colname = f"evaluation_{model}_{crit}"
        original_df[colname] = None

def normalize_text(text):
    """Helper to normalize text for comparison."""
    if pd.isna(text):
        return ""
    return str(text).strip().lower()

# Process each evaluation row.
for _, eval_row in evaluation_df.iterrows():
    # For each question slot in the evaluation pack:
    for qs in question_slots:
        eval_question = eval_row.get(qs+'_question')
        if pd.isna(eval_question):
            print('skipping eval_question')
            continue

        # Normalize evaluation question text.
        norm_eval_question = normalize_text(eval_question)

        # Find the matching original row by comparing normalized question text.
        orig_matches = original_df[original_df['question'].apply(normalize_text) == norm_eval_question]
        if orig_matches.empty:
            print(f"Warning: No match found for question:\n{eval_question}\n")
            continue

        orig_index = orig_matches.index[0]

        # For each anonymized answer slot:
        for slot in model_slots:
            # Build evaluation answer column name.
            eval_answer_col = f"{qs}_{slot}_answer"
            if eval_answer_col not in evaluation_df.columns:
                print(f"Warning: Column {eval_answer_col} not found in evaluation CSV.")
                continue

            eval_answer = eval_row.get(eval_answer_col)
            if pd.isna(eval_answer):
                print('skipping eval_answer')

                continue

            # Normalize the evaluation answer.
            norm_eval_answer = normalize_text(eval_answer)

            matched_model = None
            # Check each original model answer in the matched original row.
            for orig_model in original_model_columns:
                orig_answer = original_df.at[orig_index, orig_model]
                if normalize_text(orig_answer) == norm_eval_answer:
                    matched_model = orig_model
                    break

            if not matched_model:
                print(f"Warning: No matching answer found for question:\n{eval_question}\n in slot {slot}.")
                continue

            # Transfer evaluation criteria from the evaluation row to the corresponding new columns in original_df.
            for crit_suffix, crit_label in criteria.items():
                eval_col = f"{qs}_{slot}_evaluation_{crit_suffix}"
                if eval_col not in evaluation_df.columns:
                    print(f"Warning: Column {eval_col} not found in evaluation CSV.")
                    continue
                eval_value = eval_row.get(eval_col)
                target_col = f"evaluation_{matched_model}_{crit_label}"
                original_df.at[orig_index, target_col] = eval_value

# Save the updated original dataframe.
original_df.to_csv("original_with_evaluations.csv", index=False)