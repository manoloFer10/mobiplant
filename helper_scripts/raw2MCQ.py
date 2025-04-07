import pandas as pd
import json
import os
import random

# Read the CSV and process the data
def process_csv_to_json(csv_file_path):
    json_result_named = []
    json_result_unnamed = []


    raw_name2json_field ={
        'Question': 'question',
        'Area': 'area',
        'If there is a specific plant species involved in this question, please specify which below (bear in mind that it should also be stated in the question itself)': 'plant_species',
        'Source (link or DOI)': 'source'
    }

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Dynamically identify question groups based on column headers
    question_groups = {}
    for column in df.columns:
        if any(tag in column for tag in ["[First Question]", "[Second Question]", "[Third Question]", "[Fourth Question]", "[Fifth Question]"]):
            question_tag = column.split('[')[-1].rstrip(']')  # Extract "First Question", etc.
            if question_tag not in question_groups:
                question_groups[question_tag] = []
            question_groups[question_tag].append(column)

    # Iterate through rows to construct JSON objects
    for _, row in df.iterrows():
        for question_tag, columns in question_groups.items():
            question_data_named = {
                "name": row.get("Name and Surname", ""),
                "email": row.get("Email address", ""),
            }

            question_data_unnamed = {}
            options = []
            answer_counter = 0
            # Add all columns for this question tag
            for column in columns:
                column_key = column.replace(f" [{question_tag}]", "").strip()  # Clean up column name

                if 'answer' in column_key.lower():
                    if 'correct' in column_key.lower() and not 'incorrect' in column_key.lower():
                        correct_answer = row.get(column, "")
                    answer_counter += 1
                    value = row.get(column, "")
                    options.append(value)
                    if len(options) == 3:
                        random.shuffle(options)
                        correct_index = options.index(correct_answer)

                        question_data_named['options'] = options
                        question_data_named['answer'] = correct_index
                        question_data_unnamed['options'] = options
                        question_data_unnamed['answer'] = correct_index

                        answer_counter = 0 # reset counter for next question
                        options = [] # reset options list

                else:
                    json_key = raw_name2json_field[column_key]
                    value = row.get(column, "")
                    if pd.isna(value):
                        value = None
                    question_data_named[json_key] = value
                    question_data_unnamed[json_key] = value

            json_result_unnamed.append(question_data_unnamed)
            json_result_named.append(question_data_named)

    return json_result_named, json_result_unnamed

# Save JSON to a specified folder path
def save_json_to_folder(json_data, output_folder, output_filename):
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
    output_path = os.path.join(output_folder, output_filename)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, indent=2, ensure_ascii=False)
    print(f"JSON saved to {os.path.join(output_folder, output_filename)}")

csv_file_path = "data/raw_questions.csv"  
output_folder = "data" 
output_filename_named = "questionsMCQ_named.json"  
output_filename_unnamed = "questionsMCQ.json"  


questions_json_named, questions_json_unnamed = process_csv_to_json(csv_file_path)
save_json_to_folder(questions_json_named, output_folder, output_filename_named)
save_json_to_folder(questions_json_unnamed, output_folder, output_filename_unnamed)

