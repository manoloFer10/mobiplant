import pandas as pd
import json
import re

def generate_code_from_name(name, pack, existing_codes):
    # Extract initials from the name
    initials = ''.join([word[0].upper() for word in re.findall(r'\b\w', name)])
    
    # Generate the base code
    base_code = f"{initials}{pack}"
    
    # Ensure the code is unique
    counter = 1
    unique_code = base_code
    while unique_code in existing_codes:
        unique_code = f"{base_code}_{counter}"
        counter += 1
    
    # Add the new code to the list of existing codes
    existing_codes.add(unique_code)
    
    return unique_code

preprocessed_for_superannotate = "AnneKrapp_formatted_questions.csv"
items_template_path = 'superannotate_work/template.csv'

data_for_annotators = pd.read_csv(preprocessed_for_superannotate)
items_csv = pd.read_csv(items_template_path)

question_numbers = ['first_question', 'second_question', 'third_question', 'fourth_question', 'fifth_question']

code2person = {}
person2code = {}
existing_codes = set()

for idx, row in data_for_annotators.iterrows():
    item = {}
    annotator_data = {}
    for col, value in row.items():
        annotator_data[str(col)] = value
    
    code = generate_code_from_name(row['name'], row['pack'], existing_codes)
    code2person[code] = row['name']
    if row['name'] in person2code:
        person2code[row['name']].append(code)
    else:
        person2code[row['name']] = [code]


    item['_item_name'] = code 

    for question_number in question_numbers:
        question_in= annotator_data[question_number]
        actual_answer = annotator_data[question_number+'_answer']
        answer_model1 = annotator_data[question_number+'_model1_answer']
        answer_model2 = annotator_data[question_number+'_model2_answer']
        answer_model3 = annotator_data[question_number+'_model3_answer']
        answer_model4 = annotator_data[question_number+'_model4_answer']
        answer_model5 = annotator_data[question_number+'_model5_answer']
        answer_model6 = annotator_data[question_number+'_model6_answer']
        answer_model7 = annotator_data[question_number+'_model7_answer']
        

        item[question_number +'_question'] = question_in
        item[question_number +'_actual_answer'] = actual_answer
        item[question_number +'_model_answers_select_model1'] = answer_model1
        item[question_number +'_model_answers_select_model2'] = answer_model2
        item[question_number +'_model_answers_select_model3'] = answer_model3
        item[question_number +'_model_answers_select_model4'] = answer_model4
        item[question_number +'_model_answers_select_model5'] = answer_model5
        item[question_number +'_model_answers_select_model6'] = answer_model6
        item[question_number +'_model_answers_select_model7'] = answer_model7

        item[question_number +'_model1_answer'] = answer_model1
        item[question_number +'_model2_answer'] = answer_model2
        item[question_number +'_model3_answer'] = answer_model3
        item[question_number +'_model4_answer'] = answer_model4
        item[question_number +'_model5_answer'] = answer_model5
        item[question_number +'_model6_answer'] = answer_model6
        item[question_number +'_model7_answer'] = answer_model7        
    
    items_csv = pd.concat([items_csv, pd.DataFrame([item])], ignore_index=True)

items_csv.to_csv('superannotate_work/AnneKrapp_prepared_items.csv', index=False)

# with open ('superannotate_work/code2person.json', 'w') as f:
#     json.dump(code2person,f)

# with open ('superannotate_work/person2code.json', 'w') as f:
#     json.dump(person2code, f)
