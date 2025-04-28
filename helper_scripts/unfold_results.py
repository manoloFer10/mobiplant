import json
import csv
import os

def main():
    # Define paths for input and output files
    json_path = os.path.join("data", "doi", "synth_processed_paper_data.json")
    output_csv = "unfolded_results.csv"

    # Load JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Determine metadata keys (fields other than question, options, answer)
    metadata_keys = set()
    for record in data:
        metadata_keys.update(set(record.keys()) - {"question", "options", "answer"})
    metadata_keys = list(metadata_keys)

    # Specify CSV fieldnames including metadata keys
    fieldnames = ["question", "correct answer", "incorrect_answer 1", "incorrect_answer 2"] + metadata_keys

    # Open CSV file for writing
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process each record
        for record in data:
            question = record.get("question", "")
            options = record.get("options", [])
            correct_index = record.get("answer")
            
            # Validate the record
            if correct_index is None or not options or correct_index < 0 or correct_index >= len(options):
                continue

            correct_answer = options[correct_index]
            # Collect incorrect answers (ignoring the correct one)
            incorrect_answers = [opt for i, opt in enumerate(options) if i != correct_index]
            if len(incorrect_answers) < 2:
                continue

            # Prepare row with main fields
            row = {
                "question": question,
                "correct answer": correct_answer,
                "incorrect_answer 1": incorrect_answers[0],
                "incorrect_answer 2": incorrect_answers[1]
            }

            # Add metadata fields if present, otherwise leave them empty
            for key in metadata_keys:
                row[key] = record.get(key, "")

            writer.writerow(row)

if __name__ == '__main__':
    main()