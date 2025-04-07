#esto debería ser otro pipeline

import pandas as pd
import argparse

def clean_dataset(df):
    def to_int(x):
        try:
            return int(float(x))
        except Exception:
            return x

    # Clean answer columns for each run.
    run_suffixes = ['_first', '_second', '_third']
    for suffix in run_suffixes:
        answer_col = 'answer' + suffix
        if answer_col in df.columns:
            df[answer_col] = df[answer_col].apply(to_int)

    # Clean model prediction columns.
    model_columns = [col for col in df.columns if col.startswith('CoT_election_by_')]
    for col in model_columns:
        df[col] = df[col].apply(to_int)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].astype(pd.Int64Dtype()) 

    return df

def main():
    parser = argparse.ArgumentParser(description="Clean dataset for evaluation")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("output_file", help="Path to the output cleaned CSV file")
    args = parser.parse_args()

    # Load the dataset.
    df = pd.read_csv(args.input_file)
    
    # Clean the dataset.
    df_cleaned = clean_dataset(df)
    
    # Save the cleaned dataset.
    df_cleaned.to_csv(args.output_file, index=False)
    print(f"Cleaned dataset saved to {args.output_file}")

if __name__ == '__main__':
    main()