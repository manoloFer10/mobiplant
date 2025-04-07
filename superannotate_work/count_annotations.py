import pandas as pd

def count_evaluation_distribution(csv_path, output_distribution_csv, output_scoring_csv):
    """
    Reads the postprocessed CSV and counts the distribution of evaluation values for each criterion
    and each model. Then, it produces a CSV with a two-level row index (criterion, rating option)
    and columns corresponding to the original models. Additionally, it computes and saves a scoring
    CSV based on the provided rubric.

    Parameters:
        csv_path (str): Path to the postprocessed CSV file.
        output_distribution_csv (str): Path where the distribution CSV should be saved.
        output_scoring_csv (str): Path where the scoring CSV should be saved.
    """
    # Load the CSV.
    df = pd.read_csv(csv_path)
    
    # The evaluation columns have names like: "evaluation_long_answer_by_llama_hallucinated_content"
    eval_cols = [col for col in df.columns if col.startswith("evaluation_")]
    
    # Define the list of original model columns.
    models = [
        "long_answer_by_llama", 
        "long_answer_by_gemini", 
        "long_answer_by_claude", 
        "long_answer_by_chatgpt",
        "long_answer_by_o1-mini",
        "long_answer_by_v3",
        "long_answer_by_r1"
    ]
    
    # Define the evaluation criteria (suffixes in the evaluation columns).
    criteria = [
        'hallucinated_content',
        'irrelevant_content',
        'species_bias',
        'reading_comprehension',
        'source',
        'correct_reasoning',
        'limitations',
        'omission_information',
        'alignment_consensus'
    ]
    
    # Build a nested dictionary: distribution[criterion][model] = Series of counts.
    distribution = {crit: {} for crit in criteria}
    
    for col in eval_cols:
        parts = col.split("_")
        remaining = "_".join(parts[1:])
        model_found = None
        crit_found = None
        for m in models:
            if remaining.startswith(m):
                model_found = m
                crit_found = remaining[len(m)+1:]
                break
        if model_found is None or crit_found is None:
            continue
        if crit_found not in criteria:
            continue
        
        counts = df[col].value_counts(dropna=False)
        distribution[crit_found][model_found] = counts

    # Process to create distribution CSV
    dfs = []
    for crit in criteria:
        all_ratings = set()
        for m in models:
            s = distribution[crit].get(m)
            if s is not None:
                all_ratings.update(s.index.tolist())
        def norm_rating(val):
            return "NaN" if pd.isna(val) else str(val)
        all_ratings = sorted(norm_rating(x) for x in all_ratings)
        
        data = {}
        for m in models:
            s = distribution[crit].get(m)
            counts_dict = {}
            if s is not None:
                for k, count in s.items():
                    counts_dict[norm_rating(k)] = count
            col_counts = [counts_dict.get(r, 0) for r in all_ratings]
            data[m] = col_counts
        df_crit = pd.DataFrame(data, index=all_ratings)
        df_crit.index.name = "Rating"
        df_crit["Criterion"] = crit
        df_crit = df_crit.reset_index().set_index(["Criterion", "Rating"])
        dfs.append(df_crit)
    
    combined_df = pd.concat(dfs).sort_index(level=["Criterion", "Rating"])
    combined_df.to_csv(output_distribution_csv)
    print(f"Distribution saved to {output_distribution_csv}")

    # Process to create scoring CSV
    rubric_scores = {
        "alignment_consensus": {
            "Aligned to consensus": 1,
            "Opposed to consensus": 0,
        },
        "correct_reasoning": {
            "No": 0,
            "Partially": 0.5,
            "Yes": 1,
        },
        "hallucinated_content": {
            "No, it didn't hallucinate any content": 1,
            "Yes, and the provided answer is known to be wrong": 0,
            "Yes, but there is no evidence to support it or deny it": 0.5,
        },
        "irrelevant_content": {
            "No": 1,
            "Yes": 0,
        },
        "limitations": {
            "No": 0,
            "Yes": 1,
        },
        "omission_information": {
            "No": 1,
            "Yes, great biological significance": 0,
            "Yes, little biological significance": 0.5,
        },
        "reading_comprehension": {
            "No": 0,
            "Partially": 0.5,
            "Yes": 1,
        },
        "species_bias": {
            "No": 1,
            "Yes": 0,
        },
    }

    scores_dict = {crit: {model: None for model in models} for crit in rubric_scores}
    
    for col in eval_cols:
        parts = col.split("_")
        remaining = "_".join(parts[1:])
        model_found = None
        crit_found = None
        for m in models:
            if remaining.startswith(m):
                model_found = m
                crit_found = remaining[len(m)+1:]
                break
        if not model_found or crit_found not in rubric_scores:
            continue
        
        series = df[col]
        score_mapping = rubric_scores.get(crit_found, {})
        scores = series.map(score_mapping)
        mean_score = scores.mean(skipna=True) * 100
        scores_dict[crit_found][model_found] = mean_score
    
    scoring_df = pd.DataFrame.from_dict(scores_dict, orient='index')
    scoring_df = scoring_df.round(1)
    scoring_df.to_csv(output_scoring_csv)
    print(f"Scoring saved to {output_scoring_csv}")

# Example usage:
if __name__ == "__main__":
    input_csv = "superannotate_work/annotations\original_with_evaluations.csv"  # your postprocessed CSV
    distribution_csv = "criteria_distribution.csv"
    scoring_csv = "scoring.csv"
    count_evaluation_distribution(input_csv, distribution_csv, scoring_csv)