import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

EVALUATION_STYLES = ['complete', 'automatic-metrics', 'human-metrics', 'statistics', 'experiments', 'automatic-plotting', 'human-plotting']


def perform_complete_evaluation(df_dataset, output_folder):

    perform_metrics(df_dataset, output_folder)
    perform_descriptive_statistics(df_dataset)
    print('not implemented yet: perform_experiments(df_dataset)')

def perform_metrics(df_dataset, output_folder):

    output_folder = output_folder / 'metrics'
    os.makedirs(output_folder, exist_ok=True)

    model_columns = [col for col in df_dataset.columns if col.startswith('CoT_election_by_')]
    model_names = [col.replace('election_by_', '') for col in model_columns]
    df_dataset.rename(columns=dict(zip(model_columns, model_names)), inplace=True)

    # Preprocess model columns: convert to numeric and then to integers, invalid -> NaN
    for model in model_names:
        # Convert to numeric, coercing errors to NaN
        df_dataset[model] = pd.to_numeric(df_dataset[model], errors='coerce')
        # Convert to integer (NaN remains NaN)
        df_dataset[model] = df_dataset[model].astype(pd.Int64Dtype())  # Allows integer NaN

    # Group by plant_species and calculate accuracies
    group_by_and_score(df_dataset, 'normalized_plant_species', model_names, output_folder)

    # Group by area and calculate accuracies
    group_by_and_score(df_dataset, 'normalized_area', model_names, output_folder)

    # Calculate accuracies by area and plant species for each model.
    # for model in model_names:
    #     accuracy_df = df_dataset[model] == df_dataset['answer']

    #     accuracies = (
    #         accuracy_df.groupby([df_dataset['area'], df_dataset['plant_species']])
    #         .mean()
    #         .unstack(fill_value=0) 
    #     )
    #     os.makedirs(output_folder, exist_ok=True)
    #     file_name = model + "_accuracy_species&area.csv"
    #     output_file = os.path.join(output_folder, file_name)
    #     accuracies.to_csv(output_file)

    # Add Year and Citation binning
    df_dataset = create_year_bins(df_dataset)
    group_by_and_score(df_dataset, 'year_bin', model_names, output_folder)
    
    df_dataset = create_citation_bins(df_dataset)
    group_by_and_score(df_dataset, 'citation_bin', model_names, output_folder)

    create_answer_distribution_csv(df_dataset, model_names, output_folder / 'answer_distribution')
        
    print(f"Metrics results saved to: {output_folder}")

def group_by_and_score(df_dataset, group, model_names, output_folder):
    VALID_VALUES = {0, 1, 2, 3}
    results = {}

    df_group = df_dataset.copy()
    df_group[group] = df_group[group].apply(lambda x: x if isinstance(x, list) else [x])
    df_group = df_group.explode(group)

    # Group by the exploded column
    for grp, subset in df_group.groupby(group):
        metrics = {}
        total = len(subset)
        for model in model_names:
            suffix = model.split('_')[-1]
            answer_col = f'answer_{suffix}'
            
            valid_mask = subset[model].isin(VALID_VALUES)
            valid_count = valid_mask.sum()
            error_count = total - valid_count

            correct_count = 0
            if valid_count > 0:
                correct_count = (subset.loc[valid_mask, model] == subset.loc[valid_mask, answer_col]).sum()
                answer_accuracy = round(correct_count * 100 / valid_count, 1)
            else:
                answer_accuracy = np.nan

            error_rate = round(error_count * 100 / total, 1)
            total_accuracy = round(correct_count * 100 / total, 1)

            metrics[f'{model}_total_accuracy'] = total_accuracy
            metrics[f'{model}_answer_accuracy'] = answer_accuracy
            metrics[f'{model}_error_rate'] = error_rate
        
        metrics['count'] = total

        results[grp] = metrics

    # Overall metrics (use original DataFrame)
    overall_metrics = {}
    total_overall = len(df_dataset)
    for model in model_names:
        suffix = model.split('_')[-1]
        answer_col = f'answer_{suffix}'
        
        valid_mask = df_dataset[model].isin(VALID_VALUES)
        valid_count = valid_mask.sum()
        error_count = total_overall - valid_count

        correct_count = 0
        if valid_count > 0:
            correct_count = (df_dataset.loc[valid_mask, model] == df_dataset.loc[valid_mask, answer_col]).sum()
            answer_accuracy = round(correct_count * 100 / valid_count, 1)
        else:
            answer_accuracy = np.nan

        error_rate = round(error_count * 100 / total_overall, 1)
        total_accuracy = round(correct_count * 100 / total_overall, 1)

        overall_metrics[f'{model}_total_accuracy'] = total_accuracy
        overall_metrics[f'{model}_answer_accuracy'] = answer_accuracy
        overall_metrics[f'{model}_error_rate'] = error_rate

    results['Overall'] = overall_metrics

    # Convert results to DataFrame and aggregate
    results_df = pd.DataFrame(results).T
    base_models = list({model.rsplit('_', 1)[0] for model in model_names})
    
    for base_model in base_models:
        for metric in ['total_accuracy', 'answer_accuracy', 'error_rate']:
            run_columns = [f"{base_model}_{suffix}_{metric}" for suffix in ['first', 'second', 'third']]
            if all(col in results_df.columns for col in run_columns):
                mean_values = results_df[run_columns].mean(axis=1).round(1)
                std_values = results_df[run_columns].std(axis=1).round(1)
                results_df[f"{base_model}_{metric}_mean"] = mean_values
                results_df[f"{base_model}_{metric}_std"] = std_values
                results_df = results_df.drop(columns=run_columns)

    # Save results to CSVs
    group_folder = os.path.join(output_folder, group)
    os.makedirs(group_folder, exist_ok=True)
    
    # Split into separate DataFrames for each metric type
    metric_types = ['total_accuracy', 'answer_accuracy', 'error_rate']
    for metric in metric_types:
        cols = [col for col in results_df.columns if col.endswith(f'{metric}_mean') or col == 'count']
        if cols:
            df_metric = results_df[cols].copy()
            df_metric.columns = [col.replace('_mean', '') for col in df_metric.columns]
            df_metric.to_csv(os.path.join(group_folder, f"{metric}.csv"), index=True)
    
    # Save all results
    results_df.to_csv(os.path.join(group_folder, "all_results.csv"), index=True)


def create_year_bins(df):

    df = df.copy()
    
    # Convert and clean year data
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    
    bin_len = 4

    if not (2024-1994) % bin_len == 0:
        bins = list(range(1994, 2025, bin_len))
        labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
        bins.append(2024)
        labels.append(f"{bins[-2]}-{bins[-1]}")
    else:
        bins = list(range(1994, 2025, bin_len))
        labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-2)]
        labels.append(f"{bins[-2]}-{bins[-1]}")

    
    # Apply binning
    df['year_bin'] = pd.cut(
        df['Year'],
        bins=bins,
        labels=labels,
        right=False,  # [1994,1996) -> 1994-1995
        include_lowest=True
    ).astype(str)
    
    # Filter any residual invalid entries
    valid_bins = set(labels)
    df = df[df['year_bin'].isin(valid_bins)]
    
    return df

def create_citation_bins(df):
    df = df.copy()
    df['Citations'] = pd.to_numeric(df['Citations'], errors='coerce')
    df = df.dropna(subset=['Citations'])
    
    # Define exact bin edges and labels
    bin_edges = [-np.inf, 0, 10, 100, 500, 1000, 1702]
    labels = [
        "0",
        "1-10",
        "11-100",
        "101-500",
        "501-1000",
        "1001-1702"
    ]
    
    df['citation_bin'] = pd.cut(
        df['Citations'],
        bins=bin_edges,
        labels=labels,
        include_lowest=False,
        ordered=False
    ).astype(str)  # Convert categorical to string

    return df

def create_answer_distribution_csv(df_dataset, model_names, output_folder):
    """
    Create an answer distribution CSV using three-run logic.
    
    For each run, this function counts the occurrences of each answer option after mapping:
      0 -> A, 1 -> B, 2 -> C, and any other value -> Format Error.
    
    The CSV is structured such that:
      - Rows: "Ground Truth" (from the corresponding answer columns) and each model (base name).
      - Columns: For each run, separate columns for each option. For example, for the first run:
          A_first, B_first, C_first, Format Error_first
        and similarly for _second and _third.
    
    Parameters:
      df_dataset (pd.DataFrame): The dataset containing the answer and model prediction columns.
      model_names (list of str): List of base model names (without run suffixes).
      output_folder (str): Folder path where the CSV file will be saved.
    """
    # Define run suffixes and the expected letter mapping.
    run_suffixes = ['_first', '_second', '_third']
    letters = ["A", "B", "C", "Format Error"]
    mapping = {0: "A", 1: "B", 2: "C"}  # Other values will map to "Format Error".
    
    # Create an ordered list of output column names.
    output_columns = []
    for suffix in run_suffixes:
        for letter in letters:
            # The column name format: e.g., "A_first", "B_first", etc.
            output_columns.append(f"{letter}{suffix}")
    
    # Helper: Compute distribution counts for a series.
    def compute_counts(series, suffix):
        # Initialize counts for each expected option.
        counts = {letter: 0 for letter in letters}
        for x in series:
            try:
                # Try to interpret the value as an integer.
                val = int(x)
            except (ValueError, TypeError):
                val = None
            # Map the value to a letter if possible, otherwise "Format Error".
            letter = mapping.get(val, "Format Error")
            counts[letter] += 1
        # Create a dict with keys matching the output columns for this run.
        return {f"{letter}{suffix}": counts[letter] for letter in letters}
    
    # Prepare a dictionary to collect row data.
    # Each key is a row label ("Ground Truth" or a model base name)
    # and each value is a dictionary of column counts.
    distribution_results = {}
    
    # Process Ground Truth: use the answer columns.
    gt_counts = {}
    for suffix in run_suffixes:
        answer_col = 'answer' + suffix
        if answer_col in df_dataset.columns:
            gt_counts.update(compute_counts(df_dataset[answer_col], suffix))
        else:
            # If the answer column is missing, mark counts as "N/A".
            for letter in letters:
                gt_counts[f"{letter}{suffix}"] = "N/A"
    distribution_results["Ground Truth"] = gt_counts
    
    # Process each model prediction: each base model uses columns like base+suffix.
    for base in model_names:
        model_counts = {}
        for suffix in run_suffixes:
            model_col = base + suffix
            if model_col in df_dataset.columns:
                model_counts.update(compute_counts(df_dataset[model_col], suffix))
            else:
                for letter in letters:
                    model_counts[f"{letter}{suffix}"] = "N/A"
        distribution_results[base] = model_counts
    
    # Create a DataFrame with rows as distribution_results and ensure column order.
    df_out = pd.DataFrame.from_dict(distribution_results, orient='index')
    df_out = df_out.reindex(columns=output_columns)
    
    # Ensure output folder exists.
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, "answer_balance.csv")
    df_out.to_csv(csv_path, index=True)
    print(f"Answer distribution CSV saved to: {csv_path}")

def perform_human_metrics(df_dataset, output_folder):

    output_folder = output_folder / 'human_metrics'
    os.makedirs(output_folder, exist_ok=True)

    eval_cols = [col for col in df_dataset.columns if col.startswith("evaluation_")]

    models = [
        "long_answer_by_llama", 
        "long_answer_by_gemini", 
        "long_answer_by_claude", 
        "long_answer_by_chatgpt",
        "long_answer_by_o1-mini",
        "long_answer_by_v3",
        "long_answer_by_r1"
    ]
    
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
        
        counts = df_dataset[col].value_counts(dropna=False)
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
    
    output_distribution_csv = output_folder / 'criteria_distribution.csv'
    combined_df = pd.concat(dfs).sort_index(level=["Criterion", "Rating"])
    combined_df.to_csv(output_distribution_csv)
    print(f"Distribution saved to {output_distribution_csv}")

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
        
        series = df_dataset[col]
        score_mapping = rubric_scores.get(crit_found, {})
        scores = series.map(score_mapping)
        mean_score = scores.mean(skipna=True) * 100
        scores_dict[crit_found][model_found] = mean_score
    
    output_scoring_csv = output_folder / 'scoring.csv'
    scoring_df = pd.DataFrame.from_dict(scores_dict, orient='index')
    scoring_df = scoring_df.round(1)
    scoring_df.to_csv(output_scoring_csv)
    print(f"Scoring saved to {output_scoring_csv}")


def perform_descriptive_statistics(df_dataset, output_folder):

    output_folder = output_folder / 'stats'
    os.makedirs(output_folder, exist_ok=True)

    # Frequency Tables
    categorical_fields = ['normalized_area', 'normalized_plant_species'] 
    for field in categorical_fields:
        if field in df_dataset.columns:
            freq_table = df_dataset[field].value_counts().reset_index()
            freq_table.columns = [field, 'counts']
            freq_table['proportion'] = freq_table['counts'] / freq_table['counts'].sum()
            freq_table = freq_table.round(2).astype(str)
            
            freq_table.to_csv(output_folder / f"{field}_frequency.csv", index=False)


    print(f"Overall statistics saved to folder: {output_folder}")

def calculate_distribution(df, column_name):
    """Calculate the distribution and proportion of answers in a given column."""
    distribution = df[column_name].value_counts().reset_index()
    distribution.columns = ['answer', f'counts {column_name}']
    distribution = distribution.set_index('answer')
    distribution[f'proportion {column_name}'] = distribution[f'counts {column_name}'] / distribution[f'counts {column_name}'].sum()
    distribution = distribution.round(2).astype(str)
    return distribution


def perform_experiments(df_dataset):
    raise NotImplementedError

def perform_automatic_plots(df_dataset, output_folder):
    origin_folder = output_folder
    output_folder = output_folder / 'plots'
    os.makedirs(output_folder, exist_ok=True)

    #Spider graph; model accuracy by lang
    if os.path.exists(origin_folder / 'metrics'):
        origin_folder = origin_folder / 'metrics'
        plot_citation_bin_accuracy(origin_folder / 'citation_bin' / 'answer_accuracy.csv', output_folder)
        
        bin_df = create_year_bins(df_dataset)
        plot_year_accuracy(origin_folder / 'year_bin' / 'answer_accuracy.csv', bin_df['year_bin'].value_counts(), output_folder)
    else:
        print('No metrics results folder detected... passing to statistics plots.')


def plot_citation_bin_accuracy(data_path, output_folder):
    # Read and prepare data
    df = pd.read_csv(data_path, index_col=0)
    df = df.drop('Overall', errors='ignore')
    
    # Exclude the 'Overall' row and sort bins logically
    df = df.drop('Overall', errors='ignore')
    bin_order = ['0', '1-10', '11-100', '101-500', '501-1000', '1001-1702']
    df = df.reindex(bin_order)
    
    # Calculate mean accuracy across models
    df['mean_accuracy'] = df.mean(axis=1)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df.index, df['mean_accuracy'], color='skyblue')
    
    # Add labels and formatting
    plt.title('Mean Model Accuracy by Number of Citations', fontsize=14)
    plt.xlabel('Number of citations', fontsize=12)
    plt.ylabel('Mean Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom')
    
    # Rotate x-labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout and save/show
    plt.tight_layout()
    plt.savefig(output_folder / 'citations.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_year_accuracy(data_path, sample_counts, output_folder):
    # Read and prepare data
    df = pd.read_csv(data_path, index_col=0)
    df = df.drop('Overall', errors='ignore')
    
    sorted_bins = sorted(df.index, key=lambda x: int(x.split('-')[0]))
    df = df.loc[sorted_bins]
    
    # Calculate mean accuracy
    df['mean_accuracy'] = df.mean(axis=1)
    
    # Align sample counts with the dataframe
    sample_counts = sample_counts.reindex(sorted_bins).fillna(0)
    
    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Create scatter plot with connected lines (left axis)
    x = range(len(df.index))
    ax1.scatter(x, df['mean_accuracy'], color='darkblue', s=100, zorder=3)
    ax1.plot(x, df['mean_accuracy'], linestyle=':', color='gray', alpha=0.7)
    ax1.set_ylabel('Mean Accuracy (%)', fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)
    
    # Create bar plot for sample counts (right axis)
    ax2 = ax1.twinx()
    bars = ax2.bar(x, sample_counts, color='lightblue', alpha=0.6, label='Sample Count')
    ax2.set_ylabel('Number of Samples', fontsize=12)
    
    # Configure axes and labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(df.index, rotation=45, ha='right')
    ax1.set_title('Model Accuracy and Sample Distribution by Year', fontsize=14)
    ax1.set_xlabel('Year Range', fontsize=12)
    
    # Add value labels
    for i, acc in enumerate(df['mean_accuracy']):
        ax1.text(x[i], acc + 1, f'{acc:.1f}%', 
                ha='center', va='bottom', fontsize=9)
        
    # Add sample count labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}',
                     ha='center', va='bottom', fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(output_folder / 'years.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_overall_spidergraph(data_path: str, output_folder: str):
    # Read the CSV file with overall metrics (index column is assumed to contain group names)
    df = pd.read_csv(data_path, index_col=0)
    
    # Select only the overall metrics row
    if 'Overall' not in df.index:
        raise ValueError("Overall metrics row not found in the data.")
    overall = df.loc['Overall']
    
    # Pick the columns corresponding to total accuracy for each model
    # They are assumed to have the suffix '_answer_accuracy'
    accuracy_cols = [col for col in overall.index if col.endswith('_answer_accuracy')]
    # Extract the model names by removing the suffix
    model_names = [col.replace('_answer_accuracy', '') for col in accuracy_cols]
    # Retrieve overall accuracies (as numbers)
    overall_values = [overall[col] for col in accuracy_cols]
    
    # Prepare the angles for each axis on the spider chart
    num_models = len(model_names)
    angles = np.linspace(0, 2 * np.pi, num_models, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    # Prepare values in the same order and complete the loop for the radar plot
    values = overall_values[:]
    values += values[:1]
    
    # Create the spider (radar) chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set radial limits and labels
    ax.set_rlim(0, 100)
    ax.set_rticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=10, color='grey')
    
    # Set the model names as labels on the outer circle
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(model_names, fontsize=14, color='black')
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    
    # Choose a color (or use a colormap if desired)
    color = plt.cm.tab10.colors[0]
    ax.plot(angles, values, color=color, linewidth=2, marker='o', markersize=4, label='Overall Accuracy')
    ax.fill(angles, values, color=color, alpha=0.2)
    
    # Add legend and save the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=14, frameon=False)
    plt.tight_layout()
    
    output_path = os.path.join(output_folder, "overall_accuracy_spider.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Spider chart of overall accuracy saved to: {output_path}")