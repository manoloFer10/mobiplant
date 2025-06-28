import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import squarify
import plotly.express as px
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib import patheffects
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
from scipy.stats import mannwhitneyu


EVALUATION_STYLES = [
    'automatic-metrics', 
    'human-metrics', 
    'statistics', 
    'statistics-plotting', 
    'experiments', 
    'automatic-plotting', 
    'human-plotting'
]


# def perform_complete_evaluation(df_dataset, output_folder):

#     perform_metrics(df_dataset, output_folder)
#     perform_descriptive_statistics(df_dataset)
#     print('not implemented yet: perform_experiments(df_dataset)')

def perform_metrics(df_dataset, output_folder):

    output_folder = output_folder / 'metrics'
    os.makedirs(output_folder, exist_ok=True)

    model_names_mapping = {
        'llama': 'LLaMA',
        'gemini': 'Gemini',
        'claude': 'Claude',
        'chatgpt': 'GPT-4o',
        'o1-mini': 'O1-mini',
        'v3': 'DeepSeek V3',
        'r1': 'DeepSeek R1'
    }

    model_columns = [col for col in df_dataset.columns if col.startswith('CoT_election_by_')]
    try:
        
        model_names = [model_names_mapping[col.replace('CoT_election_by_', '')] for col in model_columns]
        scoring = 'synth'
        print('Synthetic Dataset detected. Not scoring for shuffles.')

    except KeyError as e:
        print('Expert Dataset detected. Scoring for shuffles.')
        scoring = 'expert'
        model_names = [col.replace('CoT_election_by_', '') for col in model_columns]
        clean_names = []
        for model in model_names:
            suffix = model.split('_')[1]
            name = model.split('_')[0]
            clean_name = f'{model_names_mapping[name]}_{suffix}'
            clean_names.append(clean_name)
        model_names = clean_names

    df_dataset.rename(columns=dict(zip(model_columns, model_names)), inplace=True)

    # Preprocess model columns: convert to numeric and then to integers, invalid -> NaN
    for model in model_names:
        # Convert to numeric, coercing errors to NaN
        df_dataset[model] = pd.to_numeric(df_dataset[model], errors='coerce')
        # Convert to integer (NaN remains NaN)
        df_dataset[model] = df_dataset[model].astype(pd.Int64Dtype())  # Allows integer NaN

    # Group by plant_species and calculate accuracies
    group_by_and_score(df_dataset, 'plant_species', model_names, output_folder, scoring)
    group_by_and_score(df_dataset, 'normalized_plant_species', model_names, output_folder, scoring)
    #group_by_and_score(df_dataset, 'source_journal', model_names, output_folder, scoring)
    # Group by area and calculate accuracies
    group_by_and_score(df_dataset, 'area', model_names, output_folder, scoring)
    try:
        group_by_and_score(df_dataset, 'normalized_area', model_names, output_folder, scoring)
    except Exception as e:
        print(f"Error processing normalized_area: {e}. Synth results do not have this column!")

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
    df_dataset_year_binned = create_year_bins(df_dataset)
    group_by_and_score(df_dataset_year_binned, 'year_bin', model_names, output_folder, scoring)
    
    df_dataset_citation_binned = create_citation_bins(df_dataset)
    group_by_and_score(df_dataset_citation_binned, 'citation_bin', model_names, output_folder, scoring)

    create_answer_distribution_csv(df_dataset, model_names, output_folder / 'answer_distribution')
        
    print(f"Metrics results saved to: {output_folder}")

def group_by_and_score(df_dataset, group, model_names, output_folder, scoring):
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
            if scoring == 'expert':
                suffix = model.split('_')[-1]
                answer_col = f'answer_{suffix}'
            elif scoring == 'synth':
                answer_col = f'answer'
            else: 
                raise ValueError(f"Unknown scoring type: {scoring}")
            
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
        if scoring == 'expert':
            suffix = model.split('_')[-1]
            answer_col = f'answer_{suffix}'
        elif scoring == 'synth':
            answer_col = 'answer'
        else: 
            raise ValueError(f"Unknown scoring type: {scoring}")

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
    overall_metrics['count'] = total_overall
    results['Overall'] = overall_metrics

    # Convert results to DataFrame and aggregate
    results_df = pd.DataFrame(results).T
    base_models = list({model.rsplit('_', 1)[0] for model in model_names})
    
    for base_model in base_models:
        for metric in ['total_accuracy', 'answer_accuracy', 'error_rate']:
            if scoring == 'expert':
                run_columns = [f"{base_model}_{suffix}_{metric}" for suffix in ['first', 'second', 'third']]
            elif scoring == 'synth':
                run_columns = [f"{base_model}_{metric}"]
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
        bins.append(2026)
        labels.append(f"{bins[-2]}-{bins[-1]-1}")
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
    # valid_bins = set(labels)
    # df = df[df['year_bin'].isin(valid_bins)]
    
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
      - Columns: separate columns for each option.
    
    Parameters:
      df_dataset (pd.DataFrame): The dataset containing the answer and model prediction columns.
      model_names (list of str): List of base model names (without run suffixes).
      output_folder (str): Folder path where the CSV file will be saved.
    """
    print(df_dataset)
    # Define run suffixes and the expected letter mapping.
    run_suffixes = ['_first', '_second', '_third']
    letters = ["A", "B", "C", "Format Error"]
    mapping = {0: "A", 1: "B", 2: "C"}  # Other values will map to "Format Error".
    
    # Create an ordered list of output column names.
    output_columns = []
    for letter in letters:
        output_columns.append(f"{letter}")
    
    # Helper: Compute distribution counts for a series.
    def compute_counts(series):
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
        return {f"{letter}": counts[letter] for letter in letters}
    
    # Prepare a dictionary to collect row data.
    # Each key is a row label ("Ground Truth" or a model base name)
    # and each value is a dictionary of column counts.
    distribution_results = {}
    
    # Process Ground Truth: use the answer columns.
    for suffix in run_suffixes:
        gt_counts = {}
        answer_col = 'answer' + suffix
        if answer_col in df_dataset.columns:
            gt_counts.update(compute_counts(df_dataset[answer_col]))
        else:
            if suffix == '_first':
                answer_col = 'answer'  # Fallback for first run.
                gt_counts.update(compute_counts(df_dataset[answer_col]))
            else:
                # If the answer column is missing, mark counts as "N/A".
                for letter in letters:
                    gt_counts[f"{letter}"] = "N/A"
        distribution_results[f"ground_truth{suffix}"] = gt_counts
    
    # Process each model prediction: each base model uses columns like base+suffix.
    for base in model_names:
        model_counts = {}
        model_col = base 
        suffix = model_col.split('_')[-1]
        if model_col in df_dataset.columns:
            model_counts.update(compute_counts(df_dataset[model_col]))
        else:
            for letter in letters:
                model_counts[f"{letter}"] = "N/A"
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
    categorical_fields = ['area', 'plant_species','normalized_area', 'normalized_plant_species'] 
    for field in categorical_fields:
        try:
            if field in df_dataset.columns:
                freq_table = df_dataset[field].value_counts().reset_index()
                freq_table.columns = [field, 'counts']
                freq_table['proportion'] = freq_table['counts'] / freq_table['counts'].sum()
                freq_table = freq_table.round(2).astype(str)
                
                freq_table.to_csv(output_folder / f"{field}_frequency.csv", index=False)
        except Exception as e:
            print(f"Error processing {field}: {e}")


    print(f"Overall statistics saved to folder: {output_folder}")


def calculate_distribution(df, column_name):
    """Calculate the distribution and proportion of answers in a given column."""
    distribution = df[column_name].value_counts().reset_index()
    distribution.columns = ['answer', f'counts {column_name}']
    distribution = distribution.set_index('answer')
    distribution[f'proportion {column_name}'] = distribution[f'counts {column_name}'] / distribution[f'counts {column_name}'].sum()
    distribution = distribution.round(2).astype(str)
    return distribution


def get_colors(vals):
    """Map a sequence of values to our green gradient."""
    norm = mpl.colors.Normalize(vmin=min(vals), vmax=max(vals))
    return [greens_cmap(norm(v)) for v in vals]

def plot_citation_distribution_by_journal_type(df_dataset, output_folder):
    """Generates boxplots for citation counts of review vs. paper journals."""
    
    df = df_dataset.copy()
    
    # Ensure 'Citations' is numeric and drop any NaNs that result
    df['Citations'] = pd.to_numeric(df['Citations'], errors='coerce')
    df.dropna(subset=['Citations'], inplace=True)

    # Define journal types
    review_journals = ['TIPS', 'COPB']
    
    # Separate citation data
    citations_reviews = df[df['source_journal'].isin(review_journals)]['Citations']
    citations_papers = df[~df['source_journal'].isin(review_journals)]['Citations']
    
    # Perform Mann-Whitney U test to compare distributions
    if not citations_reviews.empty and not citations_papers.empty:
        u_statistic, p_value = mannwhitneyu(citations_reviews, citations_papers, alternative='two-sided')
        print("\n--- Mann-Whitney U Test: Citations (Reviews vs. Papers) ---")
        print(f"U-statistic: {u_statistic:.2f}")
        print(f"P-value: {p_value:.4g}")
        if p_value < 0.05:
            print("Result: The p-value is less than 0.05.")
        else:
            print("Result: The p-value is not less than 0.05.")
        print("-----------------------------------------------------------\n")

    data_to_plot = [citations_papers, citations_reviews]
    labels = ['Paper Journals', 'Review Journals']

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    
    box = ax.boxplot(data_to_plot, 
                     patch_artist=True,
                     labels=labels)
    
    # Style the boxplots with green colors
    colors = [greens_cmap(0.4), greens_cmap(0.8)]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)

    # Customize plot
    ax.set_yscale('log')
    ax.set_title('Citation Distribution by Journal Type', fontsize=14, weight='bold', pad=20)
    ax.set_ylabel('Number of Citations (log scale)', fontsize=12, weight='semibold')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    plt.tight_layout()
    
    # Save outputs
    for ext in ['png', 'svg']:
        plt.savefig(output_folder / f"citation_distribution_by_journal_type.{ext}", format=ext, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Citation distribution boxplot saved to: {output_folder / 'citation_distribution_by_journal_type.png'}")


def perform_statistics_plots(df_dataset, output_folder):
    stats_folder = Path(output_folder) / 'plots' / 'statistics'
    stats_folder.mkdir(parents=True, exist_ok=True)

    # 1. Questions by Publication Year Bin
    df_year = create_year_bins(df_dataset)
    counts = df_year['year_bin'].value_counts().sort_index()
    vals = counts.values
    colors = get_colors(vals)

    plt.figure(figsize=(8, 4))
    bars = plt.bar(counts.index.astype(str), vals, color=colors, edgecolor='black')
    plt.title('Questions by Publication Year Bin')
    plt.xlabel('Year Range')
    plt.ylabel('Number of Questions')
    plt.xticks(rotation=45, ha='right')
    for bar, cnt in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width()/2, cnt, str(cnt),
                 ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(stats_folder / 'questions_by_year_bin.png', dpi=300, bbox_inches='tight')
    plt.savefig(stats_folder / 'questions_by_year_bin.svg', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Year Bin stats saved to {stats_folder / 'questions_by_year_bin.png'}")

    # 2. Questions by Citation Count Bin
    df_cit = create_citation_bins(df_dataset)
    bin_order = ["0", "1-10", "11-100", "101-500", "501-1000", "1001-1702"]
    counts = df_cit['citation_bin'].value_counts().reindex(bin_order).fillna(0)
    vals = counts.values
    colors = get_colors(vals)

    plt.figure(figsize=(8, 4))
    bars = plt.bar(counts.index, vals, color=colors, edgecolor='black')
    plt.title('Questions by Citation Count Bin')
    plt.xlabel('Citation Range')
    plt.ylabel('Number of Questions')
    plt.xticks(rotation=45, ha='right')
    for bar, cnt in zip(bars, vals):
        if cnt > 0:
            plt.text(bar.get_x() + bar.get_width()/2, cnt, str(cnt),
                     ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(stats_folder / 'questions_by_citation_bin.png', dpi=300, bbox_inches='tight')
    plt.savefig(stats_folder / 'questions_by_citation_bin.svg', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Citation stats saved to {stats_folder / 'questions_by_citation_bin.png'}")

    # New Plot: Citation Bins vs Year Bin Counts Heatmap
    df_year_cit = create_year_bins(df_dataset)
    df_year_cit = create_citation_bins(df_year_cit)

    # Ensure 'year_bin' and 'citation_bin' are present
    if 'year_bin' in df_year_cit.columns and 'citation_bin' in df_year_cit.columns:
        contingency_table = pd.crosstab(df_year_cit['year_bin'], df_year_cit['citation_bin'])
        
        # Reorder citation bins for consistency
        bin_order_cit = ["0", "1-10", "11-100", "101-500", "501-1000", "1001-1702"]
        # Reorder year bins for consistency (assuming they are sorted by create_year_bins)
        year_bin_order = sorted(df_year_cit['year_bin'].unique(), key=lambda x: int(x.split('-')[0]) if isinstance(x, str) and '-' in x else -1)

        contingency_table = contingency_table.reindex(index=year_bin_order, columns=bin_order_cit).fillna(0)

        plt.figure(figsize=(12, 8))
        plt.imshow(contingency_table, cmap=greens_cmap, aspect='auto')
        
        plt.colorbar(label='Number of Questions')
        plt.title('Question Counts: Citation Bins vs Year Bins', fontweight ='bold')
        plt.xlabel('Citation Bin')
        plt.ylabel('Year Bin')
        
        plt.xticks(ticks=np.arange(len(contingency_table.columns)), labels=contingency_table.columns, rotation=45, ha='right')
        plt.yticks(ticks=np.arange(len(contingency_table.index)), labels=contingency_table.index)

        # Add text annotations for counts
        for i in range(len(contingency_table.index)):
            for j in range(len(contingency_table.columns)):
                text_color = "white" if contingency_table.iloc[i, j] > contingency_table.values.max() / 2 else "black"
                plt.text(j, i, int(contingency_table.iloc[i, j]),
                         ha="center", va="center", color=text_color, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(stats_folder / 'citation_vs_year_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(stats_folder / 'citation_vs_year_heatmap.svg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Citation vs Year heatmap saved to {stats_folder / 'citation_vs_year_heatmap.png'}")
    else:
        print("Skipping Citation vs Year heatmap due to missing 'year_bin' or 'citation_bin' columns.")

    # 3. Area Sunburst (no change in coloring logic; Plotly handles coloration)
    # You can optionally pass a color sequence to Plotly but 
    # we'll leave this as-is so it uses its default sunburst palette.
    from plotly.offline import plot as plotly_plot
    import plotly.graph_objects as go

    df = df_dataset.copy()
    
    try:
        # try the two‐level case
        df[['primary','secondary']] = df['area'].str.split(' - ', n=1, expand=True)
        df['secondary'] = df['secondary'].fillna(df['primary'])
        
        # aggregate counts per (primary, secondary)
        agg = (
            df
            .groupby(['primary','secondary'], dropna=False)
            .size()
            .reset_index(name='count')
        )
        
        fig = px.sunburst(
            agg,
            path=['primary','secondary'],
            values='count',
            title='Questions by Area'
        )
        
    except Exception:
        # fallback: only a single level
        agg = (
            df['area']
            .value_counts()
            .rename_axis('primary')
            .reset_index(name='count')
        )
        
        fig = px.sunburst(
            agg,
            path=['primary'],
            values='count',
            title='Questions by Area'
        )

    # show both name and count on each wedge
    fig.update_traces(
        branchvalues='total',
        insidetextorientation='radial',
        textinfo='label+value'
    )
    config = {
        'toImageButtonOptions': {
            'format': 'svg',       # force SVG output
            'filename': 'questions_by_area_sunburst',
            'width': 800,
            'height': 600,
            'scale': 1            # 1× size
        },
        'displaylogo': False,      # hide the Plotly logo
    }

    # 1) If you're in a notebook or interactive session:
    fig.show(config=config)

    # 4.Plant Species Distribution
    palette = {
        "Model Organisms":          "#1f77b4",
        "Cereal Grains":            "#ff7f0e",
        "Legumes":                  "#2ca02c",
        "Solanaceae & Relatives":   "#d62728",
        "Woody Perennials & Trees": "#9467bd",
        "Other Herbaceous Crops, Spices, Fibers & Weeds": "#8c564b",
        "Non-specific":             "#7f7f7f",
    }

    counts = df_dataset['normalized_plant_species'].value_counts()
    labels = [f"{species}\n{count}" for species, count in zip(counts.index, counts.values)]
    sizes  = counts.values
    colors = [palette[group] for group in counts.index]

    # set a light neutral background
    plt.rcParams['figure.facecolor'] = 'whitesmoke'
    plt.rcParams['font.family']     = 'DejaVu Sans'

    fig, ax = plt.subplots(figsize=(10, 6))

    # draw treemap
    squarify.plot(
        sizes       = sizes,
        label       = labels,
        color       = colors,
        alpha       = 0.85,                     # slight transparency
        pad         = True,
        bar_kwargs  = {
            'linewidth': 2,                     # thicker white borders
            'edgecolor': 'white',
        },
        text_kwargs = {
            'fontsize'   : 12,
            'fontweight' : 'bold',
            'color'      : 'black',
           # 'va'         : 'center',
           # 'ha'         : 'center',
            'wrap'       : True,                # long labels will wrap
        }
    )

    # clean up
    ax.axis('off')
    plt.title(
        'Distribution of Plant Species (Treemap)',
        fontsize=16,
        fontweight='semibold',
        pad=20
    )
    plt.tight_layout()

    # save
    plt.savefig(stats_folder / 'plant_species_treemap_styled.png', dpi=300, bbox_inches='tight')
    plt.savefig(stats_folder / 'plant_species_treemap_styled.svg', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plant species stats saved to {stats_folder / 'top_plant_species.png'}")

    # New plot call for citation distribution by journal type
    plot_citation_distribution_by_journal_type(df_dataset, stats_folder)

def perform_experiments(df_dataset):
    raise NotImplementedError

def perform_automatic_plots(df_dataset, output_folder):
    origin_folder = output_folder
    output_folder = output_folder / 'plots'
    os.makedirs(output_folder, exist_ok=True)

    MODEL_COLORS  = {
        'LLaMA': '#8B4513',
        'Gemini': '#4285F4',
        'Claude': '#FF6C0A',
        'GPT-4o': '#10A37F',
        'O1-mini': '#8FB339',
        'DeepSeek V3': '#0B5E99',
        'DeepSeek R1': '#003366'
    }

    MODEL_COLORS  = {
        'LLaMA': '#8B4513',
        'Gemini': '#4285F4',
        'Claude': '#FF6C0A',
        'GPT-4o': '#10A37F',
        'O1-mini': '#8FB339',
        'DeepSeek V3': '#0B5E99',
        'DeepSeek R1': '#003366'
    }

    if os.path.exists(origin_folder / 'metrics'):
        origin_folder = origin_folder / 'metrics'
        #generate_overall_spidergraph(origin_folder / 'normalized_area' / 'answer_accuracy.csv', 'area', output_folder )
        plot_citation_bin_accuracy(origin_folder / 'citation_bin' / 'all_results.csv', output_folder)
        
        bin_df = create_year_bins(df_dataset)
        plot_year_accuracy(origin_folder / 'year_bin' / 'all_results.csv', bin_df['year_bin'].value_counts(), output_folder)
        plot_journal_type_accuracy(origin_folder / 'source_journal' / 'answer_accuracy.csv', MODEL_COLORS, output_folder)

        try:
            plot_lollipop_chart(origin_folder / 'normalized_area' / 'all_results.csv', MODEL_COLORS, output_folder)
            plot_bump_chart(origin_folder / 'normalized_area' / 'answer_accuracy.csv', MODEL_COLORS, output_folder)
            plot_scatter_ranking_chart(origin_folder / 'normalized_area' / 'answer_accuracy.csv', MODEL_COLORS, output_folder)
            plot_grouped_barplot(origin_folder / 'normalized_area' / 'answer_accuracy.csv', MODEL_COLORS, output_folder)
        except:
            plot_lollipop_chart(origin_folder / 'area' / 'all_results.csv', MODEL_COLORS, output_folder)
            plot_bump_chart(origin_folder / 'area' / 'answer_accuracy.csv', MODEL_COLORS, output_folder)
            plot_scatter_ranking_chart(origin_folder / 'area' / 'answer_accuracy.csv', MODEL_COLORS, output_folder)
            plot_grouped_barplot(origin_folder / 'area' / 'answer_accuracy.csv', MODEL_COLORS, output_folder)
    else:
        print('No metrics results folder detected... passing to statistics plots.')


greens_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'greens', ["#bdf3be", '#006d2c']
)
normalize = mpl.colors.Normalize(vmin=70, vmax=100)

model_names_mapping = {
        'llama': 'LLaMA',
        'gemini': 'Gemini',
        'claude': 'Claude',
        'chatgpt': 'GPT-4o',
        'o1-mini': 'O1-mini',
        'v3': 'DeepSeek V3',
        'r1': 'DeepSeek R1'
    }

def plot_citation_bin_accuracy(data_path, output_folder):
    df = pd.read_csv(data_path, index_col=0).drop('Overall', errors='ignore').drop('count', axis=1)
    bin_order = ['0', '1-10', '11-100', '101-500', '501-1000', '1001-1702']
    df = df.reindex(bin_order)

    # Mean accuracy and std deviation across models
    df_mean = df[[col for col in df.columns if col.endswith('_answer_accuracy_mean')]].mean(axis=1)
    df_std = df[[col for col in df.columns if col.endswith('_answer_accuracy_std')]].mean(axis=1)

    plt.figure(figsize=(12, 6))
    heights = df_mean
    errors = df_std
    colors = [greens_cmap(normalize(h)) for h in heights]
    bars = plt.bar(df_mean.index, heights, yerr=errors, capsize=5, color=colors)

    plt.title('Model Accuracy by Citation Count', fontweight='bold', fontsize=16)
    plt.xlabel('Number of citations', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(70, 105)
    plt.grid(axis='y', alpha=0.3)

    for bar, height in zip(bars, heights):
        plt.text(bar.get_x() + bar.get_width() / 4, height,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=14)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'svg']:
        plt.savefig(output_folder / f'citations.{ext}', format=ext, dpi=300, bbox_inches='tight')
    plt.close()

    # Individual models
    models = list(model_names_mapping.values())
    n_models = len(models)
    if n_models == 0:
        return

    fig, axes = plt.subplots(n_models, 1, figsize=(12, 3 * n_models))
    if n_models == 1:
        axes = [axes]

    for model, ax in zip(models, axes):
        heights = df[f'{model}_answer_accuracy_mean']
        errors = df[f'{model}_answer_accuracy_std']  
        colors = [greens_cmap(normalize(h)) for h in heights]
        bars = ax.bar(df.index, heights, yerr=errors, capsize=3, color=colors)
        ax.set_title(f'Model: {model}', fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)

        for bar, height in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 4, height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

        if model == models[-1]:
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            for label in ax.get_xticklabels():
                label.set_ha('right')
        else:
            ax.set_xticklabels([])

        ax.tick_params(axis='y', labelsize=8)

    fig.text(0.5, 0.02, 'Number of citations', ha='center', va='center', fontsize=12)
    fig.text(0.02, 0.5, 'Accuracy (%)', ha='center', va='center',
             rotation='vertical', fontsize=12)

    plt.tight_layout()
    plt.suptitle('Model Accuracies by Number of Citations', fontsize=14, y=0.98)
    for ext in ['png', 'svg']:
        plt.savefig(output_folder / f'citations_models.{ext}', format=ext, dpi=300, bbox_inches='tight')
    plt.close()

def plot_year_accuracy(data_path, sample_counts, output_folder):
    df = pd.read_csv(data_path, index_col=0).drop('Overall', errors='ignore').drop('count', axis=1)
    sorted_bins = sorted(df.index, key=lambda x: int(x.split('-')[0]))
    df = df.loc[sorted_bins]
    sample_counts = sample_counts.reindex(sorted_bins).fillna(0)

    # Mean accuracy and std deviation
    df_mean = df[[col for col in df.columns if col.endswith('_answer_accuracy_mean')]].mean(axis=1)
    df_std = df[[col for col in df.columns if col.endswith('_answer_accuracy_std')]].mean(axis=1)
    window_size = 3
    moving_avg = df_mean.rolling(window=window_size, min_periods=1, center=True).mean()

    fig, ax1 = plt.subplots(figsize=(14, 7))
    x = range(len(df_mean.index))

    colors = [greens_cmap(normalize(h)) for h in df_mean]
    ax1.errorbar(x, df_mean, yerr=df_std, fmt='o', color='black',
                 ecolor='black', elinewidth=1, capsize=5, zorder=2, label='Mean Accuracy')
    ax1.scatter(x, df_mean, color=colors, s=300, edgecolors= 'black', zorder=3)
    ax1.plot(x, df_mean, linestyle=':', color='gray', alpha=0.7)
    ax1.plot(x, moving_avg, color='#006d2c', linewidth=2, label=f'{window_size}-bin Moving Average')

    ax1.set_ylabel('Mean Accuracy (%)', fontsize=12)
    ax1.set_ylim(70, 100)
    ax1.grid(axis='y', alpha=0.3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df_mean.index, rotation=45)
    for label in ax1.get_xticklabels():
        label.set_ha('right')

    ax1.set_title('Model Accuracy by Year', fontsize=14)
    ax1.set_xlabel('Year Range', fontsize=12)
    ax1.legend(loc='best')

    for i, acc in enumerate(df_mean):
        ax1.text(x[i], acc + 1, f'{acc:.1f}%', ha='center', va='bottom', fontsize=14)

    plt.tight_layout()
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'svg']:
        plt.savefig(output_folder / f'years.{ext}', format=ext, dpi=300, bbox_inches='tight')
    plt.close()

    # Individual model plots
    models = list(model_names_mapping.values())
    n_models = len(models)
    if n_models == 0:
        return

    fig, axes = plt.subplots(n_models, 1, figsize=(14, 4 * n_models))
    if n_models == 1:
        axes = [axes]

    for model, ax in zip(models, axes):
        heights = df[f'{model}_answer_accuracy_mean']
        errors = df[f'{model}_answer_accuracy_std'] 
        ax1_sub = ax
        x = range(len(df.index))

        colors = [greens_cmap(normalize(h)) for h in heights]
        ax1_sub.errorbar(x, heights, yerr=errors, fmt='o', color='black',
                         ecolor='black', elinewidth=1, capsize=3, zorder=2)
        ax1_sub.scatter(x, heights, color=colors, s=60, zorder=3)
        ax1_sub.plot(x, heights, linestyle=':', color='gray', alpha=0.7)

        ax1_sub.set_ylabel('Accuracy (%)', fontsize=10)
        ax1_sub.set_ylim(45, 105)
        ax1_sub.grid(axis='y', alpha=0.3)

        ax1_sub.set_xticks(x)
        if model == models[-1]:
            ax1_sub.set_xticklabels(df.index, rotation=45)
            for label in ax1_sub.get_xticklabels():
                label.set_ha('right')
            ax1_sub.set_xlabel('Year Range', fontsize=10)
        else:
            ax1_sub.set_xticklabels([])

        for j, acc in enumerate(heights):
            ax1_sub.text(x[j], acc + 1, f'{acc:.1f}%', 
                         ha='center', va='bottom', fontsize=14)

        ax1_sub.set_title(f'Model: {model}', fontsize=11, pad=10)
        ax1_sub.tick_params(axis='y', labelsize=8)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.suptitle('Model Accuracies and Sample Distribution by Year', fontsize=14, y=0.98)
    for ext in ['png', 'svg']:
        plt.savefig(output_folder / f'years_models.{ext}', format=ext, dpi=300, bbox_inches='tight')
    plt.close()

def plot_journal_type_accuracy(data_path: str, model_colors: dict, output_folder: Path):
    """Generates a grouped bar plot comparing model accuracy on review vs. paper journals."""
    
    # Load and process data
    df = pd.read_csv(data_path, index_col=0)
    df = df.drop('Overall', errors='ignore').drop('count', axis=1)
    
    # Define journal types
    review_journals = ['TIPS', 'COPB']
    
    # Separate dataframes
    df_reviews = df[df.index.isin(review_journals)]
    df_papers = df[~df.index.isin(review_journals)]
    
    # Calculate mean accuracies
    review_accuracies = df_reviews.mean()
    paper_accuracies = df_papers.mean()
    
    # Prepare data for plotting
    model_names = [col.replace('_answer_accuracy', '') for col in df.columns]
    
    plot_df = pd.DataFrame({
        'Model': model_names,
        'Reviews': review_accuracies.values,
        'Papers': paper_accuracies.values
    }).set_index('Model')

    # Get colors for each model
    colors = [model_colors.get(model, '#333333') for model in plot_df.index]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    
    x = np.arange(len(plot_df.index))
    width = 0.35
    
    # Bar for reviews (hatched)
    rects1 = ax.bar(x - width/2, plot_df['Reviews'], width, 
                    color=colors, 
                    hatch='//', edgecolor='black', linewidth=0.5)
    
    # Bar for papers (solid)
    rects2 = ax.bar(x + width/2, plot_df['Papers'], width, 
                    color=colors, 
                    edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_ylabel('Accuracy (%)', fontsize=12, weight='semibold')
    ax.set_title('Model Accuracy by Journal Type (Reviews vs. Papers)', 
                 pad=20, fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=45, ha='right', fontsize=10, weight='semibold')
    ax.set_ylim(70, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add data labels
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='grey', edgecolor='black', hatch='//', label='Review Journals'),
        Patch(facecolor='grey', edgecolor='black', label='Paper Journals')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    plt.tight_layout()
    
    # Save outputs
    output_path_svg = output_folder / "journal_type_accuracy.svg"
    plt.savefig(output_path_svg, format='svg', bbox_inches="tight")
    output_path_png = output_folder / "journal_type_accuracy.png"
    plt.savefig(output_path_png, format='png', bbox_inches="tight")
    plt.close()
    
    print(f"Journal type accuracy plot saved to: {output_path_svg}")


def plot_lollipop_chart(data_path: str, model_colors: dict, output_folder: Path):
    # Load and process data
    df = pd.read_csv(data_path, index_col=0)
    overall_row = df.loc['Overall']
    
    # Extract models and metrics
    models = []
    means = []
    stds = []
    
    for col in df.columns:
        if '_answer_accuracy_mean' in col:
            model = col.split('_')[0]  # Extract model name
            models.append(model)
            means.append(overall_row[col])
            std_col = col.replace('_mean', '_std')
            stds.append(overall_row[std_col])
    
    # Create dataframe and sort
    data = pd.DataFrame({'Model': models, 'Accuracy': means, 'Std': stds})
    data = data.sort_values('Accuracy', ascending=True)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    
    # Plot horizontal lollipops
    for i, (acc, model, std) in enumerate(zip(data['Accuracy'], data['Model'], data['Std'])):
        color = model_colors.get(model, '#333333')  # Default to dark gray if unknown
        
        # Horizontal line with subtle gradient, +- std
        ax.hlines(y=i, xmin=acc - std, xmax=acc + std, color=color, linewidth=2.0, alpha=0.5)

        # Marker to the left and right at std
        ax.scatter(acc - std, i, s=200, color=color, linewidth=2.0, marker='|', zorder=10)
        ax.scatter(acc + std, i, s=200, color=color, linewidth=2.0, marker='|', zorder=10)
        
        # Custom marker with white border
        ax.scatter(acc, i, s=100, color=color, linewidth=2.0, marker='o', zorder=10)
        
        # Smart label placement
        label_x = acc + std + 1  # Offset from marker
        ha = 'left' if (100 - acc) > 10 else 'right'  # Flip if near edge
        if ha == 'right':
            label_x = acc - 2  # Left offset for edge cases
            
        ax.text(label_x, i, 
               f'{acc:.1f}% ±{std:.1f}',
               va='center', ha=ha, 
               fontsize=10, weight='bold',
               color=color,
               path_effects=[patheffects.withStroke(linewidth=2, foreground='white')])
    
    # Customize plot
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data['Model'], ha='right', va='center',
                      fontsize=11, weight='semibold')
    ax.set_xlabel('Accuracy (%)', fontsize=12, labelpad=15, weight='semibold')
    ax.set_title('Global Model Performance in APOLO Benchmark', 
               pad=20, fontsize=14, weight='bold')
    ax.set_xlim(70, 100)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    #ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Clean frame
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    
    plt.tight_layout()
    output_path_svg = output_folder / "lollipop_general_performance.svg"
    plt.savefig(output_path_svg, format='svg', bbox_inches="tight")
    output_path_png = output_folder / "lollipop_general_performance.png"
    plt.savefig(output_path_png, format='png', bbox_inches="tight")
    plt.close()

    print(f"Lollipop chart saved to: {output_path_svg}")


def plot_bump_chart(data_path: str, model_colors: dict, output_folder: Path):
    """Generate bump chart for performance by domain with custom styling"""

    # Load and process data
    df = pd.read_csv(data_path).dropna()
    df = df[df.iloc[:, 0] != 'Overall']  # Remove overall row
    areas = df.iloc[:, 0].values
    model_cols = [col for col in df.columns if '_answer_accuracy' in col]
    models = [col.split('_')[0] for col in model_cols]
    
    # Prepare ranking data
    rank_data = []
    for _, row in df.iterrows():
        scores = {model: row[col] for model, col in zip(models, model_cols)}
        sorted_models = sorted(scores.items(), key=lambda x: -x[1])
        ranks = {model: i+1 for i, (model, _) in enumerate(sorted_models)}
        rank_data.append(ranks)
    
    ranks_df = pd.DataFrame(rank_data, index=areas)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    
    # Plot lines with custom styling
    for model in models:
        model_ranks = ranks_df[model].values
        x = np.arange(len(areas))
        y = model_ranks
        
        # Create line segments with model color
        color = model_colors.get(model, '#777777')  # Gray for unknown models
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = LineCollection(segments, 
                          colors=[color]*(len(segments)),  # Solid color
                          linewidth=2.5,
                          alpha=0.9,
                          zorder=2)
        ax.add_collection(lc)
        
        # Add styled markers
        ax.scatter(x, y, s=120, color=color,
                 edgecolor='white', linewidth=1.2,
                 marker='o', zorder=3, label=model)
        
        # Add score annotations
        model_col = model_cols[models.index(model)]  # Get column name for scores
        scores = df[model_col].values  # Actual scores for this model
        for xi, yi, score in zip(x, y, scores):
            ax.text(xi, yi - 0.25, 
                    f"{score:.1f}", 
                    ha='center', 
                    va='bottom', 
                    fontsize=8, 
                    color=color)
    
    # Customize axis
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(areas)))
    ax.set_xticklabels(areas, rotation=45, ha='right', 
                      fontsize=10, weight='semibold')
    ax.set_yticks(np.arange(1, len(models)+1))
    ax.set_title('Model Rank by Question Area', 
               pad=20, fontsize=14, weight='bold')
    ax.set_yticklabels([f'Rank {i}' for i in range(1, len(models)+1)], 
                     fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0.5, 7.5)
    # invert y axis
    ax.invert_yaxis()
    
    # Add legend with custom handles
    legend_handles = [Line2D([0], [0], marker='o', color=model_colors[model],
                      label=model, markersize=10, linestyle='-')
                   for model in models if model in model_colors]
    ax.legend(handles=legend_handles, loc='upper left',
            bbox_to_anchor=(1, 1), frameon=True,
            title='Models', title_fontsize=11)
    
    # Save output
    plt.tight_layout()
    output_path_svg = output_folder / "bump_chart_domain_ranking.svg"
    plt.savefig(output_path_svg, format='svg', bbox_inches="tight")
    output_path_png = output_folder / "bump_chart_domain_ranking.png"
    plt.savefig(output_path_png, format='png', bbox_inches="tight")
    plt.close()
    
    print(f"Bump chart saved to: {output_path_svg}")


def plot_scatter_ranking_chart(data_path: str, model_colors: dict, output_folder: Path):
    """Generate score chart with actual values on y-axis"""
    
    # Load and process data
    df = pd.read_csv(data_path).dropna()
    df = df[df.iloc[:, 0] != 'Overall']  # Remove overall row
    areas = df.iloc[:, 0].values
    model_cols = [col for col in df.columns if '_answer_accuracy' in col]
    models = [col.split('_')[0] for col in model_cols]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    
    # Plot lines with actual scores
    for model in models:
        model_col = f"{model}_answer_accuracy"
        scores = df[model_col].values
        x = np.arange(len(areas))
        y = scores
        
        # Create line segments with model color
        color = model_colors.get(model, '#777777')
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = LineCollection(segments, 
                          colors=[color]*(len(segments)),
                          linewidth=2.5,
                          alpha=0.9,
                          zorder=2)
        ax.add_collection(lc)
        
        # Add styled markers
        ax.scatter(x, y, s=120, color=color,
                 edgecolor='white', linewidth=1.2,
                 marker='o', zorder=3, label=model)

    # Customize axis
    ax.set_ylim(60, 105)  # Set fixed y-axis range
    ax.set_xticks(np.arange(len(areas)))
    ax.set_xticklabels(areas, rotation=45, ha='right', 
                      fontsize=10, weight='semibold')
    ax.set_yticks(np.arange(60, 101, 5))  # Grid lines every 5 points
    ax.set_title('Model Performance by Question Area', 
               pad=20, fontsize=14, weight='bold')
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Add legend
    legend_handles = [Line2D([0], [0], marker='o', color=model_colors[model],
                      label=model, markersize=10, linestyle='-')
                   for model in models if model in model_colors]
    ax.legend(handles=legend_handles, loc='upper left',
            bbox_to_anchor=(1, 1), frameon=True,
            title='Models', title_fontsize=11)
    
    # Save output
    plt.tight_layout()
    output_path_svg = output_folder / "score_chart_domain.svg"
    plt.savefig(output_path_svg, format='svg', bbox_inches="tight")
    output_path_png = output_folder / "score_chart_domain.png"
    plt.savefig(output_path_png, format='png', bbox_inches="tight")
    plt.close()
    
    print(f"Score chart saved to: {output_path_svg}")


def plot_grouped_barplot(data_path: str, model_colors: dict, output_folder: Path):
    """Generate a grouped bar plot for each domain showing model performance"""
    
    # Load and process data
    df = pd.read_csv(data_path).dropna()
    df = df[df.iloc[:, 0] != 'Overall']  # Remove overall row
    areas = df.iloc[:, 0].values
    model_cols = [col for col in df.columns if '_answer_accuracy' in col]
    models = [col.split('_')[0] for col in model_cols]
    
    # Create plot with adjusted size
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    
    # Calculate bar dimensions
    num_domains = len(areas)
    num_models = len(models)
    bar_width = 0.8 / num_models  # Width of each model's bar

    # Plot bars for each model
    for model_idx, model in enumerate(models):
        model_col = f"{model}_answer_accuracy"
        scores = df[model_col].values
        
        # Calculate positions for this model's bars
        x_positions = np.arange(num_domains) + model_idx * bar_width - (bar_width * (num_models - 1) / 2)
        
        # Get color from dictionary or use default
        color = model_colors.get(model, '#777777')
        
        # Plot bars with styling
        ax.bar(x_positions, 
               scores, 
               width=bar_width, 
               color=color,
               edgecolor='white', 
               linewidth=1.2,
               label=model)

    # Customize axis
    ax.set_ylim(60, 100)
    ax.set_xticks(np.arange(num_domains))
    ax.set_xticklabels(areas, rotation=45, ha='right', fontsize=10, weight='semibold')
    ax.set_yticks(np.arange(60, 101, 5))
    ax.set_title('Model Performance by Question Area', pad=20, fontsize=14, weight='bold')
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Create legend with model colors
    legend_handles = [Patch(color=model_colors[model], label=model) 
                     for model in models if model in model_colors]
    ax.legend(handles=legend_handles, 
             loc='upper left',
             bbox_to_anchor=(1, 1), 
             frameon=True,
             title='Models', 
             title_fontsize=11)

    # Save outputs
    plt.tight_layout()
    output_path_svg = output_folder / "grouped_bar_chart.svg"
    plt.savefig(output_path_svg, format='svg', bbox_inches="tight")
    output_path_png = output_folder / "grouped_bar_chart.png"
    plt.savefig(output_path_png, format='png', bbox_inches="tight")
    plt.close()
    
    print(f"Grouped bar plot saved to: {output_path_svg}")