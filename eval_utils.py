import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib import patheffects
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection


EVALUATION_STYLES = ['complete', 'automatic-metrics', 'human-metrics', 'statistics', 'experiments', 'automatic-plotting', 'human-plotting']


def perform_complete_evaluation(df_dataset, output_folder):

    perform_metrics(df_dataset, output_folder)
    perform_descriptive_statistics(df_dataset)
    print('not implemented yet: perform_experiments(df_dataset)')

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
    model_names = [model_names_mapping[col.replace('CoT_election_by_', '')] for col in model_columns]

    # clean_names = []
    # for model in model_names:
    #     suffix = model.split('_')[1]
    #     name = model.split('_')[0]
    #     clean_name = f'{model_names_mapping[name]}_{suffix}'
    #     clean_names.append(clean_name)
    # model_names = clean_names

    df_dataset.rename(columns=dict(zip(model_columns, model_names)), inplace=True)

    # Preprocess model columns: convert to numeric and then to integers, invalid -> NaN
    for model in model_names:
        # Convert to numeric, coercing errors to NaN
        df_dataset[model] = pd.to_numeric(df_dataset[model], errors='coerce')
        # Convert to integer (NaN remains NaN)
        df_dataset[model] = df_dataset[model].astype(pd.Int64Dtype())  # Allows integer NaN

    # Group by plant_species and calculate accuracies
    group_by_and_score(df_dataset, 'plant_species', model_names, output_folder)

    # Group by area and calculate accuracies
    group_by_and_score(df_dataset, 'area', model_names, output_folder)

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
    group_by_and_score(df_dataset_year_binned, 'year_bin', model_names, output_folder)
    
    df_dataset_citation_binned = create_citation_bins(df_dataset)
    group_by_and_score(df_dataset_citation_binned, 'citation_bin', model_names, output_folder)

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
            #suffix = model.split('_')[-1]
            #answer_col = f'answer_{suffix}'
            answer_col = f'answer'
            
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
        #suffix = model.split('_')[-1]
        #answer_col = f'answer_{suffix}'
        answer_col = 'answer'
        
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
            #run_columns = [f"{base_model}_{suffix}_{metric}" for suffix in ['first', 'second', 'third']]
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
        plot_citation_bin_accuracy(origin_folder / 'citation_bin' / 'answer_accuracy.csv', output_folder)
        
        bin_df = create_year_bins(df_dataset)
        plot_year_accuracy(origin_folder / 'year_bin' / 'answer_accuracy.csv', bin_df['year_bin'].value_counts(), output_folder)

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


def plot_citation_bin_accuracy(data_path, output_folder):
    # Read and prepare data
    df = pd.read_csv(data_path, index_col=0)
    df = df.drop('Overall', errors='ignore')
    
    # Exclude the 'Overall' row and sort bins logically
    df = df.drop('Overall', errors='ignore')
    df = df.drop('count', axis=1)
    bin_order = ['0', '1-10', '11-100', '101-500', '501-1000', '1001-1702']
    df = df.reindex(bin_order)
    
    # --- Plot 1: Mean Accuracy ---
    df_mean = df.copy()
    df_mean['mean_accuracy'] = df.mean(axis=1)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df_mean.index, df_mean['mean_accuracy'], color='skyblue')
    
    plt.title('Model Accuracy by Citation Count', fontweight='bold', fontsize=16)
    plt.xlabel('Number of citations', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(70, 100)
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_folder / 'citations.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(output_folder / 'citations.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Plot 2: Individual Models ---
    models = df.columns.tolist()
    n_models = len(models)
    
    if n_models == 0:
        return  # No models to plot
    
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 3 * n_models))
    if n_models == 1:
        axes = [axes]
    
    for i, (model, ax) in enumerate(zip(models, axes)):
        bars = ax.bar(df.index, df[model], color='skyblue')
        ax.set_title(f'Model: {model}', fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        # Value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=8)
        
        # X-axis labels only on bottom subplot
        # X-axis labels only on bottom subplot
        if i == n_models - 1:
            # Rotate labels and set alignment
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            # Correct horizontal alignment after rotation
            for label in ax.get_xticklabels():
                label.set_ha('right')
        else:
            ax.set_xticklabels([])
        
        ax.tick_params(axis='y', labelsize=8)
    
    # Common labels
    fig.text(0.5, 0.02, 'Number of citations', ha='center', va='center', fontsize=12)
    fig.text(0.02, 0.5, 'Accuracy (%)', ha='center', va='center', rotation='vertical', fontsize=12)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])  # Adjust for labels
    plt.suptitle('Model Accuracies by Number of Citations', fontsize=14, y=0.98)
    plt.savefig(output_folder / 'citations_models.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(output_folder / 'citations_models.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()


def plot_year_accuracy(data_path, sample_counts, output_folder):
        # Read and prepare data
    df = pd.read_csv(data_path, index_col=0)
    df = df.drop('Overall', errors='ignore')
    df = df.drop('count', axis=1)
    
    sorted_bins = sorted(df.index, key=lambda x: int(x.split('-')[0]))
    df = df.loc[sorted_bins]
    sample_counts = sample_counts.reindex(sorted_bins).fillna(0)
    
    # --- Plot 1: Mean Accuracy ---
    df_mean = df.copy()
    df_mean['mean_accuracy'] = df.mean(axis=1)
    # Calculate moving average with a window of 3 (adjust window size as needed)
    window_size = 3
    df_mean['moving_avg'] = df_mean['mean_accuracy'].rolling(
        window=window_size,
        min_periods=1,
        center=True
    ).mean()
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    x = range(len(df_mean.index))
    
    # Accuracy plot
    ax1.scatter(x, df_mean['mean_accuracy'], color='darkblue', s=100, zorder=3, label='Mean Accuracy')
    ax1.plot(x, df_mean['mean_accuracy'], linestyle=':', color='gray', alpha=0.7)
    ax1.plot(x, df_mean['moving_avg'], color='red', linewidth=2, label=f'{window_size}-bin Moving Average')

    ax1.set_ylabel('Mean Accuracy (%)', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # Sample count plot
    # ax2 = ax1.twinx()
    # bars = ax2.bar(x, sample_counts, color='lightblue', alpha=0.6)
    # ax2.set_ylabel('Number of Samples', fontsize=12)
    
    # Configure axes
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_mean.index, rotation=45)
    # Fix label alignment after rotation
    for label in ax1.get_xticklabels():
        label.set_ha('right')
    
    ax1.set_title('Model Accuracy by Year', fontsize=14)
    ax1.set_xlabel('Year Range', fontsize=12)
    ax1.legend(loc='best')  # Add legend
    
    # Add value labels
    for i, acc in enumerate(df_mean['mean_accuracy']):
        ax1.text(x[i], acc + 1, f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
        
    # for bar in bars:
    #     height = bar.get_height()
    #     if height > 0:
    #         ax2.text(bar.get_x() + bar.get_width()/2., height,
    #                  f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    plt.ylim(70,105)
    plt.tight_layout()
    plt.savefig(output_folder / 'years.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(output_folder / 'years.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Plot 2: Individual Models ---
    models = df.columns.tolist()
    n_models = len(models)
    
    if n_models == 0:
        return
    
    fig, axes = plt.subplots(n_models, 1, figsize=(14, 4 * n_models))
    if n_models == 1:
        axes = [axes]
    
    for i, (model, ax) in enumerate(zip(models, axes)):
        # Create twin axes for each subplot
        ax1_sub = ax
        ax2_sub = ax.twinx()
        
        x = range(len(df.index))
        
        # Model accuracy plot
        ax1_sub.scatter(x, df[model], color='darkblue', s=60, zorder=3)
        ax1_sub.plot(x, df[model], linestyle=':', color='gray', alpha=0.7)
        ax1_sub.set_ylabel('Accuracy (%)', fontsize=10)
        ax1_sub.set_ylim(0, 105)
        ax1_sub.grid(axis='y', alpha=0.3)
        
        # Sample count plot (shared across all models)
        bars = ax2_sub.bar(x, sample_counts, color='lightblue', alpha=0.4)
        ax2_sub.set_ylabel('Samples', fontsize=9)
        
        # Configure axes
        ax1_sub.set_xticks(x)
        if i == n_models - 1:
            ax1_sub.set_xticklabels(df.index, rotation=45)
            # Fix label alignment after rotation
            for label in ax1_sub.get_xticklabels():
                label.set_ha('right')
            ax1_sub.set_xlabel('Year Range', fontsize=10)
        else:
            ax1_sub.set_xticklabels([])
        
        # Add value labels
        for j, acc in enumerate(df[model]):
            ax1_sub.text(x[j], acc + 1, f'{acc:.1f}%', 
                        ha='center', va='bottom', fontsize=8)
            
        # Add model title
        ax1_sub.set_title(f'Model: {model}', fontsize=11, pad=10)
        
        # Set tick label sizes
        ax1_sub.tick_params(axis='y', labelsize=8)
        ax2_sub.tick_params(axis='y', labelsize=8)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.suptitle('Model Accuracies and Sample Distribution by Year', fontsize=14, y=0.98)
    plt.savefig(output_folder / 'years_models.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(output_folder / 'years_models.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()


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
        
        # Horizontal line with subtle gradient
        ax.hlines(y=i, xmin=0, xmax=acc, color=color, 
                alpha=0.9, linewidth=4, linestyle='-')
        
        # Custom marker with white border
        ax.scatter(acc, i, s=200, color=color, edgecolor='white',
                 linewidth=1.5, marker='o', zorder=10)
        
        # Smart label placement
        label_x = acc + 2  # Offset from marker
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
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
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
            ax.text(xi, yi - 0.15, 
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
    ax.set_ylim(60, 100)  # Set fixed y-axis range
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