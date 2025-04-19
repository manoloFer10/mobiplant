import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_static_dot_plot(scoring_csv_path, output_path, dpi=300):
    # Load and prepare data
    df = pd.read_csv(scoring_csv_path, index_col=0)
    df = df.drop('Overall', axis=0)
    df = df.reset_index().rename(columns={'index': 'Criterion'})
    melted_df = df.melt(id_vars=['Criterion'], 
                       var_name='Model', 
                       value_name='Score')
    
    # Define rubric labels using actual text from your scoring system
    RUBRIC_LABELS = {
        "Alignment with Scientific Consensus": {
            'min': "Opposed to consensus",
            'max': "Aligned to consensus"
        },
        "Evidence of Correct Reasoning": {
            'min': "No correct reasoning",
            'max': "Complete reasoning"
        },
        "Inclusion of Hallucinated Content": {
            'min': "Hallucinates content\nknown to be wrong",
            'max': "No hallucinations"
        },
        "Presence of Irrelevant Content": {
            'min': "Contains irrelevant\ncontent",
            'max': "Fully relevant"
        },
        "Acknowledgement of Self Limitations": {
            'min': "No limitations noted",
            'max': "Limitations\nacknowledged"
        },
        "Omission of Important Information": {
            'min': "Omits critical\ninformation",
            'max': "No omissions"
        },
        "Evidence of Reading Comprehension": {
            'min': "Misinterprets question",
            'max': "Full comprehension"
        },
        "Potential of Species Bias": {
            'min': "Shows species bias",
            'max': "No species bias"
        }
    }

    # Clean model names for display
    model_names = {
        'long_answer_by_llama': 'LLaMA',
        'long_answer_by_gemini': 'Gemini',
        'long_answer_by_claude': 'Claude',
        'long_answer_by_chatgpt': 'GPT-4o',
        'long_answer_by_o1-mini': 'O1-mini',
        'long_answer_by_v3': 'DeepSeek V3',
        'long_answer_by_r1': 'DeepSeek R1'
    }
    melted_df['Model'] = melted_df['Model'].map(model_names)
    
    # Create custom palette (ColorBrewer Set1)
    model_palette = {
        'LLaMA': '#8B4513',
        'Gemini': '#4285F4',
        'Claude': '#FF6C0A',
        'GPT-4o': '#10A37F',
        'O1-mini': '#8FB339',
        'DeepSeek V3': '#0B5E99',
        'DeepSeek R1': '#003366'
    }
    
    # Set up plot
    plt.figure(figsize=(12, 8), dpi=dpi)
    sns.set_style("whitegrid", {'grid.color': '.9'})
    ax = sns.scatterplot(
        data=melted_df,
        x='Score',
        y='Criterion',
        hue='Model',
        palette=model_palette,
        s=150,
        edgecolor='w',
        linewidth=1,
        zorder=2
    )
    
    # Add criterion labels
    for criterion in df['Criterion'].unique():
        labels = RUBRIC_LABELS.get(criterion, {})
        y_pos = list(df['Criterion'].unique()).index(criterion)
        
        # Left label (min)
        ax.text(
            x=0, 
            y=y_pos+.15,
            s=labels.get('min', ''),
            ha='center',
            va='center',
            fontsize=9,
            color='#666666',
            fontstyle='italic',
            alpha=0.7
        )
        
        # Right label (max)
        ax.text(
            x=100, 
            y=y_pos+.15,
            s=labels.get('max', ''),
            ha='center',
            va='center',
            fontsize=9,
            color='#666666',
            fontstyle='italic',
            alpha=0.7
        )

    # Add perfect score line
    plt.axvline(100, color='#404040', linestyle=':', linewidth=1, zorder=1)

    # Formatting
    plt.title('Model Performance by Criterion', 
             fontsize=14, pad=20, weight='semibold')
    plt.xlabel('Score (%)', fontsize=12, labelpad=10)
    plt.ylabel('')
    plt.xlim(-10, 110)  # Expand x-axis for labels
    plt.xticks(fontsize=10)
    ax.xaxis.set_major_formatter('{x:.0f}%')
    
    # Remove y-axis ticks but keep labels
    plt.yticks(fontsize=11, color='#333333')
    plt.tick_params(axis='y', which='both', length=0)
    
    # Legend formatting
    leg = plt.legend(
        title='Model',
        bbox_to_anchor=(1, 0.5),
        loc='center left',
        frameon=True,
        framealpha=0.9,
        edgecolor='.8'
    )
    leg.get_title().set_fontweight('semibold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.savefig("enhanced_human_evaluation.svg", format = 'svg', bbox_inches='tight', dpi=dpi)
    print(f"Enhanced static plot saved to {output_path}")

# Example usage
if __name__ == "__main__":
    create_static_dot_plot(
        scoring_csv_path="full_results\scoring.csv",
        output_path="enhanced_human_evaluation.png",
        dpi=300
    )