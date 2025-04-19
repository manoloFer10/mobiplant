import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import patheffects

def plot_hybrid_evaluation(human_scoring_csv, auto_scores_csv, output_path):
    # Custom green palette based on #2EAA48
    PALETTE = {
        'primary': '#2EAA48',    # Your specified vibrant green
        'secondary': '#6AC87A',  # Complementary light green
        'accent': '#1C6D34',     # Dark green for contrast
        'background': '#F5FCF7', # Ultra-light background
        'text': '#1A3D24'        # Deep green for text
    }
    
    # Configure global style
    sns.set_style("whitegrid")
    plt.rcParams['axes.edgecolor'] = PALETTE['text']
    plt.rcParams['axes.labelcolor'] = PALETTE['text']
    plt.rcParams['text.color'] = PALETTE['text']
    
    # Load and prepare human evaluation data
    human_df = pd.read_csv(human_scoring_csv, index_col=0)
    human_df = human_df.drop('Overall')
    human_long = human_df.reset_index().melt(
        id_vars=['index'], 
        var_name='Model', 
        value_name='Human Score'
    ).rename(columns={'index': 'Criterion'})
    
    # Clean and map model names
    human_long['Model'] = human_long['Model'].str.replace('long_answer_by_', '').str.title()
    model_name_mapping = {
        'Gemini': 'Gemini',
        'Claude': 'Claude',
        'Chatgpt': 'GPT-4o',
        'O1-mini': 'O1-mini',
        'V3': 'DeepSeek V3',
        'R1': 'DeepSeek R1',
        'Llama': 'LLaMA'
    }
    human_long['Model'] = human_long['Model'].map(model_name_mapping)

    # Load and prepare automatic evaluation data
    auto_df = pd.read_csv(auto_scores_csv).iloc[0:1]
    auto_metrics = [col for col in auto_df.columns if '_total_accuracy_mean' in col]
    
    auto_long = pd.DataFrame([
        {
            'Model': col.split('_')[0],
            'Auto Score': auto_df[col].values[0],
            'Auto Std': auto_df[col.replace('_mean', '_std')].values[0]
        }
        for col in auto_metrics
    ])

    # Merge datasets
    merged_df = pd.merge(human_long, auto_long, on='Model')
    
    
    # Create figure with enhanced styling
    plt.figure(figsize=(16, 10))
    
    # Human evaluation boxplot
    box = sns.boxplot(
        x='Model', 
        y='Human Score', 
        data=merged_df,
        color=PALETTE['primary'],
        width=0.6,
        linewidth=2.5,
        flierprops=dict(
            markerfacecolor=PALETTE['secondary'],
            marker='D',
            markersize=10,
            markeredgecolor=PALETTE['accent']
        ),
        boxprops=dict(edgecolor=PALETTE['accent']),
        whiskerprops=dict(color=PALETTE['accent']),
        capprops=dict(color=PALETTE['accent']),
        medianprops=dict(color=PALETTE['background']),
        zorder=3
    )
    
    # Individual criteria points
    # strip = sns.stripplot(
    #     x='Model', 
    #     y='Human Score', 
    #     data=merged_df,
    #     color=PALETTE['secondary'],
    #     size=10,
    #     alpha=0.9,
    #     jitter=0.25,
    #     edgecolor=PALETTE['accent'],
    #     linewidth=1,
    #     zorder=2
    # )
    
    # Automatic evaluation markers
    auto_plot = sns.pointplot(
        x='Model', 
        y='Auto Score', 
        data=merged_df,
        color=PALETTE['accent'],
        markers='o',
        scale=2.5,
        linestyles='',
        errorbar=None,
        zorder=4,
        markeredgecolor=PALETTE['text'],
        markerfacecolor=PALETTE['text']
    )
    
    # Enhanced error bars
    # for i, model in enumerate(merged_df['Model'].unique()):
    #     model_data = merged_df[merged_df['Model'] == model].iloc[0]
    #     plt.errorbar(
    #         x=i,
    #         y=model_data['Auto Score'],
    #         yerr=model_data['Auto Std'],
    #         color=PALETTE['accent'],
    #         fmt='none',
    #         capsize=12,
    #         capthick=3,
    #         elinewidth=3,
    #         alpha=0.9,
    #         zorder=5
    #     )
    
    # Final styling
    plt.gca().set_facecolor(PALETTE['background'])
    plt.grid(color=PALETTE['secondary'], linestyle=':', alpha=0.4)
    
    plt.title('Human Criteria vs Automatic Performance',
             fontsize=18, pad=25, fontweight='bold')
    plt.ylim(0, 100)

    # Custom legend
    legend_elements = [
        plt.Line2D([], [], color=PALETTE['primary'], lw=4, label='Human Criteria Distribution'),
        # plt.Line2D([], [], color=PALETTE['secondary'], marker='o', linestyle='None',
        #           markersize=10, label='Individual Criteria Scores'),
        plt.Line2D([], [], color=PALETTE['accent'], marker='o', linestyle='None',
                  markersize=10, label='Automatic Scores')
    ]
    
    plt.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, -0.12), ncol=2,
              frameon=True, framealpha=0.9,
              facecolor=PALETTE['background'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.savefig('hybrid_evaluation_plot.svg', format='svg', dpi=300, bbox_inches='tight', transparent=False)
# Example usage:
if __name__ == "__main__":
    human_csv = "full_results\scoring.csv"
    auto_csv = "full_results\overall.csv"
    output_img = "hybrid_evaluation_plot.png"
    plot_hybrid_evaluation(human_csv, auto_csv, output_img)