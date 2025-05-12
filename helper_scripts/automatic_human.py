import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

MODEL_COLORS = {
    'LLaMA': '#8B4513',
    'Gemini': '#4285F4',
    'Claude': '#FF6C0A',
    'GPT-4o': '#10A37F',
    'O1-mini': '#8FB339',
    'DeepSeek V3': '#0B5E99',
    'DeepSeek R1': '#003366'
}

def plot_hybrid_evaluation(human_scoring_csv, auto_scores_csv, output_path):
    # Configure global style
    sns.set_style("whitegrid", {'grid.color': '.9'})
    
     # Load and process human evaluation data
    human_df = pd.read_csv(human_scoring_csv, index_col=0)
    human_df = human_df.drop('Overall', errors='ignore')  # Safer drop
    human_long = human_df.reset_index().melt(
        id_vars=['index'], 
        var_name='Model', 
        value_name='Human Score'
    ).rename(columns={'index': 'Criterion'})
    
    # Enhanced model name cleaning
    human_long['Model'] = (
        human_long['Model']
        .str.replace('long_answer_by_', '')
        .str.title()
        .str.replace(' ', '')
    )
    
    model_name_mapping = {
        'Gemini': 'Gemini',
        'Claude': 'Claude',
        'Chatgpt': 'GPT-4o',
        'O1-Mini': 'O1-mini',  # Handle case consistency
        'V3': 'DeepSeek V3',
        'R1': 'DeepSeek R1',
        'Llama': 'LLaMA'
    }
    human_long['Model'] = human_long['Model'].map(model_name_mapping)
    
    # Calculate average human scores
    human_avg = human_long.groupby('Model', as_index=False)['Human Score'].mean()
    
    # Load and process automatic evaluation data
    auto_df = pd.read_csv(auto_scores_csv)
    auto_metrics = [col for col in auto_df.columns if '_answer_accuracy_mean' in col]
    
    auto_long = pd.DataFrame([{
        'Model': col.split('_')[0],
        'Auto Score': auto_df[col].values[0],
        'Auto Std': auto_df[col.replace('_mean', '_std')].values[0]
    } for col in auto_metrics])
    
    
    # Safer merge with indicator
    merged_df = pd.merge(
        human_avg, 
        auto_long, 
        on='Model', 
        how='outer',  # Show all models from both sides
        indicator=True
    )
    
    # Filter only models present in both datasets
    merged_df = merged_df[merged_df['_merge'] == 'both'].drop(columns='_merge')
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    scatter = sns.scatterplot(
        data=merged_df,
        x='Human Score',
        y='Auto Score',
        hue='Model',
        palette=MODEL_COLORS,
        s=150,
        zorder=3
    )
    
    
    # Add identity line
    plt.plot([0, 100], [0, 100], linestyle='--', color='lightgray', linewidth=1.5)
    
    # Styling)
    plt.title('Human vs Automatic Performance', fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('Human Score (Average)')
    plt.ylabel('Automatic Accuracy')
    plt.xlim(40, 90)
    plt.ylim(40, 90)
    
    # Custom legend
    handles, labels = scatter.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], linestyle=':', color='gray', linewidth=1.5))
    
    plt.legend(
        handles=handles,
        labels=labels,
        loc='upper left',
        bbox_to_anchor=(1.05, 1),
        frameon=True,
        framealpha=0.9,
        facecolor='white'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight', transparent=False)
    plt.close()

# Example usage:
if __name__ == "__main__":
    human_csv = "full_results/scoring.csv"
    auto_csv = "full_results/overall.csv"
    output_img = "hybrid_evaluation_plot.svg"
    plot_hybrid_evaluation(human_csv, auto_csv, output_img)