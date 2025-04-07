import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CLEAN_MODEL_NAMES = {
    'long_answer_by_llama': 'Llama 3.1 405B',
    'long_answer_by_gemini': 'Gemini 1.5 Pro',
    'long_answer_by_claude': 'Claude 3.5 Sonnet',
    'long_answer_by_chatgpt': 'GPT-4o',
    'long_answer_by_o1-mini': 'o1-mini',
    'long_answer_by_v3': 'Deepseek V3',  # Assuming v3 is same as above
    'long_answer_by_r1': 'Deepseek R1'
}

CLEAN_CRITERIA = {
    'alignment_consensus': 'Alignment with Scientific Consensus',
    'correct_reasoning': 'Evidence of \nCorrect Reasoning',
    'hallucinated_content': 'Inclusion of \nHallucinated Content',
    'irrelevant_content': 'Inclusion of Irrelevant Content',
    'limitations': 'Acknowledgement of Limitations',
    'omission_information': 'Omission of Important Information',
    'reading_comprehension': 'Evidence of \nReading Comprehension',
    'species_bias': 'Species Bias'
}

def generate_spidergraph(data_path: str, output_path: str):
    """Generate a spider graph from the scoring CSV."""
    # Read and clean data
    df = pd.read_csv(data_path, index_col=0)
    
    # Clean criteria names and filter
    df = df.rename(index=CLEAN_CRITERIA)
    df = df.loc[list(CLEAN_CRITERIA.values())]  # Maintain order
    
    # Clean model names
    df = df.rename(columns=CLEAN_MODEL_NAMES)
    
    # Prepare plot
    num_vars = len(df.index)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 100)
    ax.set_rticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], 
                      fontsize=10, color='grey')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(df.index, fontsize=12, color='black')
    plt.xticks(rotation=45, ha='center')
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    
    # Plot each model
    colors = plt.cm.tab10.colors
    for i, model in enumerate(df.columns):
        values = df[model].values.flatten().tolist()
        values += values[:1]
        color = colors[i % len(colors)]
        
        ax.plot(angles, values, color=color, linewidth=2,
                marker='o', markersize=4, label=model)
        ax.fill(angles, values, color=color, alpha=0.2)
    
    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
             ncol=4, fontsize=12, frameon=False)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Spider chart saved to: {output_path}")

# Example usage:
if __name__ == "__main__":
    generate_spidergraph(
        data_path="scoring.csv",
        output_path="model_comparison_spider.png"
    )