####
# No tiene sentido porque solo hay 1 anotador por pregunta.
####

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_annotator_flow_plot(distribution_csv_path, output_path, dpi=300):
    # Load and prepare data
    df = pd.read_csv(distribution_csv_path)
    df = df.rename(columns={'Unnamed: 0': 'Criterion', 'Unnamed: 1': 'Rating'})
    
    # Your rubric scores
    RUBRIC_SCORES = {
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

    # Clean model names
    model_names = {
        'long_answer_by_llama': 'LLaMA',
        'long_answer_by_gemini': 'Gemini',
        'long_answer_by_claude': 'Claude',
        'long_answer_by_chatgpt': 'ChatGPT',
        'long_answer_by_o1-mini': 'O1-Mini',
        'long_answer_by_v3': 'DeepSeek V3',
        'long_answer_by_r1': 'DeepSeek R1'
    }
    df.columns = [model_names.get(col, col) for col in df.columns]

    # Create score-based color mapping
    def get_score_color(criterion, rating):
        score = RUBRIC_SCORES.get(criterion, {}).get(str(rating), None)
        if pd.isna(rating) or pd.isna(score):
            return '#CCCCCC'  # Gray for missing data
        return {
            1: '#77dd77',    # Green
            0.5: '#ffe66d',  # Yellow
            0: '#ff6b6b'     # Red
        }.get(score, '#999999')

    # Set up plot
    plt.figure(figsize=(16, 12), dpi=dpi)
    sns.set_style("whitegrid", {'grid.color': '.9'})
    ax = plt.gca()

    # Layout parameters
    MODEL_WIDTH = 1.2
    CRITERION_WIDTH = 2.5
    SPACING = 1.2
    MODEL_SPACE = 4.0

    # Create vertical offsets
    models = list(model_names.values())
    model_positions = {model: i * (len(df.Criterion.unique()) * (SPACING + 0.7)) 
                      for i, model in enumerate(models)}

    # Create flow bands
    for model_idx, model in enumerate(models):
        y_level = model_positions[model]
        
        for criterion in df.Criterion.unique():
            criterion_df = df[(df.Criterion == criterion)].sort_values(
                by='Rating',
                key=lambda x: x.map(RUBRIC_SCORES.get(criterion, {})),
                ascending=False
            )
            total = criterion_df[model].sum()
            
            if total == 0:
                continue
                
            # Model to Criterion flow
            plt.fill_betweenx(
                [y_level, y_level + SPACING],
                x1=model_idx * MODEL_SPACE,
                x2=model_idx * MODEL_SPACE + MODEL_WIDTH,
                color='#f0f0f0',
                alpha=0.4,
                edgecolor='none'
            )
            
            # Criterion to Rating flow
            x_start = model_idx * MODEL_SPACE + MODEL_WIDTH
            current_x = x_start
            
            for _, row in criterion_df.iterrows():
                rating = row['Rating']
                count = row[model]
                if count == 0:
                    continue
                
                color = get_score_color(criterion, rating)
                proportion = count / total
                band_width = CRITERION_WIDTH * proportion
                
                # Add band
                plt.fill_betweenx(
                    [y_level, y_level + SPACING],
                    x1=current_x,
                    x2=current_x + band_width,
                    color=color,
                    alpha=0.7,
                    edgecolor='none'
                )
                
                # Add score label
                if proportion > 0.15:  # Only label large enough segments
                    score = RUBRIC_SCORES.get(criterion, {}).get(str(rating), '?')
                    plt.text(
                        current_x + band_width/2,
                        y_level + SPACING/2,
                        f"{score}",
                        ha='center',
                        va='center',
                        color='white' if score in [0, 0.5] else 'black',
                        fontsize=9,
                        weight='bold'
                    )
                
                current_x += band_width

            y_level += SPACING + 0.7

    # Add model labels
    for model_idx, model in enumerate(models):
        plt.text(
            model_idx * MODEL_SPACE + MODEL_WIDTH/2,
            model_positions[model] - 4,
            model,
            ha='center',
            va='center',
            fontsize=12,
            color='#333333',
            weight='semibold',
            rotation=45
        )

    # Formatting
    plt.xlim(-1, len(models) * MODEL_SPACE)
    plt.ylim(-5, max(model_positions.values()) + len(df.Criterion.unique()) * (SPACING + 0.7))
    plt.axis('off')
    
    # Create semantic legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, color='#77dd77', label='Good (Score 1)'),
        plt.Rectangle((0,0),1,1, color='#ffe66d', label='Medium (Score 0.5)'),
        plt.Rectangle((0,0),1,1, color='#ff6b6b', label='Poor (Score 0)'),
        plt.Rectangle((0,0),1,1, color='#cccccc', label='Missing/Unknown'),
        plt.Rectangle((0,0),1,1, color='#f0f0f0', alpha=0.4, label='Model → Criterion')
    ]
    
    plt.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        frameon=True,
        framealpha=0.9,
        edgecolor='#333333',
        fontsize=10
    )

    plt.title('Annotated Judgment Flow: Model Performance by Rubric Scores\n',
             fontsize=14, pad=25, weight='semibold')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    print(f"Enhanced flow plot saved to {output_path}")

if __name__ == "__main__":
    create_annotator_flow_plot(
        distribution_csv_path="full_results\criteria_distribution.csv",
        output_path="annotator_flow.png",
        dpi=300
    )