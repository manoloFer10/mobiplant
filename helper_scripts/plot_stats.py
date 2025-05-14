import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import argparse
import plotly.graph_objects as go
from eval_utils import create_citation_bins, create_year_bins

def plot_year_bins(df):
    counts = df['year_bin'].value_counts().sort_index()
    counts.plot(kind='bar', figsize=(8,4))
    plt.title('Questions by Publication Year Bin')
    plt.xlabel('Year Range')
    plt.ylabel('Number of Questions')
    plt.tight_layout()
    plt.show()

def plot_citation_bins(df):
    counts = df['citation_bin'].value_counts().reindex(
        ["0", "1-10", "11-100", "101-500", "501-1000", "1001-1702"]
    )
    counts.plot(kind='bar', figsize=(8,4))
    plt.title('Questions by Citation Count Bin')
    plt.xlabel('Citation Range')
    plt.ylabel('Number of Questions')
    plt.tight_layout()
    plt.show()

def plot_area_sunburst(df):
    # 1) split at most once
    df[['primary','secondary']] = df['area'].str.split(' - ', n=1, expand=True)
    
    # 2) aggregate counts
    agg = df.groupby(['primary','secondary'], dropna=False).size().reset_index(name='count')
    
    labels = []
    ids    = []
    parents= []
    values = []
    
    for _, row in agg.iterrows():
        P, S, cnt = row['primary'], row['secondary'], row['count']
        
        if pd.isna(S):
            # ONLY primary → create a unique blank parent
            parent_id = f"__blank__{P}"
            # inner blank wedge
            labels.append("")            
            ids.append(parent_id)
            parents.append("")        
            values.append(cnt)
            
            # outer wedge with the real label
            labels.append(P + f'({cnt})')
            ids.append(P)              
            parents.append(parent_id)
            values.append(cnt)
        else:
            # real two‐level: ensure we add the primary node once
            if P not in ids:
                ids.append(P)
                parents.append("")
                # sum of all children under this primary
                total = agg.loc[agg.primary==P, 'count'].sum()
                labels.append(P + f'({total})')
                values.append(total)
            # now add the secondary wedge
            labels.append(S)
            ids.append(f"{P}→{S}")  # unique id
            parents.append(P)
            values.append(cnt)
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        ids=ids,
        parents=parents,
        values=values,
        branchvalues='total',
        insidetextorientation='radial'
    ))
    fig.update_layout(title='Questions by Area (Sunburst)')
    fig.show()


def plot_species_distribution(df, top_n=20):
    counts = df['normalized_plant_species'].value_counts().nlargest(top_n)
    counts.plot(kind='bar', figsize=(8,4))
    plt.title(f'Top {top_n} Plant Species')
    plt.xlabel('Plant Species')
    plt.ylabel('Number of Questions')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def main(json_path):
    df = pd.read_json(json_path)
    df_year = create_year_bins(df)
    df_cit = create_citation_bins(df)
    
    plot_year_bins(df_year)
    plot_citation_bins(df_cit)
    plot_area_sunburst(df)
    plot_species_distribution(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot QA dataset stats")
    parser.add_argument("json_path", help="Path to JSONL file of QA dataset")
    args = parser.parse_args()
    main(args.json_path)