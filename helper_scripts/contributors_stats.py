import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import numpy as np
import matplotlib as mpl

# Load your dataset
df = pd.read_csv(r'data\contributors.csv')

# Clean age column
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
age_clean = df['Age'].dropna()

# Define the plant-green gradient
greens_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'greens', ['#edf8e9', '#74c476', '#006d2c']
)

def get_colors(vals):
    """Map a sequence of values to our green gradient."""
    norm = mpl.colors.Normalize(vmin=min(vals), vmax=max(vals))
    return [greens_cmap(norm(v)) for v in vals]

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(18, 12))

# 1. Age Distribution with 5-year bins
plt.subplot(2, 2, 3)
if not age_clean.empty:
    min_age = int(age_clean.min())
    max_age = int(age_clean.max())
    bins = np.arange(min_age //5 *5, max_age +5, 5)
    counts, edges = np.histogram(age_clean, bins=bins)
    bin_centers = edges[:-1] + np.diff(edges)/2

    # Use our gradient for the bars
    bar_colors = get_colors(counts)
    bars = plt.bar(bin_centers, counts, width=np.diff(edges), color=bar_colors, edgecolor='black')
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            plt.text(bar.get_x() + bar.get_width()/2, cnt,
                     str(int(cnt)), ha='center', va='bottom', fontsize=8)

    # Re-label x-ticks
    labels = [f'{int(edges[i])}-{int(edges[i+1]-1)}' for i in range(len(edges)-1)]
    plt.xticks(bin_centers, labels, rotation=45, ha='right')
    plt.gca().grid(False, axis='x')
else:
    plt.text(0.5, 0.5, 'No age data available', ha='center', va='center')
plt.title('Age Distribution', fontweight='bold')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.tight_layout()

# 2. Gender Identity Distribution
plt.subplot(2, 2, 1)
required = ['Cis Female','Cis Male','Trans Female','Trans Male','Not Binary','Agender','Prefer not to say']
gender_counts = df['Gender identity'].value_counts().reindex(required, fill_value=0)
others = [x for x in df['Gender identity'].unique() if x not in required and pd.notna(x)]
if others:
    gender_counts['Other'] = df['Gender identity'].isin(others).sum()

vals = gender_counts.values
bar_colors = get_colors(vals)
bars = plt.bar(gender_counts.index, vals, color=bar_colors, edgecolor='black')
for bar, cnt in zip(bars, vals):
    if cnt > 0:
        plt.text(bar.get_x() + bar.get_width()/2, cnt,
                 str(int(cnt)), ha='center', va='bottom', fontsize=8)

plt.title('Gender Identity Distribution', fontweight='bold')
plt.xlabel('Gender Identity')
plt.ylabel('Count')
plt.xticks(rotation=22.5, ha='right')
plt.tight_layout()

# 3. Nationality Analysis
plt.subplot(2, 2, 4)
try:
    df['Nationality/ies'] = df['Nationality/ies'].apply(literal_eval)
except:
    df['Nationality/ies'] = df['Nationality/ies'].str.split(',\s*', regex=True)
nationalities = df['Nationality/ies'].explode().str.strip().dropna()
nat_counts = nationalities.value_counts()

vals = nat_counts.values
bar_colors = get_colors(vals)
bars = plt.barh(nat_counts.index, vals, color=bar_colors, edgecolor='black')
for bar, cnt in zip(bars, vals):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             str(int(cnt)), ha='left', va='center', fontsize=8)

plt.title("Contributors Nationalities", fontweight='bold')
plt.xlabel('Count')
plt.ylabel('Nationality')
plt.tick_params(axis='y', labelsize=8)
plt.tight_layout()

# 4. Country of Residence
plt.subplot(2, 2, 2)
country_counts = df['Country of residence'].value_counts()

vals = country_counts.values
bar_colors = get_colors(vals)
bars = plt.barh(country_counts.index, vals, color=bar_colors, edgecolor='black')
for bar, cnt in zip(bars, vals):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             str(int(cnt)), ha='left', va='center', fontsize=8)

plt.title("Contributors Countries of Residence", fontweight='bold')
plt.xlabel('Count')
plt.ylabel('Country')
plt.tick_params(axis='y', labelsize=8)

plt.tight_layout()
plt.show()
