import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import numpy as np

# Load your dataset
df = pd.read_csv('data\contributors.csv')

# Clean age column
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
age_clean = df['Age'].dropna()

# Set style for plots
sns.set_style("whitegrid")
plt.figure(figsize=(18, 12))

# 1. Age Distribution with 5-year bins (no vertical grid lines)
plt.subplot(2, 2, 1)
if not age_clean.empty:
    min_age = int(age_clean.min())
    max_age = int(age_clean.max())
    bins = np.arange(min_age //5 *5, max_age +5, 5)
    hist = sns.histplot(age_clean, kde=False, bins=bins, color='skyblue')
    
    # Add labels to each bar
    for patch in hist.patches:
        height = patch.get_height()
        if height > 0:
            x_center = patch.get_x() + patch.get_width()/2
            hist.annotate(f'{int(height)}', 
                         (x_center, height), 
                         ha='center', va='bottom',
                         fontsize=8)
    
    # Format x-axis labels and remove vertical grid
    bin_labels = [f'{int(bins[i])}-{int(bins[i+1]-1)}' for i in range(len(bins)-1)]
    plt.xticks(bins[:-1] + 2.5, bin_labels, rotation=45)
    plt.title('Age Distribution')
    plt.gca().grid(False, axis='x')  # Remove vertical grid lines
else:
    plt.text(0.5, 0.5, 'No age data available', ha='center', va='center')
    plt.title('Age Distribution - No Valid Data')

plt.xlabel('Age Group')
plt.ylabel('Count')

# 2. Gender Identity Distribution (unchanged)
plt.subplot(2, 2, 2)

# Create list of required categories (maintain this order)
required_categories = [
    'Cis Female',
    'Cis Male',
    'Trans Female', 
    'Trans Male', 
    'Not Binary',
    'Agender',
    'Prefer not to say',
]

# Process gender counts
gender_counts = df['Gender identity'].value_counts()
gender_counts = gender_counts.reindex(required_categories, fill_value=0)

# Preserve original "Other" responses
original_other = [x for x in df['Gender identity'].unique() 
                 if x not in required_categories and pd.notna(x)]
if original_other:
    other_count = gender_counts.get('Other', 0) + df['Gender identity'].isin(original_other).sum()
    gender_counts['Other'] = other_count

# Filter to only include required categories and other
gender_counts = gender_counts[required_categories + ['Other'] if 'Other' in gender_counts else required_categories]

# Create plot with vertical orientation
ax = sns.barplot(x=gender_counts.index, y=gender_counts.values, 
                order=required_categories, palette='viridis')

plt.title('Gender Identity Distribution')
plt.xlabel('Gender Identity')
plt.ylabel('Count')

# Rotate x-labels and adjust layout
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# 3. Nationality Analysis - Smaller fonts
plt.subplot(2, 2, 3)
try:
    df['Nationality/ies'] = df['Nationality/ies'].apply(literal_eval)
except:
    df['Nationality/ies'] = df['Nationality/ies'].str.split(',\s*')

nationalities = df['Nationality/ies'].explode().str.strip().dropna()
nationality_counts = nationalities.value_counts()

ax = sns.barplot(x=nationality_counts.values, y=nationality_counts.index, 
                palette='viridis', dodge=False)
plt.title("Contributor's Nationalities")
plt.xlabel('Count')
plt.ylabel('Nationality')
ax.tick_params(axis='y', labelsize=8)  # Smaller y-axis labels

# 4. Country of Residence - Smaller fonts, different color
plt.subplot(2, 2, 4)
country_counts = df['Country of residence'].value_counts()
ax = sns.barplot(x=country_counts.values, y=country_counts.index, 
                palette='viridis', dodge=False)  # Reversed viridis
plt.title("Contributor's Countries of Residence")
plt.xlabel('Count')
plt.ylabel('Country')
ax.tick_params(axis='y', labelsize=8)  # Smaller y-axis labels

plt.tight_layout()
plt.show()

# Print statistics
print("Descriptive Statistics for Age:")
print(f"Valid entries: {len(age_clean)}")
print(f"'Prefer not to say' responses: {df['Age'].isna().sum()}")
if not age_clean.empty:
    print(age_clean.describe())