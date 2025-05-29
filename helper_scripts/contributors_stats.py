import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import numpy as np
import matplotlib as mpl
import geopandas as gpd
import matplotlib.colors # Required for ListedColormap

# Load your dataset
df = pd.read_csv(r'data/contributors/contributors.csv')

# Clean age column
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
age_clean = df['Age'].dropna()

# Define the plant-green gradient
# Original: greens_cmap = mpl.colors.LinearSegmentedColormap.from_list(
#    'greens', ['#edf8e9', '#74c476', '#006d2c']
# )
# Adjusted for more intensity at the lower end:
greens_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'greens', ['#c7e9c0', '#74c476', '#006d2c'] # Changed '#edf8e9' to '#c7e9c0'
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
plt.show()

# Load contributor areas data
df_areas = pd.read_csv(r'data/contributors/contributors_areas.csv')

# Remove existing Area and Sub-Area plots if this new plot replaces them
# # Plot for Areas
# plt.figure(figsize=(12, 8))
# area_counts = df_areas['area'].value_counts()
# area_bar_colors = get_colors(area_counts.values)
# bars_area = plt.barh(area_counts.index, area_counts.values, color=area_bar_colors, edgecolor='black')
# for bar, cnt in zip(bars_area, area_counts.values):
#     plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
#              str(int(cnt)), ha='left', va='center', fontsize=8)
# plt.title('Contributors by Area', fontweight='bold')
# plt.xlabel('Count')
# plt.ylabel('Area')
# plt.gca().invert_yaxis() # To display the highest count at the top
# plt.tight_layout()
# plt.show()

# # Plot for Sub-areas
# plt.figure(figsize=(12, 10)) # Adjusted figure size for potentially more sub-areas
# sub_area_counts = df_areas['sub-area'].value_counts()
# sub_area_bar_colors = get_colors(sub_area_counts.values)
# bars_sub_area = plt.barh(sub_area_counts.index, sub_area_counts.values, color=sub_area_bar_colors, edgecolor='black')
# for bar, cnt in zip(bars_sub_area, sub_area_counts.values):
#     plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
#              str(int(cnt)), ha='left', va='center', fontsize=8)
# plt.title('Contributors by Sub-Area', fontweight='bold')
# plt.xlabel('Count')
# plt.ylabel('Sub-Area')
# plt.gca().invert_yaxis() # To display the highest count at the top
# plt.tight_layout()
# plt.show()

# New Sunburst chart for Areas and Sub-Areas
import plotly.express as px

if not df_areas.empty:
    # Prepare data for sunburst: counts of sub-areas within areas
    # Ensure 'area' and 'sub-area' are treated as strings to avoid issues with missing values being floats
    df_areas_copy = df_areas.copy()
    df_areas_copy['area'] = df_areas_copy['area'].astype(str)
    
    # Split 'area' by '-' and keep the first part
    df_areas_copy['area'] = df_areas_copy['area'].apply(lambda x: x.split('-', 1)[0].strip())
    
    df_areas_copy['sub-area'] = df_areas_copy['sub-area'].astype(str)
    
    # Handle potential NaN or placeholder strings like 'nan' if they exist after conversion
    df_areas_copy.replace('nan', 'N/A', inplace=True)


    sunburst_data = df_areas_copy.groupby(['area', 'sub-area'], observed=False).size().reset_index(name='count')

    # Define the custom Plotly continuous color scale from your greens_cmap
    # greens_cmap is defined earlier as:
    # mpl.colors.LinearSegmentedColormap.from_list('greens', ['c7e9c0', '#74c476', '#006d2c'])
    plotly_greens_scale = [
        [0.0, '#c7e9c0'],  # Start of your greens_cmap
        [0.5, '#74c476'],  # Middle of your greens_cmap
        [1.0, '#006d2c']   # End of your greens_cmap
    ]

    fig_sunburst = px.sunburst(sunburst_data,
                               path=['area', 'sub-area'],
                               values='count',
                               color='count', # Color segments by their count
                               color_continuous_scale=plotly_greens_scale,
                               title='Contributors by Area and Sub-Area',
                               hover_data={'count': True}) # Show count on hover

    fig_sunburst.update_layout(margin = dict(t=50, l=25, r=25, b=25))
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
    fig_sunburst.show(config=config)
else:
    print("Contributor areas data is empty. Skipping sunburst chart.")

# Additional plot: Choropleth map for Country of Residence

# Prepare country counts for merging
country_counts = df['Country of residence'].value_counts()
country_counts_df = country_counts.reset_index()
country_counts_df.columns = ['country_name', 'count']

# Load a world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Merge map data with country counts
# Note: Country name discrepancies between your data and the map data might occur.
# You may need to preprocess country_name or world.name for better matching.
merged_map = world.merge(country_counts_df, left_on='name', right_on='country_name', how='left')

# Fill NaN counts with 0 (for countries in the map but not in your data)
merged_map['count'] = merged_map['count'].fillna(0)

# Define bins and labels for the counts
# Bins: 1-5, 6-10, 11-15, 16-20, 21+
bins = [0, 5, 10, 15, 20, np.inf]
labels = ['1-5', '6-10', '11-15', '16-20', '21+']

# Apply binning
# Countries with 0 counts will result in NaN here, and will be colored by 'missing_kwds'
merged_map['count_bin'] = pd.cut(merged_map['count'],
                                 bins=bins,
                                 labels=labels,
                                 right=True,
                                 include_lowest=False) # Counts of 0 will not be included in '1-5'

# Define colors for the bins using the existing greens_cmap
# greens_cmap is already defined: mpl.colors.LinearSegmentedColormap.from_list('greens', ['#edf8e9', '#74c476', '#006d2c'])
num_bins = len(labels)
map_colors = [greens_cmap(i / max(1, num_bins - 1)) for i in range(num_bins)] # Get 5 colors from the cmap
custom_cmap = matplotlib.colors.ListedColormap(map_colors)

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(20, 12))
merged_map.plot(column='count_bin',
                ax=ax,
                legend=True,
                categorical=True, # Treat 'count_bin' as categorical data for distinct colors
                cmap=custom_cmap,
                missing_kwds={
                    "color": "lightgrey",
                    # "label": "0 or No Data", # Removed this line
                },
                legend_kwds={'title': "Number of Contributors", 'loc': 'lower left'})

ax.set_title('Contributors by Country of Residence', fontdict={'fontsize': 20, 'fontweight': 'bold'})
ax.set_axis_off() # Remove axis ticks and labels

plt.tight_layout()
plt.show()
