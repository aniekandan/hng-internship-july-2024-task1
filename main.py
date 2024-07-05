import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd

# Load the geocoded dataset
aks_df = pd.read_csv('data/AKWA IBOM_crosschecked_geocoded.csv')

# Extract latitude and longitude columns
lat_lon = aks_df[['latitude', 'longitude']].values

# Calculate the distance matrix between all polling units
dist_matrix = distance_matrix(lat_lon, lat_lon)

# Define the radius for neighbors (in kilometers)
radius_km = 1.0

# Initialize list to store outlier scores
outlier_scores = []

# Iterate over each polling unit to calculate the outlier scores
for idx, observation in aks_df.iterrows():
    neighbors_of_idx = aks_df[(dist_matrix[idx] <= radius_km) & (aks_df.index != idx)]
    
    # Calculate the outlier score for each party in the neighbourhood
    apc_outlier = abs(observation['APC'] - neighbors_of_idx['APC'].mean()) if not neighbors_of_idx.empty else 0
    lp_outlier = abs(observation['LP'] - neighbors_of_idx['LP'].mean()) if not neighbors_of_idx.empty else 0
    pdp_outlier = abs(observation['PDP'] - neighbors_of_idx['PDP'].mean()) if not neighbors_of_idx.empty else 0
    nnpp_outlier = abs(observation['NNPP'] - neighbors_of_idx['NNPP'].mean()) if not neighbors_of_idx.empty else 0
    
    # Store the results
    outlier_scores.append({
        'Address': observation['address'],
        'Latitude': observation['latitude'],
        'Longitude': observation['longitude'],
        'APC_outlier': apc_outlier,
        'LP_outlier': lp_outlier,
        'PDP_outlier': pdp_outlier,
        'NNPP_outlier': nnpp_outlier,
        'Neighbors': neighbors_of_idx['address'].tolist()
    })

# Convert the results list to a DataFrame
outlier_scores_df = pd.DataFrame(outlier_scores)

def get_sorted_outliers(df, party):
    # Select relevant columns for the given party
    selected_columns = ["Address", "Latitude", "Longitude", f"{party}_outlier"]
    party_df = df[selected_columns]
    # Sort the DataFrame by the party's outlier score in descending order
    sorted_df = party_df.sort_values(f'{party}_outlier', ascending=False)
    return sorted_df

def display_top_outliers(dfs_dict, qty):
    # Set display options for better readability
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # Create a dictionary with the top 'qty' outliers for each party
    top_n_dict = {party: df.head(qty) for party, df in dfs_dict.items()}
    
    # Display the top outliers for each party
    for party, party_df in top_n_dict.items():
        print(f"Top {qty} {party} outliers_________________________________:")
        print(party_df)
        print()

# List of parties
parties = ['APC', 'LP', 'PDP', 'NNPP']

# Create a dictionary of sorted DataFrames for each party
df_dict = {party: get_sorted_outliers(outlier_scores_df, party) for party in parties}

# Display the top 3 outliers for each party
display_top_outliers(df_dict, qty=3)

def save_to_excel(dfs_dict, file_path):
    # Use ExcelWriter to create a new Excel file
    with pd.ExcelWriter(file_path) as writer:
        # Iterate through each DataFrame in the dictionary
        for sheet_name, df in dfs_dict.items():
            # Write each DataFrame to a separate sheet in the Excel file
            # Sheet name is the party name, and index is not included
            df.to_excel(writer, sheet_name=sheet_name, index=False)

# Define the path for the output Excel file
excel_file_path = 'data/output/Polling_Unit_Outlier_Scores.xlsx'

# Call the function to save the DataFrames to the Excel file
# df_dict contains the sorted outlier scores for each party
save_to_excel(df_dict, excel_file_path)

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Define the parties and their corresponding colors
parties = ['APC', 'PDP', 'LP', 'NNPP']
colors = ['blue', 'green', 'red', 'purple']

# Iterate through parties and create a histogram for each
for i, (party, color) in enumerate(zip(parties, colors)):
    row = i // 2  # Calculate row index
    col = i % 2   # Calculate column index
    # Create histogram for the current party
    sns.histplot(outlier_scores_df[f'{party}_outlier'], bins=20, ax=axes[row, col], color=color)
    # Set title and labels for the subplot
    axes[row, col].set_title(f'Outlier Scores for {party}')
    axes[row, col].set_xlabel('Outlier Score')
    axes[row, col].set_ylabel('Frequency')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# Set up the map projection
fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})

# Add map features
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# The coordinate extents of the bounding box of Akwa Ibom state
northern_extent, western_extent = 5.5, 7.3
southern_extent, eastern_extent = 4.4, 8.4

# Set the extent of the map to focus on Akwa Ibom using provided bounding box coordinates
ax.set_extent([western_extent, eastern_extent, southern_extent, northern_extent], crs=ccrs.PlateCarree())

# Load the boundary file using Geopandas    
boundary_file = 'data/ng_borders.json'  # Path to the boundary file
boundary_gdf = gpd.read_file(boundary_file)

# Filter for Akwa Ibom state
state_name = 'Akwa Ibom'
akwa_ibom_boundary = boundary_gdf[boundary_gdf['name'] == state_name]

# Plot the boundary
akwa_ibom_boundary.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1)

# Define the parties
parties = ['APC', 'PDP', 'LP', 'NNPP']

# Define colors for each party
colors = ['blue', 'green', 'red', 'purple']

# Plot top outliers for each party
for party, color in zip(parties, colors):
    top_outliers = outlier_scores_df.nlargest(5, f'{party}_outlier')
    ax.scatter(top_outliers['Longitude'], top_outliers['Latitude'], 
                color=color, s=50, label=party, transform=ccrs.PlateCarree())

# Add gridlines
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

# Add a title and legend
plt.title('Top 5 Outliers for Each Party in Akwa Ibom', fontsize=16)
plt.legend()

plt.tight_layout()
plt.show()

