#!/usr/bin/env python
# coding: utf-8

# This notebook can be used to Cluster the Network properties using Gaussian Mixture Models, use Bayesian Informatio criterion to determine the optimal number of clusters.
# 
# Then use the fuzzy logic rules to manually allocate labels to the clusters, to classify the streams based on Network connectivity

# In[ ]:


import geopandas as gpd
import pandas as pd


# In[13]:


# Load all three shapefiles into separate GeoDataFrames
gdf1 = gpd.read_file('./NorthEast/Hydrography/NHDFlowline.shp')
gdf2 = gpd.read_file('./Mid-Atlantic/Hydrography/NHDFlowline.shp')
gdf3 = gpd.read_file('./South Atlantic North/Hydrography/NHDFlowline.shp')
gdf4 = gpd.read_file('./South Atlantic South/Hydrography/NHDFlowline.shp')
gdf5 = gpd.read_file('./South Atlantic West/Hydrography/NHDFlowline.shp')



# In[4]:


# Load all three shapefiles into separate GeoDataFrames
gdf6 = gpd.read_file('./Ohio/Hydrography/NHDFlowline.shp')
gdf7 = gpd.read_file('./Tennessee/Hydrography/NHDFlowline.shp')
gdf8 = gpd.read_file('./Upper Mississipi/Hydrography/NHDFlowline.shp')
gdf9 = gpd.read_file('./Lower Mississipi/Hydrography/NHDFlowline.shp')
gdf10 = gpd.read_file('./Upper Missouri/Hydrography/NHDFlowline.shp')
gdf12 = gpd.read_file('./Ark Red White/Hydrography/NHDFlowline.shp')





# In[147]:


gdf13 = gpd.read_file('./Upper Colarado/Hydrography/NHDFlowline.shp')
gdf14 = gpd.read_file('./Lower Colarado/Hydrography/NHDFlowline.shp')
gdf15 = gpd.read_file('./Texas/Hydrography/NHDFlowline.shp')
gdf16 = gpd.read_file('./Rio Grande/Hydrography/NHDFlowline.shp')





# In[7]:


gdf11 = gpd.read_file('./Lower Missouri/Hydrography/NHDFlowline.shp')


# In[14]:


# Combine all three GeoDataFrames into one using pd.concat
gdf = pd.concat([gdf1, gdf2, gdf3, gdf4, gdf5], ignore_index=True)


# In[8]:


# Combine all three GeoDataFrames into one using pd.concat
gdf = pd.concat([gdf6, gdf7, gdf8, gdf9, gdf10, gdf11, gdf12], ignore_index=True)


# In[148]:


gdf = pd.concat([gdf13, gdf14, gdf15, gdf16], ignore_index=True)


# In[23]:


gdf.tail()


# The Mississippi river basin data is considered for classifying the streams

# In[10]:


import geopandas as gpd
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Create an empty graph
G = nx.Graph()

# Create a spatial index for the GeoDataFrame to speed up intersection checks
gdf_sindex = gdf.sindex

# Add nodes and edges to the graph with progress tracking
print("Adding nodes and edges to the graph...")
for idx, row in tqdm(gdf.iterrows(), total=gdf.shape[0], desc="Processing Nodes and Edges"):
    point = row['geometry'].centroid
    G.add_node(idx, x=point.x, y=point.y)

    # Get potential neighbors using the spatial index
    possible_neighbors = list(gdf_sindex.intersection(row['geometry'].bounds))
    neighbors = [n for n in possible_neighbors if n != idx and gdf.loc[n, 'geometry'].intersects(row['geometry'])]

    # Add edges between the current node and its neighbors
    for neighbor in neighbors:
        G.add_edge(idx, neighbor)

# Calculate approximate Betweenness Centrality with k=100
print("Calculating Betweenness Centrality with k=100...")
betweenness_centrality = nx.betweenness_centrality(G, k=100, normalized=True)

# Combine Betweenness Centrality into a DataFrame
betweenness_df = pd.DataFrame({
    'COMID': gdf['COMID'],
    'X_Centroid': [G.nodes[idx]['x'] for idx in G.nodes],
    'Y_Centroid': [G.nodes[idx]['y'] for idx in G.nodes],
    'Betweenness Centrality': pd.Series(betweenness_centrality)
})

# Export the DataFrame to a CSV file
betweenness_df.to_csv('./MidUS_BetweennessCentrality_Approx.csv', index=False)

print("Betweenness Centrality has been calculated and exported to Mid-Atlantic_BetweennessCentrality_Approx.csv")

# Visualize the graph
pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes}
nx.draw(G, pos, node_size=0.001, node_color='b', edge_color='g', with_labels=False)
plt.title("Stream Network Visualization")
plt.show()


# In[13]:


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

# Define the number of classes
num_classes = 5

# Calculate the percentile thresholds for each class
percentiles = np.linspace(0, 100, num_classes + 1)
thresholds = np.percentile(bc_values, percentiles)

# Create a dictionary to map betweenness centrality values to classes
bc_classes = {}
for i in range(num_classes):
    lower_bound = thresholds[i]
    upper_bound = thresholds[i+1]
    bc_classes[i] = [node for node in G.nodes if lower_bound <= betweenness_centrality[node] <= upper_bound]

# Use a seaborn diverging palette
class_colors = sns.color_palette("Spectral", num_classes)  # 'RdYlBu' for a nice diverging look

# Draw the graph with nodes colored based on betweenness centrality classes
plt.figure(figsize=(10, 8))
for i, class_nodes in bc_classes.items():
    nx.draw_networkx_nodes(G, pos, nodelist=class_nodes, node_size=1.0, node_color=class_colors[i])

nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3)
plt.title('Network Graph with Node Color Based on Betweenness Centrality Classes')

# Creating a legend manually for better control
from matplotlib.patches import Patch
legend_labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']
legend_handles = [Patch(facecolor=color, label=label) for color, label in zip(class_colors, legend_labels)]
plt.legend(handles=legend_handles, title="Betweenness Centrality Classes")

plt.show()


# In[93]:


data.head()


# In[94]:


data.tail()


# In[31]:


import pandas as pd

# Load your data
data = pd.read_csv('./HUC(1-3)_NetworkMetrics.csv')

# Drop all rows after row 1317272
data = data.iloc[:625266]

# Save the updated data to a new CSV file
data.to_csv('./HUC(1-3)Network_Properties.csv', index=False)

print("Rows after 1317272 have been dropped. Updated data saved to 'trimmed_data.csv'.")


# In[22]:


import pandas as pd

# Load your data
data = pd.read_csv('./MidUS_BetweennessCentrality_Approx.csv')

# Drop all rows after row 1317272
data = data.iloc[:1317272]

# Save the updated data to a new CSV file
data.to_csv('./trimmedMidUSBetweenness_data.csv', index=False)

print("Rows after 1317272 have been dropped. Updated data saved to 'trimmed_data.csv'.")


# In[33]:


import pandas as pd

# Load the two CSV files
csv1 = pd.read_csv('./HUC(1-3)Network_Properties.csv')  # Replace with your first CSV file path
csv2 = pd.read_csv('./HUC(1-3)_Betweenness_Centrality.csv')  # Replace with your second CSV file path

# Merge the two DataFrames based on the 'COMID' column
merged_df = pd.merge(csv1, csv2, on='COMID', how='inner')  # 'inner' keeps only matching rows

# Save the merged DataFrame to a new CSV file
output_file = './HUC(1-3)merged_file.csv'
merged_df.to_csv(output_file, index=False)

print(f"Merged file saved to: {output_file}")


# In[58]:


import pandas as pd

# Specify the input and output file paths
input_file = './merged_file_UCLC.csv'
output_file = './merged_file_UCLC.csv'

# Read the CSV file
df = pd.read_csv(input_file)

# Replace missing values with zeros
df_filled = df.fillna(0)

# Save the updated DataFrame to a new CSV file
df_filled.to_csv(output_file, index=False)

print(f"Missing values have been replaced with zeros and saved to: {output_file}")


# In[39]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer

# Load your data
data = pd.read_csv('./trimmedMidUS_data.csv')

# Select features
features = data[['Degree Centrality', 'Betweenness Centrality', 'Closeness Centrality', 'Clustering Coefficient', 'Eigenvector Centrality']]

# Check for missing values
print("Missing values per feature:\n", features.isnull().sum())

# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)

# Standardizing the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Fit a Gaussian Mixture Model
gmm = GaussianMixture(n_components=5, random_state=0)
gmm.fit(features_scaled)
data['Cluster'] = gmm.predict(features_scaled)

# Get the centroids of the clusters
centroids = gmm.means_

# Save the results
data.to_csv('./clustered_data.csv', index=False)
print("Clustered data saved to 'clustered_data.csv'")



# In[41]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

bic_scores = []
aic_scores = []
n_components_range = range(1, 11)

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=0)
    gmm.fit(features_scaled)
    bic_scores.append(gmm.bic(features_scaled))
    aic_scores.append(gmm.aic(features_scaled))

# Plot BIC and AIC
plt.figure(figsize=(8, 5))
plt.plot(n_components_range, bic_scores, label='BIC', marker='o')
plt.plot(n_components_range, aic_scores, label='AIC', marker='s')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('GMM Model Selection: BIC vs AIC')
plt.legend()
#plt.grid(True)
plt.show()


# In[47]:


# Plot BIC and AIC
plt.figure(figsize=(8, 6))
plt.plot(n_components_range, bic_scores, label='BIC', marker='o')
#plt.plot(n_components_range, aic_scores, label='AIC', marker='s')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('GMM Model Selection: BIC vs AIC')
plt.legend()
plt.grid(False)
plt.savefig("Bayesian_Information_Criterion.png", dpi = 1000)
plt.show()


# In[46]:


# Plot BIC and AIC
plt.figure(figsize=(8, 6))
#plt.plot(n_components_range, bic_scores, label='BIC', marker='o')
plt.plot(n_components_range, aic_scores, label='AIC', marker='s')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('GMM Model Selection: BIC vs AIC')
plt.legend()
plt.grid(False)
plt.show()


# In[61]:


import matplotlib.pyplot as plt
import numpy as np

# Create a DataFrame for the centroids for easier manipulation and visualization
centroid_df = pd.DataFrame(centroids, columns=features.columns)
centroid_df.plot(kind='bar')
plt.title('Cluster Centroids for Stream Network Properties')
plt.xticks(ticks=np.arange(centroid_df.shape[0]), labels=[f'Cluster {i+1}' for i in range(centroid_df.shape[0])], rotation=0)
plt.xlabel('Cluster')
plt.ylabel('Scaled Feature Value')
plt.legend(title="Features")
plt.show()

# Ensuring all values are positive
min_value = np.min(centroid_df.values)
shifted_values = centroid_df - min_value + 0.1  # Adding 0.1 to avoid log(0)

# Taking the logarithm of the shifted values
log_centroids = np.log(shifted_values)

log_centroids.plot(kind='bar')
plt.title('Log of Adjusted Cluster Centroids for Stream Network Properties')
plt.xticks(ticks=np.arange(log_centroids.shape[0]), labels=[f'Cluster {i+1}' for i in range(log_centroids.shape[0])], rotation=0)
plt.xlabel('Cluster')
plt.ylabel('Log of Adjusted Scaled Feature Value')
plt.legend(title="Features")
plt.show()


# In[62]:


# Define the updated cluster-to-stream type mapping based on the figure
cluster_to_stream_type = {
    0: 'Peripheral Streams',    # Cluster 1
    1: 'Central Streams',        # Cluster 2
    2: 'Mixed Roles',           # Cluster 3
    3: 'Bridge Streams',       # Cluster 4
    4: 'Cluster Streams'        # Cluster 5
}

# Assign stream types to clusters in the DataFrame
data['Stream Type'] = data['Cluster'].map(cluster_to_stream_type)

# Save the updated file with stream types
output_file = './StreamClasses_UCLC_recent.csv'
data.to_csv(output_file, index=False)

print(f"Updated file with refined stream types saved to: {output_file}")


# In[50]:


cluster_to_stream_type = {
    0: 'Peripheral Streams',           # Cluster 1
    1: 'Cluster Streams',       # Cluster 2
    2: 'Central Streams',    # Cluster 3
    3: 'Mixed Roles',       # Cluster 4
    4: 'Bridge Streams'         # Cluster 5
}

# Assign stream types to clusters in the DataFrame
data['Stream Type'] = data['Cluster'].map(cluster_to_stream_type)

# Save the updated file with stream types
output_file = './Stream_Classes_MidUS_revised.csv'
data.to_csv(output_file, index=False)

print(f"Updated file with refined stream types saved to: {output_file}")


# In[109]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_MidUS_revised.csv')

# Load all shapefiles into separate GeoDataFrames
gdf6 = gpd.read_file('./Ohio/Hydrography/NHDFlowline.shp')
gdf7 = gpd.read_file('./Tennessee/Hydrography/NHDFlowline.shp')
gdf8 = gpd.read_file('./Upper Mississipi/Hydrography/NHDFlowline.shp')
gdf9 = gpd.read_file('./Lower Mississipi/Hydrography/NHDFlowline.shp')
gdf10 = gpd.read_file('./Upper Missouri/Hydrography/NHDFlowline.shp')
gdf11 = gpd.read_file('./Lower Missouri/Hydrography/NHDFlowline.shp')
gdf12 = gpd.read_file('./Ark Red White/Hydrography/NHDFlowline.shp')

# Combine all GeoDataFrames into a single GeoDataFrame
nhd_flowline = pd.concat([gdf6, gdf7, gdf8, gdf9, gdf10, gdf11, gdf12], ignore_index=True)

# Ensure 'COMID' is the key for merging (update column names if different)
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Merge the combined shapefile with the stream data based on 'COMID'
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Plot the streams colored by Stream Type
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Define a color palette for Stream Types
colors = {
    'Central Streams': 'blue',
    'Cluster Streams': 'green',
    'Mixed Roles': 'purple',
    'Peripheral Streams': 'orange',
    'Bridge Streams': 'red'
}

# Plot each Stream Type
for stream_type, color in colors.items():
    merged_gdf[merged_gdf['Stream Type'] == stream_type].plot(ax=ax, color=color, linewidth=1, label=stream_type)

# Add plot details
plt.title('Stream Visualization by Stream Types')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
# plt.legend(title="Stream Types")
plt.grid(False)
plt.tight_layout()
plt.show()


# In[176]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_MidUS_revised.csv')

# Load all shapefiles into separate GeoDataFrames
gdf6 = gpd.read_file('./Ohio/Hydrography/NHDFlowline.shp')
gdf7 = gpd.read_file('./Tennessee/Hydrography/NHDFlowline.shp')
gdf8 = gpd.read_file('./Upper Mississipi/Hydrography/NHDFlowline.shp')
gdf9 = gpd.read_file('./Lower Mississipi/Hydrography/NHDFlowline.shp')
gdf10 = gpd.read_file('./Upper Missouri/Hydrography/NHDFlowline.shp')
gdf11 = gpd.read_file('./Lower Missouri/Hydrography/NHDFlowline.shp')
gdf12 = gpd.read_file('./Ark Red White/Hydrography/NHDFlowline.shp')

# Combine all GeoDataFrames into a single GeoDataFrame
nhd_flowline = pd.concat([gdf6, gdf7, gdf8, gdf9, gdf10, gdf11, gdf12], ignore_index=True)

# Ensure 'COMID' is the key for merging (update column names if different)
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Rename stream types
stream_data['Stream Type'] = stream_data['Stream Type'].replace({
    'Bridge Streams': 'Connector Streams',
    'Mixed Roles': 'Convergent Streams'
})

# Merge the combined shapefile with the stream data based on 'COMID'
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Plot the streams colored by Stream Type
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Define a color palette for Stream Types
colors = {
    'Central Streams': 'blue',
    #'Cluster Streams': 'green',
    'Convergent Streams': 'purple',
    #'Peripheral Streams': 'orange',
    'Connector Streams': 'red'
}

# Plot each Stream Type
for stream_type, color in colors.items():
    merged_gdf[merged_gdf['Stream Type'] == stream_type].plot(ax=ax, color=color, linewidth=1, label=stream_type)

# Add plot details
plt.title('Stream Visualization by Stream Types')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title="Stream Types", loc='lower right')  # Legend placed in the lower right
plt.grid(False)
plt.tight_layout()
plt.show()



# In[4]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_MidUS_revised.csv')

# Load all shapefiles for the Mississippi River Basin into separate GeoDataFrames
gdf6 = gpd.read_file('./Ohio/Hydrography/NHDFlowline.shp')
gdf7 = gpd.read_file('./Tennessee/Hydrography/NHDFlowline.shp')
gdf8 = gpd.read_file('./Upper Mississipi/Hydrography/NHDFlowline.shp')
gdf9 = gpd.read_file('./Lower Mississipi/Hydrography/NHDFlowline.shp')
gdf10 = gpd.read_file('./Upper Missouri/Hydrography/NHDFlowline.shp')
gdf11 = gpd.read_file('./Lower Missouri/Hydrography/NHDFlowline.shp')
gdf12 = gpd.read_file('./Ark Red White/Hydrography/NHDFlowline.shp')

# Combine all GeoDataFrames into a single GeoDataFrame for the Mississippi River Basin
nhd_flowline = pd.concat([gdf6, gdf7, gdf8, gdf9, gdf10, gdf11, gdf12], ignore_index=True)

# Ensure 'COMID' is the key for merging (update column names if different)
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Rename stream types
stream_data['Stream Type'] = stream_data['Stream Type'].replace({
    'Bridge Streams': 'Connector Streams',
    'Mixed Roles': 'Convergent Streams'
})

# Merge the combined shapefile with the stream data based on 'COMID'
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Load the pharmaceutical manufacturing dataset
pharma_path = "./Pharmaceutical Techno-Economic Data (APAP)(Validation Base Dataset).csv"
pharma_df = pd.read_csv(pharma_path, encoding="ISO-8859-1")

# Extract relevant columns
latitude_col = "SITE LATITUDE"
longitude_col = "SITE LONGITUDE"

# Drop missing values
pharma_df = pharma_df.dropna(subset=[latitude_col, longitude_col])

# Convert to numeric
pharma_df[latitude_col] = pd.to_numeric(pharma_df[latitude_col], errors="coerce")
pharma_df[longitude_col] = pd.to_numeric(pharma_df[longitude_col], errors="coerce")

# **Filter manufacturing sites within the Mississippi River Basin**
mississippi_lat_min, mississippi_lat_max = 25, 50  # Approximate latitude range
mississippi_lon_min, mississippi_lon_max = -105, -80  # Approximate longitude range

pharma_df = pharma_df[
    (pharma_df[latitude_col] >= mississippi_lat_min) & (pharma_df[latitude_col] <= mississippi_lat_max) &
    (pharma_df[longitude_col] >= mississippi_lon_min) & (pharma_df[longitude_col] <= mississippi_lon_max)
]

# Convert manufacturing sites to GeoDataFrame
gdf_manufacturing = gpd.GeoDataFrame(
    pharma_df, geometry=gpd.points_from_xy(pharma_df[longitude_col], pharma_df[latitude_col]), crs="EPSG:4326"
)

# **Plot the Mississippi River Basin with Manufacturing Sites**
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Define a color palette for Stream Types
colors = {
    'Central Streams': 'blue',
    #'Cluster Streams': 'green',
    #'Convergent Streams': 'purple',
    #'Peripheral Streams': 'orange',
    'Connector Streams': 'red'
}

# Plot each Stream Type
for stream_type, color in colors.items():
    merged_gdf[merged_gdf['Stream Type'] == stream_type].plot(ax=ax, color=color, linewidth=1, label=stream_type)

# **Overlay manufacturing sites (black dots)**
gdf_manufacturing.plot(ax=ax, color="black", markersize=40, alpha=0.9, edgecolor="white", label="Manufacturing Sites")

# **Add plot details**
plt.title('Mississippi River Basin: Stream Classification & Manufacturing Sites')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(fontsize=12, loc='upper right')
plt.grid(False)
plt.tight_layout()
plt.savefig("Mississippi_Streams_Manufacturing_Sites.png", dpi=700)
plt.show()


# In[164]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_MidUS_revised.csv')

# Load all shapefiles into separate GeoDataFrames
gdf6 = gpd.read_file('./Ohio/Hydrography/NHDFlowline.shp')
#gdf7 = gpd.read_file('./Tennessee/Hydrography/NHDFlowline.shp')


# Combine all GeoDataFrames into a single GeoDataFrame
nhd_flowline = pd.concat([gdf6], ignore_index=True)

# Ensure 'COMID' is the key for merging (update column names if different)
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Merge the combined shapefile with the stream data based on 'COMID'
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Plot the streams colored by Stream Type
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Define a color palette for Stream Types
colors = {
    'Central Streams': 'blue',
    'Cluster Streams': 'green',
    'Mixed Roles': 'purple',
    'Peripheral Streams': 'orange',
    'Bridge Streams': 'red'
}

# Plot each Stream Type
for stream_type, color in colors.items():
    merged_gdf[merged_gdf['Stream Type'] == stream_type].plot(ax=ax, color=color, linewidth=2, label=stream_type)

# Add plot details
plt.title('Stream Visualization by Stream Types')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
# plt.legend(title="Stream Types")
plt.grid(False)
plt.tight_layout()
plt.savefig("Ohio_StreamClasses.png", dpi =700)
plt.show()


# In[155]:


import matplotlib.pyplot as plt
import numpy as np

# Define categories (network properties)
categories = network_properties
num_vars = len(categories)

# Normalize the data using min-max scaling (0-1) for better visualization
heatmap_data_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

# Define angles for radar chart
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# Complete the loop for radar chart
angles += angles[:1]

# Create radar chart for each cluster
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Define labels for high, medium, and low
high, medium, low = 1, 0.5, 0

# Plot each cluster's normalized values
for cluster in heatmap_data_norm.index:
    values = heatmap_data_norm.loc[cluster].tolist()
    values += values[:1]  # Ensure loop closes
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=cluster)
    ax.fill(angles, values, alpha=0.25)

# Add labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)

# Add radial labels for High, Medium, Low
ax.set_yticklabels(["Low", "Medium", "High"], fontsize=9)

# Title and legend
#plt.title("Radar Plot of Network Properties Across Clusters (Ohio)", fontsize=12)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.show()


# In[28]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_MidUS_revised.csv')

# Load all shapefiles into separate GeoDataFrames
gdf9 = gpd.read_file('./Lower Mississipi/Hydrography/NHDFlowline.shp')

# Combine all GeoDataFrames into a single GeoDataFrame
nhd_flowline = pd.concat([gdf9], ignore_index=True)

# Ensure 'COMID' is the key for merging (update column names if different)
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Merge the combined shapefile with the stream data based on 'COMID'
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Plot the streams colored by Stream Type
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Define a color palette for Stream Types
colors = {
    'Central Streams': 'blue',
    'Cluster Streams': 'green',
    'Mixed Roles': 'purple',
    'Peripheral Streams': 'orange',
    'Bridge Streams': 'red'
}

# Plot each Stream Type
for stream_type, color in colors.items():
    merged_gdf[merged_gdf['Stream Type'] == stream_type].plot(ax=ax, color=color, linewidth=1, label=stream_type)

# Add plot details
plt.title('Stream Visualization by Stream Types')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
# plt.legend(title="Stream Types")
plt.grid(False)
plt.tight_layout()
#plt.savefig("Connector_Streams.png",dpi = 1000)
plt.show()


# In[30]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_MidUS_revised.csv')

# Load all shapefiles into separate GeoDataFrames
gdf9 = gpd.read_file('./Lower Mississipi/Hydrography/NHDFlowline.shp')

# Combine all GeoDataFrames into a single GeoDataFrame
nhd_flowline = pd.concat([gdf9], ignore_index=True)

# Ensure 'COMID' is the key for merging (update column names if different)
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Merge the combined shapefile with the stream data based on 'COMID'
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Plot the streams colored by Stream Type
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Define a color palette for Stream Types
colors = {
    'Central Streams': 'blue',
   'Cluster Streams': 'green',
   'Mixed Roles': 'purple',
   'Peripheral Streams': 'orange',
    'Bridge Streams': 'red'
}

# Plot each Stream Type
for stream_type, color in colors.items():
    merged_gdf[merged_gdf['Stream Type'] == stream_type].plot(ax=ax, color=color, linewidth=1.5, label=stream_type)

# Add plot details
plt.title('Stream Visualization by Stream Types')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
# plt.legend(title="Stream Types")
plt.grid(False)
plt.tight_layout()
plt.savefig("Connector_Streams_LM.png",dpi = 1000)
plt.show()


# In[5]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_MidUS_revised.csv')

# Load all shapefiles into separate GeoDataFrames
gdf9 = gpd.read_file('./Lower Mississipi/Hydrography/NHDFlowline.shp')

# Combine all GeoDataFrames into a single GeoDataFrame
nhd_flowline = pd.concat([gdf9], ignore_index=True)

# Ensure 'COMID' is the key for merging (update column names if different)
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Merge the combined shapefile with the stream data based on 'COMID'
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Plot the streams colored by Stream Type
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Define a color palette for Stream Types
colors = {
    'Central Streams': 'blue',
   #'Cluster Streams': 'green',
   #'Mixed Roles': 'purple',
   #'Peripheral Streams': 'orange',
    #'Bridge Streams': 'red'
}

# Plot each Stream Type
for stream_type, color in colors.items():
    merged_gdf[merged_gdf['Stream Type'] == stream_type].plot(ax=ax, color=color, linewidth=1.5, label=stream_type)

# Add plot details
plt.title('Stream Visualization by Stream Types')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
# plt.legend(title="Stream Types")
plt.grid(False)
plt.tight_layout()
plt.savefig("Connector_Streams_Lower_Mississippi1.png",dpi = 1000)
plt.show()


# In[2]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_MidUS_revised.csv')

# Load all shapefiles into separate GeoDataFrames
gdf9 = gpd.read_file('./Lower Mississipi/Hydrography/NHDFlowline.shp')

# Combine all GeoDataFrames into a single GeoDataFrame
nhd_flowline = pd.concat([gdf9], ignore_index=True)

# Ensure 'COMID' is the key for merging (update column names if different)
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Merge the combined shapefile with the stream data based on 'COMID'
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Plot the streams colored by Stream Type
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Define a color palette for Stream Types
colors = {
    #'Central Streams': 'blue',
   'Cluster Streams': 'green',
   #'Mixed Roles': 'purple',
   #'Peripheral Streams': 'orange',
    #'Bridge Streams': 'red'
}

# Plot each Stream Type
for stream_type, color in colors.items():
    merged_gdf[merged_gdf['Stream Type'] == stream_type].plot(ax=ax, color=color, linewidth=1.5, label=stream_type)

# Add plot details
plt.title('Stream Visualization by Stream Types')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
# plt.legend(title="Stream Types")
plt.grid(False)
plt.tight_layout()
plt.savefig("Connector_Streams_Lower_Mississippi1.png",dpi = 1000)
plt.show()


# In[3]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_MidUS_revised.csv')

# Load all shapefiles into separate GeoDataFrames
gdf9 = gpd.read_file('./Lower Mississipi/Hydrography/NHDFlowline.shp')

# Combine all GeoDataFrames into a single GeoDataFrame
nhd_flowline = pd.concat([gdf9], ignore_index=True)

# Ensure 'COMID' is the key for merging (update column names if different)
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Merge the combined shapefile with the stream data based on 'COMID'
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Plot the streams colored by Stream Type
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Define a color palette for Stream Types
colors = {
    #'Central Streams': 'blue',
   #'Cluster Streams': 'green',
   #'Mixed Roles': 'purple',
   'Peripheral Streams': 'orange',
    #'Bridge Streams': 'red'
}

# Plot each Stream Type
for stream_type, color in colors.items():
    merged_gdf[merged_gdf['Stream Type'] == stream_type].plot(ax=ax, color=color, linewidth=1.5, label=stream_type)

# Add plot details
plt.title('Stream Visualization by Stream Types')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
# plt.legend(title="Stream Types")
plt.grid(False)
plt.tight_layout()
plt.savefig("Connector_Streams_Lower_Mississippi1.png",dpi = 1000)
plt.show()


# In[4]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_MidUS_revised.csv')

# Load all shapefiles into separate GeoDataFrames
gdf9 = gpd.read_file('./Lower Mississipi/Hydrography/NHDFlowline.shp')

# Combine all GeoDataFrames into a single GeoDataFrame
nhd_flowline = pd.concat([gdf9], ignore_index=True)

# Ensure 'COMID' is the key for merging (update column names if different)
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Merge the combined shapefile with the stream data based on 'COMID'
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Plot the streams colored by Stream Type
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Define a color palette for Stream Types
colors = {
    #'Central Streams': 'blue',
   #'Cluster Streams': 'green',
   'Mixed Roles': 'purple',
  # 'Peripheral Streams': 'orange',
    #'Bridge Streams': 'red'
}

# Plot each Stream Type
for stream_type, color in colors.items():
    merged_gdf[merged_gdf['Stream Type'] == stream_type].plot(ax=ax, color=color, linewidth=1.5, label=stream_type)

# Add plot details
plt.title('Stream Visualization by Stream Types')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
# plt.legend(title="Stream Types")
plt.grid(False)
plt.tight_layout()
plt.savefig("Connector_Streams_Lower_Mississippi1.png",dpi = 1000)
plt.show()


# In[ ]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_MidUS_revised.csv')

# Load all shapefiles into separate GeoDataFrames
gdf9 = gpd.read_file('./Lower Mississipi/Hydrography/NHDFlowline.shp')

# Combine all GeoDataFrames into a single GeoDataFrame
nhd_flowline = pd.concat([gdf9], ignore_index=True)

# Ensure 'COMID' is the key for merging (update column names if different)
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Merge the combined shapefile with the stream data based on 'COMID'
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Plot the streams colored by Stream Type
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Define a color palette for Stream Types
colors = {
    #'Central Streams': 'blue',
   #'Cluster Streams': 'green',
  # 'Mixed Roles': 'purple',
  # 'Peripheral Streams': 'orange',
    #'Bridge Streams': 'red'
}

# Plot each Stream Type
for stream_type, color in colors.items():
    merged_gdf[merged_gdf['Stream Type'] == stream_type].plot(ax=ax, color=color, linewidth=1.5, label=stream_type)

# Add plot details
plt.title('Stream Visualization by Stream Types')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
# plt.legend(title="Stream Types")
plt.grid(False)
plt.tight_layout()
plt.savefig("Connector_Streams_Lower_Mississippi1.png",dpi = 1000)
plt.show()


# In[13]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_MidUS_revised.csv')

# Load all shapefiles into separate GeoDataFrames
gdf9 = gpd.read_file('./Lower Mississipi/Hydrography/NHDFlowline.shp')

# Combine all GeoDataFrames into a single GeoDataFrame
nhd_flowline = pd.concat([gdf9], ignore_index=True)

# Ensure 'COMID' is the key for merging (update column names if different)
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Merge the combined shapefile with the stream data based on 'COMID'
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Plot the streams colored by Stream Type
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Define a color palette for Stream Types
colors = {
   # 'Central Streams': 'blue',
    'Cluster Streams': 'green',
   # 'Mixed Roles': 'purple',
    'Peripheral Streams': 'orange',
    #'Bridge Streams': 'red'
}

# Plot each Stream Type
for stream_type, color in colors.items():
    merged_gdf[merged_gdf['Stream Type'] == stream_type].plot(ax=ax, color=color, linewidth=2, label=stream_type)

# Add plot details
plt.title('Stream Visualization by Stream Types')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
# plt.legend(title="Stream Types")
plt.grid(False)
plt.tight_layout()
#plt.savefig("Connector_Streams.png",dpi = 1000)
plt.show()


# In[33]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_MidUS_revised.csv')

# Load shapefile
gdf9 = gpd.read_file('./Lower Mississipi/Hydrography/NHDFlowline.shp')

# Combine
nhd_flowline = gdf9.copy()

# Ensure 'COMID' is the key for merging
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Merge the shapefile with the stream data
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Define a color palette for Stream Types
colors = {
    'Central Streams': 'blue',
    'Cluster Streams': 'green',
    'Mixed Roles': 'purple',
    'Peripheral Streams': 'orange',
    'Bridge Streams': 'red'
}

# Plot setup
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Plot all stream types with faded colors (alpha=0.2)
for stream_type, color in colors.items():
    if stream_type not in ['Central Streams', 'Bridge Streams']:
        subset = merged_gdf[merged_gdf['Stream Type'] == stream_type]
        subset.plot(ax=ax, color=color, linewidth=1, alpha=0.3)

# Plot the two focus stream types with full color
for stream_type in ['Central Streams', 'Bridge Streams']:
    subset = merged_gdf[merged_gdf['Stream Type'] == stream_type]
    subset.plot(ax=ax, color=colors[stream_type], linewidth=2, label=stream_type, alpha=1)

# Final touches
plt.title('Stream Visualization by Stream Types (Emphasizing Bridge & Central)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc ="upper left")
plt.grid(False)
plt.tight_layout()
plt.savefig("Central_ConnectorStreams.png", dpi=600)
plt.show()


# In[38]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_MidUS_revised.csv')

# Rename 'Bridge Streams' to 'Connector Streams'
stream_data['Stream Type'] = stream_data['Stream Type'].replace({'Bridge Streams': 'Connector Streams'})

# Load shapefile
gdf9 = gpd.read_file('./Lower Mississipi/Hydrography/NHDFlowline.shp')

# Copy and prepare for merge
nhd_flowline = gdf9.copy()
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Merge shapefile with stream data
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Define updated color palette
colors = {
    'Central Streams': 'blue',
    'Cluster Streams': 'green',
    'Mixed Roles': 'purple',
    'Peripheral Streams': 'orange',
    'Connector Streams': 'red'  # updated label
}

# Plot setup
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Plot all stream types with faded colors (alpha=0.3)
for stream_type, color in colors.items():
    if stream_type not in ['Central Streams', 'Connector Streams']:
        subset = merged_gdf[merged_gdf['Stream Type'] == stream_type]
        subset.plot(ax=ax, color=color, linewidth=1, alpha=0.3)

# Plot the two focus stream types with full color
for stream_type in ['Central Streams', 'Connector Streams']:
    subset = merged_gdf[merged_gdf['Stream Type'] == stream_type]
    subset.plot(ax=ax, color=colors[stream_type], linewidth=2, label=stream_type, alpha=1)

# Final touches
plt.title('Stream Visualization by Stream Types (Emphasizing Central & Connector)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc="upper left")
plt.grid(False)
plt.tight_layout()
plt.savefig("Central_ConnectorStreams.png", dpi=600)
plt.show()


# In[37]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_MidUS_revised.csv')

# Load shapefile
gdf9 = gpd.read_file('./Lower Mississipi/Hydrography/NHDFlowline.shp')

# Combine
nhd_flowline = gdf9.copy()

# Ensure 'COMID' is the key for merging
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Merge the shapefile with the stream data
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Define a color palette for Stream Types
colors = {
    'Central Streams': 'blue',
    'Cluster Streams': 'green',
    'Mixed Roles': 'purple',
    'Peripheral Streams': 'orange',
    'Bridge Streams': 'red'
}

# Plot setup
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Plot all stream types with faded colors (alpha=0.2)
for stream_type, color in colors.items():
    if stream_type not in ['Peripheral Streams']:
        subset = merged_gdf[merged_gdf['Stream Type'] == stream_type]
        subset.plot(ax=ax, color=color, linewidth=1, alpha=0.3)

# Plot the two focus stream types with full color
for stream_type in ['Peripheral Streams']:
    subset = merged_gdf[merged_gdf['Stream Type'] == stream_type]
    subset.plot(ax=ax, color=colors[stream_type], linewidth=2, label=stream_type, alpha=1)

# Final touches
plt.title('Stream Visualization by Stream Types (Emphasizing Bridge & Central)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc ='upper left')
plt.grid(False)
plt.tight_layout()
plt.savefig("Peripheral_Stream_LM.png", dpi=1000)
plt.show()


# In[34]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_MidUS_revised.csv')

# Load shapefile
gdf9 = gpd.read_file('./Lower Mississipi/Hydrography/NHDFlowline.shp')

# Combine
nhd_flowline = gdf9.copy()

# Ensure 'COMID' is the key for merging
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Merge the shapefile with the stream data
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Define a color palette for Stream Types
colors = {
    'Central Streams': 'blue',
    'Cluster Streams': 'green',
    'Mixed Roles': 'purple',
    'Peripheral Streams': 'orange',
    'Bridge Streams': 'red'
}

# Plot setup
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Plot all stream types with faded colors (alpha=0.2)
for stream_type, color in colors.items():
    if stream_type not in ['Cluster Streams']:
        subset = merged_gdf[merged_gdf['Stream Type'] == stream_type]
        subset.plot(ax=ax, color=color, linewidth=1, alpha=0.3)

# Plot the two focus stream types with full color
for stream_type in ['Cluster Streams']:
    subset = merged_gdf[merged_gdf['Stream Type'] == stream_type]
    subset.plot(ax=ax, color=colors[stream_type], linewidth=2, label=stream_type, alpha=1)

# Final touches
plt.title('Stream Visualization by Stream Types (Emphasizing Bridge & Central)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc= "upper left")
plt.grid(False)
plt.tight_layout()
plt.savefig("Cluster_Streams.png", dpi=600)
plt.show()


# In[40]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_MidUS_revised.csv')

# Load shapefile
gdf9 = gpd.read_file('./Lower Mississipi/Hydrography/NHDFlowline.shp')

# Combine
nhd_flowline = gdf9.copy()

# Ensure 'COMID' is the key for merging
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Merge the shapefile with the stream data
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Define a color palette for Stream Types
colors = {
    'Central Streams': 'blue',
    'Cluster Streams': 'green',
    'Mixed Roles': 'purple',
    'Peripheral Streams': 'orange',
    'Bridge Streams': 'red'
}

# Plot setup
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Plot all stream types with faded colors (alpha=0.2)
for stream_type, color in colors.items():
    if stream_type not in ['Mixed Roles']:
        subset = merged_gdf[merged_gdf['Stream Type'] == stream_type]
        subset.plot(ax=ax, color=color, linewidth=1, alpha=0.3)

# Plot the two focus stream types with full color
for stream_type in ['Mixed Roles']:
    subset = merged_gdf[merged_gdf['Stream Type'] == stream_type]
    subset.plot(ax=ax, color=colors[stream_type], linewidth=2, label=stream_type, alpha=1)

# Final touches
plt.title('Stream Visualization by Stream Types (Emphasizing Bridge & Central)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title="Highlighted Stream Types")
plt.grid(False)
plt.tight_layout()
plt.savefig("C.png", dpi=600)
plt.show()


# In[42]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_MidUS_revised.csv')

# Rename 'Mixed Roles' to 'Convergent Streams'
stream_data['Stream Type'] = stream_data['Stream Type'].replace({'Mixed Roles': 'Convergent Streams'})

# Load shapefile
gdf9 = gpd.read_file('./Lower Mississipi/Hydrography/NHDFlowline.shp')

# Combine
nhd_flowline = gdf9.copy()

# Ensure 'COMID' is the key for merging
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Merge the shapefile with the stream data
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Define a color palette for Stream Types
colors = {
    'Central Streams': 'blue',
    'Cluster Streams': 'green',
    'Convergent Streams': 'purple',  # updated label
    'Peripheral Streams': 'orange',
    'Bridge Streams': 'red'
}

# Plot setup
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Plot all stream types with faded colors (alpha=0.3)
for stream_type, color in colors.items():
    if stream_type != 'Convergent Streams':
        subset = merged_gdf[merged_gdf['Stream Type'] == stream_type]
        subset.plot(ax=ax, color=color, linewidth=1, alpha=0.3)

# Plot the focus stream type with full color
subset = merged_gdf[merged_gdf['Stream Type'] == 'Convergent Streams']
subset.plot(ax=ax, color=colors['Convergent Streams'], linewidth=2, label='Convergent Streams', alpha=1)

# Final touches
plt.title('Stream Visualization by Stream Types (Emphasizing Convergent Streams)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc = 'upper left')
plt.grid(False)
plt.tight_layout()
plt.savefig("Convergent_Streams.png", dpi=600)
plt.show()


# In[114]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_MidUS_revised.csv')

# Load all shapefiles into separate GeoDataFrames
gdf6 = gpd.read_file('./Ohio/Hydrography/NHDFlowline.shp')

# Combine all GeoDataFrames into a single GeoDataFrame
nhd_flowline = pd.concat([gdf6], ignore_index=True)

# Ensure 'COMID' is the key for merging (update column names if different)
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Merge the combined shapefile with the stream data based on 'COMID'
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Plot the streams colored by Stream Type
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Define a color palette for Stream Types
colors = {
    'Mixed Roles': 'purple',
    #'Peripheral Streams': 'orange',
}

# Plot each Stream Type with increased linewidth
for stream_type, color in colors.items():
    merged_gdf[merged_gdf['Stream Type'] == stream_type].plot(ax=ax, color=color, linewidth=4, label=stream_type)  # Increased linewidth

# Add plot details
plt.title('Stream Visualization by Stream Types', fontsize=16)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.grid(False)
plt.tight_layout()
plt.savefig("Covergent_Streams.png", dpi= 1000)
plt.show()


# In[115]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_MidUS_revised.csv')

# Load all shapefiles into separate GeoDataFrames
gdf6 = gpd.read_file('./Ohio/Hydrography/NHDFlowline.shp')

# Combine all GeoDataFrames into a single GeoDataFrame
nhd_flowline = pd.concat([gdf6], ignore_index=True)

# Ensure 'COMID' is the key for merging (update column names if different)
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Merge the combined shapefile with the stream data based on 'COMID'
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Plot the streams colored by Stream Type
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Define a color palette for Stream Types
colors = {
    #'Mixed Roles': 'purple',
    #'Peripheral Streams': 'orange',
    'Central Streams': 'blue',
}

# Plot each Stream Type with increased linewidth
for stream_type, color in colors.items():
    merged_gdf[merged_gdf['Stream Type'] == stream_type].plot(ax=ax, color=color, linewidth=5, label=stream_type)  # Increased linewidth

# Add plot details
plt.title('Stream Visualization by Stream Types', fontsize=16)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.grid(False)
plt.tight_layout()
plt.savefig("Central_Streams.png",dpi = 1000)
plt.show()


# In[102]:


# Plot the streams colored by Stream Type
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Define a color palette
colors = {
    'Central Streams': 'blue',
    'Cluster Streams': 'green',
    'Mixed Roles': 'purple',
    'Peripheral Streams': 'orange',
    'Bridge Streams': 'red',
    'Imbounded': 'black'
}

# Plot each Stream Type
for stream_type, color in colors.items():
    merged_gdf[merged_gdf['Stream Type'] == stream_type].plot(ax=ax, color=color, linewidth=1, label=stream_type)

# Add plot details
plt.title('Stream Visualization with Imbounded Streams')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title="Stream Types")
plt.grid(False)
plt.tight_layout()
plt.show()


# In[101]:


# Add a new column to mark imbounded streams using the correct column
merged_gdf['Imbounded'] = merged_gdf['COMID'].isin(imbounded_streams['COMID_left'])

# Update the Stream Type for imbounded streams
merged_gdf['Stream Type'] = merged_gdf.apply(
    lambda row: 'Imbounded' if row['Imbounded'] else row['Stream Type'], axis=1
)


# In[100]:


print(imbounded_streams.columns)


# In[98]:


print(merged_gdf.columns)


# In[40]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_HUC(1-3).csv')

# Load all shapefiles into separate GeoDataFrames
# Load all three shapefiles into separate GeoDataFrames
gdf1 = gpd.read_file('./NorthEast/Hydrography/NHDFlowline.shp')
gdf2 = gpd.read_file('./Mid-Atlantic/Hydrography/NHDFlowline.shp')
gdf3 = gpd.read_file('./South Atlantic North/Hydrography/NHDFlowline.shp')
gdf4 = gpd.read_file('./South Atlantic South/Hydrography/NHDFlowline.shp')
gdf5 = gpd.read_file('./South Atlantic West/Hydrography/NHDFlowline.shp')


# Combine all GeoDataFrames into a single GeoDataFrame
nhd_flowline = pd.concat([gdf1, gdf2, gdf3, gdf4, gdf5], ignore_index=True)

# Ensure 'COMID' is the key for merging (update column names if different)
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Merge the combined shapefile with the stream data based on 'COMID'
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Plot the streams colored by Stream Type
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Define a color palette for Stream Types
colors = {
    'Central Streams': 'blue',
    'Cluster Streams': 'green',
    'Mixed Roles': 'purple',
    'Peripheral Streams': 'orange',
    'Bridge Streams': 'red'
}

# Plot each Stream Type
for stream_type, color in colors.items():
    merged_gdf[merged_gdf['Stream Type'] == stream_type].plot(ax=ax, color=color, linewidth=1, label=stream_type)

# Add plot details
plt.title('Stream Visualization by Stream Types')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title="Stream Types")
plt.grid(False)
plt.tight_layout()
plt.show()


# In[4]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with cluster results and stream types
stream_data = pd.read_csv('./Stream_Classes_HUC(1-3).csv')

# Load all shapefiles into separate GeoDataFrames
# Load all three shapefiles into separate GeoDataFrames
gdf1 = gpd.read_file('./NorthEast/Hydrography/NHDFlowline.shp')
gdf2 = gpd.read_file('./Mid-Atlantic/Hydrography/NHDFlowline.shp')
gdf3 = gpd.read_file('./South Atlantic North/Hydrography/NHDFlowline.shp')
gdf4 = gpd.read_file('./South Atlantic South/Hydrography/NHDFlowline.shp')
gdf5 = gpd.read_file('./South Atlantic West/Hydrography/NHDFlowline.shp')


# Combine all GeoDataFrames into a single GeoDataFrame
nhd_flowline = pd.concat([gdf1, gdf2, gdf3, gdf4, gdf5], ignore_index=True)

# Ensure 'COMID' is the key for merging (update column names if different)
nhd_flowline = nhd_flowline.rename(columns={'COMID': 'COMID'})
stream_data = stream_data.rename(columns={'COMID': 'COMID'})

# Merge the combined shapefile with the stream data based on 'COMID'
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')

# Plot the streams colored by Stream Type
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Define a color palette for Stream Types
colors = {
    'Central Streams': 'blue',
    'Cluster Streams': 'green',
    'Mixed Roles': 'purple',
    'Peripheral Streams': 'orange',
    'Bridge Streams': 'red'
}

# Plot each Stream Type
for stream_type, color in colors.items():
    merged_gdf[merged_gdf['Stream Type'] == stream_type].plot(ax=ax, color=color, linewidth=1, label=stream_type)

# Add plot details
plt.title('Stream Visualization by Stream Types')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title="Stream Types")
plt.grid(False)
plt.tight_layout()
plt.show()


# In[8]:


# Merge the combined shapefile with the stream data based on 'COMID'
merged_gdf = nhd_flowline.merge(stream_data[['COMID', 'Stream Type']], on='COMID', how='left')



# In[9]:


merged_gdf


# In[3]:


stream_data


# In[ ]:




