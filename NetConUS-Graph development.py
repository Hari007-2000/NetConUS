#!/usr/bin/env python
# coding: utf-8

# This notebook mainly focuses on Modeling the Stream Networks,incorporating the fragmentation due to Dams and reservoirs.
# Importing the necessary libararies- Pandas, Geopandas and Networkx

# In[14]:


import pandas as pd
import geopandas as gpd
import networkx as nx


# In[2]:


# Loadoing NHD flowline data into a GeoDataFrame
gdf = gpd.read_file('./Shape/NHDFlowline.shp')


# In[3]:


gdf.head()


# The code below can be used to visualize the Impoundments due to NHD waterbody on the NHD Flowline.shp
# 
# All the below networkgraph is developed based on the NHD HUC-4 data that can be downloaded from this link: https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Hydrography/NHD/HU4/ 

# In[4]:


import geopandas as gpd
import matplotlib.pyplot as plt

# Load the NHD Flowline and Waterbody shapefiles
gdf_flowline = gpd.read_file('./Shape/NHDFlowline.shp')
gdf_waterbody = gpd.read_file('./Shape/NHDWaterbody.shp')

# Plot the Flowline and Waterbody
fig, ax = plt.subplots(figsize=(15, 10))

# Plot the NHD Flowline in one color (e.g., blue)
gdf_flowline.plot(ax=ax, color='blue', linewidth=0.5, label='Flowlines')

# Plot the NHD Waterbody in a different color (e.g., green)
gdf_waterbody.plot(ax=ax, color='green', alpha=1, label='Waterbody')

# Customize the plot
plt.title("NHDFlowline and NHDWaterbody Map")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()

plt.savefig("Reservoirs.png", dpi = 1000)

# Display the plot
plt.show()


# The code below can be used to spatially join the Impoundments due to the NHD Waterbody, and remove the nodes that are impounded due to the reservoir

# In[6]:


import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt

# Load the NHD flowline and waterbody (reservoir) data into GeoDataFrames
gdf_flowline = gpd.read_file('./Shape/NHDFlowline.shp')
gdf_reservoir = gpd.read_file('./Shape/NHDWaterbody.shp')

# Perform a spatial join between flowlines and reservoirs
gdf_impounded = gpd.sjoin(gdf_flowline, gdf_reservoir, how="inner", predicate="intersects")

# Check and print the columns of the joined GeoDataFrame
print("Impounded columns:", gdf_impounded.columns)

# Extract the COMIDs of the impounded flowlines using the correct column name
impounded_comids = gdf_impounded['permanent__left'].unique()

# Create an empty graph
G = nx.Graph()

# Precompute centroids to avoid recalculating them in the loop
centroids = gdf_flowline['geometry'].centroid

# Create a spatial index for the GeoDataFrame to speed up intersection checks
gdf_flowline_sindex = gdf_flowline.sindex

# Add nodes and edges to the graph
for idx, row in gdf_flowline.iterrows():
    point = centroids[idx]
    G.add_node(idx, x=point.x, y=point.y, comid=row['permanent_'])
    
    # Get potential neighbors using the spatial index
    possible_neighbors = list(gdf_flowline_sindex.intersection(row['geometry'].bounds))
    neighbors = [n for n in possible_neighbors if n != idx and gdf_flowline.loc[n, 'geometry'].intersects(row['geometry'])]

    # Add edges between the current node and its neighbors
    for neighbor in neighbors:
        G.add_edge(idx, neighbor)

# Remove impounded COMIDs from the graph
nodes_to_remove = [n for n, attr in G.nodes(data=True) if attr['comid'] in impounded_comids]
G.remove_nodes_from(nodes_to_remove)

# Optionally, plot the graph without the impounded nodes
pos = {idx: (row['geometry'].centroid.x, row['geometry'].centroid.y) for idx, row in gdf_flowline.iterrows() if idx in G.nodes}
plt.figure(figsize=(10, 10))
nx.draw(G, pos, node_size=5)
plt.title("Network Graph Without Impounded Flowlines")
plt.show()

# Display the impounded COMIDs
print("Impounded COMIDs:", impounded_comids)


# In[189]:


gdf.describe()


# In[190]:


gdf.flowdir.mean()


# In[ ]:


data.iloc


# The code below can be used to build the network graph from scartch based on the NHD Flowline data
# 
# The code spatially joins thew dams that cause fragmentation to the streams. The dams data can be downloaded from the National Inventory of Dams: https://nid.sec.usace.army.mil/#/

# In[9]:


import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Load NHD flowline data into a GeoDataFrame
gdf = gpd.read_file('./Shape/NHDFlowline.shp')

# Create an empty NetworkX graph
G = nx.Graph()

# Add nodes to the graph from NHD flowline data
for idx, row in gdf.iterrows():
    point = row['geometry'].centroid
    G.add_node(idx, x=point.x, y=point.y)

# Create a spatial index for the GeoDataFrame to speed up intersection checks
gdf_sindex = gdf.sindex

# Add edges to the graph
for idx, row in gdf.iterrows():
    possible_neighbors = list(gdf_sindex.intersection(row['geometry'].bounds))
    neighbors = [n for n in possible_neighbors if n != idx and gdf.loc[n, 'geometry'].intersects(row['geometry'])]
    for neighbor in neighbors:
        G.add_edge(idx, neighbor)

# Load dam information from CSV
dam_data = pd.read_csv('./nation.csv')

# Filter dam nodes based on their coordinates
filtered_dam_data = dam_data[(dam_data['Latitude'] >= gdf.bounds['miny'].min()) &
                             (dam_data['Latitude'] <= gdf.bounds['maxy'].max()) &
                             (dam_data['Longitude'] >= gdf.bounds['minx'].min()) &
                             (dam_data['Longitude'] <= gdf.bounds['maxx'].max())]

# Add filtered dam nodes to the graph
for index, row in filtered_dam_data.iterrows():
    latitude = row['Latitude']
    longitude = row['Longitude']
    dam_name = row['Dam Name']

    # Add dam as a node to the graph
    G.add_node(dam_name, x=longitude, y=latitude)

# Draw the graph with dam nodes highlighted
pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes}
nx.draw(G, pos, node_size=0.01, node_color='b', edge_color='g', with_labels=False)

# Highlight dam nodes with a different color
dam_nodes = [node for node in G.nodes if node in filtered_dam_data['Dam Name'].values]
nx.draw_networkx_nodes(G, pos, nodelist=dam_nodes, node_size=5, node_color='r')

plt.show()


# The code below removes that edges of the nodes that are fragmented due to the Dams and forms the modified graph

# In[54]:


import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import Point
from geopy.distance import geodesic
import matplotlib.pyplot as plt

# Load the NHD Flowline shapefile into a GeoDataFrame
gdf = gpd.read_file('./Shape/NHDFlowline.shp')

# Create an empty graph
G = nx.Graph()

# Add nodes to the graph using the centroids of the stream segments
for idx, row in gdf.iterrows():
    point = row['geometry'].centroid
    G.add_node(idx, x=point.x, y=point.y)

# Create a spatial index for the GeoDataFrame to speed up intersection checks
gdf_sindex = gdf.sindex

# Add edges to the graph based on intersecting stream segments
for idx, row in gdf.iterrows():
    possible_neighbors = list(gdf_sindex.intersection(row['geometry'].bounds))
    neighbors = [n for n in possible_neighbors if n != idx and gdf.loc[n, 'geometry'].intersects(row['geometry'])]
    for neighbor in neighbors:
        G.add_edge(idx, neighbor)

# Load the dams data from CSV
dam_data = pd.read_csv('nation.csv')

# Calculate the overall spatial extent of the stream network
min_latitude = gdf.bounds['miny'].min()
max_latitude = gdf.bounds['maxy'].max()
min_longitude = gdf.bounds['minx'].min()
max_longitude = gdf.bounds['maxx'].max()

# Filter the dam data to include only those dams within the spatial bounds of the stream network
filtered_dam_data = dam_data[
    (dam_data['Latitude'] >= min_latitude) &
    (dam_data['Latitude'] <= max_latitude) &
    (dam_data['Longitude'] >= min_longitude) &
    (dam_data['Longitude'] <= max_longitude)
]

# Function to find the nearest node in the graph to a dam location
def find_nearest_node(dam_point, graph_nodes):
    min_distance = float('inf')
    nearest_node = None
    for node, data in graph_nodes:
        node_point = (data['y'], data['x'])
        distance = geodesic(dam_point, node_point).meters
        if distance < min_distance:
            min_distance = distance
            nearest_node = node
    return nearest_node

# Iterate over the filtered dams and remove edges at the nearest node
for idx, dam in filtered_dam_data.iterrows():
    dam_point = (dam['Latitude'], dam['Longitude'])
    nearest_node = find_nearest_node(dam_point, G.nodes(data=True))
    
    # Remove all edges connected to the node nearest to the dam
    G.remove_edges_from(list(G.edges(nearest_node)))

# Draw the graph with custom settings
pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes}
nx.draw(G, pos, node_size=0.01, node_color='b', edge_color='g', with_labels=False)
plt.show()


# This code below simply visualizes the NHD Waterbody data

# In[25]:


import geopandas as gpd
import matplotlib.pyplot as plt

# Load the NHD Waterbody shapefile into a GeoDataFrame
gdf_reservoir = gpd.read_file('./Shape/NHDWaterbody.shp')

# Plot the GeoDataFrame using Matplotlib
plt.figure(figsize=(10, 10))
gdf_reservoir.plot(edgecolor='k', color='lightblue')

# Add a title and display the plot
plt.title('NHD HUC4 Waterbodies')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# In[12]:


import pandas as pd
import geopandas as gpd

# Step 1: Load the HUC4 network properties CSV file into a DataFrame
df_network = pd.read_csv('./HUC4_network_properties.csv')

# Step 2: Load the NHD flowline and waterbody (reservoir) data into GeoDataFrames
gdf_flowline = gpd.read_file('./Shape/NHDFlowline.shp')
gdf_reservoir = gpd.read_file('./Shape/NHDWaterbody.shp')

# Step 3: Perform a spatial join to identify impounded COMIDs
gdf_impounded = gpd.sjoin(gdf_flowline, gdf_reservoir, how="inner", predicate="intersects")
impounded_comids = gdf_impounded['permanent__left'].unique()

# Step 4: Update the 'Stream Type Final' column for impounded streams
df_network.loc[df_network['COMID'].isin(impounded_comids), 'Stream Type Final'] = 'Impounded'

# Step 5: Save the updated DataFrame back to the CSV file
df_network.to_csv('./HUC4 network properties.csv', index=False)

# Optionally, display the updated rows to verify
print(df_network[df_network['COMID'].isin(impounded_comids)].head())


# In[2]:


import umap

# Reducing dimensions with U-MAP
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(features)  # You can use scaled features if necessary

import matplotlib.pyplot as plt
import seaborn as sns

# Create a DataFrame for the U-MAP results
umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
umap_df['Cluster'] = data['Cluster']  # Add cluster labels for coloring

# Plotting
plt.figure(figsize=(10, 8))
sns.scatterplot(x='UMAP1', y='UMAP2', hue='Cluster', data=umap_df, palette=sns.color_palette("Spectral", len(umap_df['Cluster'].unique())))
plt.title('U-MAP Visualization of Stream Clusters')
plt.savefig("Clusters.png", dpi =1000)
plt.show()


# In[26]:


##Stream Classification based on Network Connectivity
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('./HUC4 network properties.csv') 

# Assuming the 'Cluster' column exists and contains the cluster labels for each stream
unique_types = data['Stream Type Final'].unique()

# Create a color map using a colormap from matplotlib
color_map = plt.cm.get_cmap('Spectral', len(unique_types))  # Adjust the colormap as needed
color_dict = {utype: color_map(i) for i, utype in enumerate(unique_types)}

# Plotting
plt.figure(figsize=(10, 8))

for stream_type in unique_types:
    subset = data[data['Stream Type Final'] == stream_type]
    plt.scatter(subset['X_Centroid_x'], subset['Y_Centroid_x'], s=6.0, color=color_dict[stream_type], label=f"{stream_type}")

plt.xlabel('X Centroid')
plt.ylabel('Y Centroid')
plt.title('Stream Types Visualization')
plt.legend(title='Stream Type')
plt.grid(False)  # Set to True if you want to enable the grid
plt.show()


# In[34]:


import matplotlib.pyplot as plt
import pandas as pd

# Load the data
data = pd.read_csv('./HUC4 network properties.csv') 

# Filter the data to include only "Peripheral Streams"
peripheral_streams = data[data['Stream Type Final'] == 'Peripheral Streams']

# Plotting
plt.figure(figsize=(10, 8))

# Plot the Peripheral Streams
plt.scatter(peripheral_streams['X_Centroid_x'], peripheral_streams['Y_Centroid_x'], s=6.0, color='blue', label="Peripheral Streams")

plt.xlabel('X Centroid')
plt.ylabel('Y Centroid')
plt.title('Peripheral Streams Visualization')
plt.legend(title='Stream Type')
plt.grid(False)  # Set to True if you want to enable the grid
plt.show()


# In[35]:


import matplotlib.pyplot as plt
import pandas as pd

# Load the data
data = pd.read_csv('./HUC4 network properties.csv') 

# Filter the data to include only "Bridge Streams"
bridge_streams = data[data['Stream Type Final'] == 'Bridge Streams']

# Plotting
plt.figure(figsize=(10, 8))

# Plot the Bridge Streams
plt.scatter(bridge_streams['X_Centroid_x'], bridge_streams['Y_Centroid_x'], s=6.0, color='red', label="Bridge Streams")

plt.xlabel('X Centroid')
plt.ylabel('Y Centroid')
plt.title('Bridge Streams Visualization')
plt.legend(title='Stream Type')
plt.grid(False)  # Set to True if you want to enable the grid
plt.show()


# In[36]:


import matplotlib.pyplot as plt
import pandas as pd

# Load the data
data = pd.read_csv('./HUC4 network properties.csv') 

# Filter the data to include only "High Centrality Streams"
central_streams = data[data['Stream Type Final'] == 'High Centrality Streams']

# Plotting
plt.figure(figsize=(10, 8))

# Plot the Central Streams
plt.scatter(central_streams['X_Centroid_x'], central_streams['Y_Centroid_x'], s=6.0, color='green', label="High Centrality Streams")

plt.xlabel('X Centroid')
plt.ylabel('Y Centroid')
plt.title('High Centrality Streams Visualization')
plt.legend(title='Stream Type')
plt.grid(False)  # Set to True if you want to enable the grid
plt.show()


# In[37]:


import matplotlib.pyplot as plt
import pandas as pd

# Load the data
data = pd.read_csv('./HUC4 network properties.csv') 

# Filter the data to include only "Impounded Streams"
impounded_streams = data[data['Stream Type Final'] == 'Impounded']

# Plotting
plt.figure(figsize=(10, 8))

# Plot the Impounded Streams
plt.scatter(impounded_streams['X_Centroid_x'], impounded_streams['Y_Centroid_x'], s=6.0, color='purple', label="Impounded Streams")

plt.xlabel('X Centroid')
plt.ylabel('Y Centroid')
plt.title('Impounded Streams Visualization')
plt.legend(title='Stream Type')
plt.grid(False)  # Set to True if you want to enable the grid
plt.show()


# In[38]:


import matplotlib.pyplot as plt
import pandas as pd

# Load the data
data = pd.read_csv('./HUC4 network properties.csv') 

# Filter the data to include only "Cluster Streams"
cluster_streams = data[data['Stream Type Final'] == 'Cluster Streams']

# Plotting
plt.figure(figsize=(10, 8))

# Plot the Cluster Streams
plt.scatter(cluster_streams['X_Centroid_x'], cluster_streams['Y_Centroid_x'], s=6.0, color='orange', label="Cluster Streams")

plt.xlabel('X Centroid')
plt.ylabel('Y Centroid')
plt.title('Cluster Streams Visualization')
plt.legend(title='Stream Type')
plt.grid(False)  # Set to True if you want to enable the grid
plt.show()


# In[39]:


import matplotlib.pyplot as plt
import pandas as pd

# Load the data
data = pd.read_csv('./HUC4 network properties.csv') 

# Filter the data to include only "Mixed roles" streams
mixed_roles_streams = data[data['Stream Type Final'] == 'Mixed roles']

# Plotting
plt.figure(figsize=(10, 8))

# Plot the Mixed Roles Streams
plt.scatter(mixed_roles_streams['X_Centroid_x'], mixed_roles_streams['Y_Centroid_x'], s=6.0, color='brown', label="Mixed Roles Streams")

plt.xlabel('X Centroid')
plt.ylabel('Y Centroid')
plt.title('Mixed Roles Streams Visualization')
plt.legend(title='Stream Type')
plt.grid(False)  # Set to True if you want to enable the grid
plt.show()

