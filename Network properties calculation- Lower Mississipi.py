#!/usr/bin/env python
# coding: utf-8

# This notebook mainly focuses on calculating the Network properties for the developed Network graph(esspecially for the Lower Mississpi Region)
# 
# The network properties calculated are : Degree centrality, Betweenness Centrality, Eigen-vector centrality, Closeness centrality, Page rank, Clustering co-efficient
# 
# The notebook also mentions how to do parallel computation for the node properties calculation

# In[ ]:


import pandas as pd
import geopandas as gpd
import networkx as nx


# In[2]:


# Load NHD flowline data into a GeoDataFrame
gdf = gpd.read_file('./Hydrography/NHDFlowline.shp')


# In[3]:


gdf.head()


# This code can be used to Visualize the HUC-2 streams in this case, the Lower Mississipi, location in the United states

# In[1]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Load the shapefiles for Ohio and Tennessee NHDFlowlines
ohio_gdf = gpd.read_file('./Hydrography/NHDFlowline.shp')


# Combine the shapefiles into a single GeoDataFrame
combined_gdf = pd.concat([ohio_gdf], ignore_index=True)

# Extract latitude and longitude of centroids
combined_gdf['geometry_centroid'] = combined_gdf['geometry'].centroid
lats = combined_gdf['geometry_centroid'].y.values
lons = combined_gdf['geometry_centroid'].x.values

# Set up the map
fig, ax = plt.subplots(figsize=(10, 8))
m = Basemap(
    projection='merc',
    llcrnrlat=24,
    urcrnrlat=50,
    llcrnrlon=-125,
    urcrnrlon=-66,
    resolution='i',
    ax=ax
)

# Draw the US border
m.drawcoastlines()
m.drawcountries()

# Plot the Ohio and Tennessee streams on the map
x, y = m(lons, lats)
m.scatter(x, y, marker='o', color='blue', s=0.5, zorder=5, label='Ohio & Tennessee Streams')

# Add legend and adjust the layout
#plt.legend(loc='lower left', title="Regions")
plt.tight_layout()

plt.savefig("LM_USA.png", dpi = 1000)

# Show the plot
plt.show()


# This code is parallelized version that can be used to calculate the node properties for the Network properties

# In[11]:


import geopandas as gpd
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures

# Load the NHD file into a GeoDataFrame
gdf = gpd.read_file('./Hydrography/NHDFlowline.shp')

# Create an empty graph
G = nx.Graph()

# Create a spatial index for the GeoDataFrame to speed up intersection checks
gdf_sindex = gdf.sindex

# Add nodes and edges to the graph
for idx, row in gdf.iterrows():
    point = row['geometry'].centroid
    G.add_node(idx, x=point.x, y=point.y)
    
    # Get potential neighbors using the spatial index
    possible_neighbors = list(gdf_sindex.intersection(row['geometry'].bounds))
    neighbors = [n for n in possible_neighbors if n != idx and gdf.loc[n, 'geometry'].intersects(row['geometry'])]

    # Add edges between the current node and its neighbors
    for neighbor in neighbors:
        G.add_edge(idx, neighbor)

def calculate_degree_centrality(G):
    return nx.degree_centrality(G)

def calculate_closeness_centrality(G):
    return nx.closeness_centrality(G)

def calculate_clustering_coefficient(G):
    return nx.clustering(G)

def calculate_eigenvector_centrality(G):
    return nx.eigenvector_centrality_numpy(G)

# Calculate centrality measures in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    future_degree = executor.submit(calculate_degree_centrality, G)
    future_closeness = executor.submit(calculate_closeness_centrality, G)
    future_clustering = executor.submit(calculate_clustering_coefficient, G)
    future_eigenvector = executor.submit(calculate_eigenvector_centrality, G)

    degree_centrality = future_degree.result()
    closeness_centrality = future_closeness.result()
    clustering_coefficient = future_clustering.result()
    eigenvector_centrality = future_eigenvector.result()

# Combine all measures into a DataFrame
metrics_df = pd.DataFrame({
    'COMID': gdf['COMID'],
    'X_Centroid': [G.nodes[idx]['x'] for idx in G.nodes],
    'Y_Centroid': [G.nodes[idx]['y'] for idx in G.nodes],
    'Degree Centrality': pd.Series(degree_centrality),
    'Closeness Centrality': pd.Series(closeness_centrality),
    'Clustering Coefficient': pd.Series(clustering_coefficient),
    'Eigenvector Centrality': pd.Series(eigenvector_centrality)
})

# Export the DataFrame to a CSV file
metrics_df.to_csv('./LowerMississpi_Networkproperties_degree.csv', index=False)

# Draw the graph
pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes}
nx.draw(G, pos, node_size=0.001, node_color='b', edge_color='g', with_labels=False)
plt.savefig("Undirected_graph1.png", dpi =600)
plt.show()


# This code can be used to build an Undirected Network graph 

# In[21]:


import geopandas as gpd
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures

# Load the NHD file into a GeoDataFrame
gdf = gpd.read_file('./Hydrography/NHDFlowline.shp')

# Create an empty graph
G = nx.Graph()

# Create a spatial index for the GeoDataFrame to speed up intersection checks
gdf_sindex = gdf.sindex

# Add nodes and edges to the graph
for idx, row in gdf.iterrows():
    point = row['geometry'].centroid
    G.add_node(idx, x=point.x, y=point.y)
    
    # Get potential neighbors using the spatial index
    possible_neighbors = list(gdf_sindex.intersection(row['geometry'].bounds))
    neighbors = [n for n in possible_neighbors if n != idx and gdf.loc[n, 'geometry'].intersects(row['geometry'])]

    # Add edges between the current node and its neighbors
    for neighbor in neighbors:
        G.add_edge(idx, neighbor)

# Draw the graph
pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes}
nx.draw(G, pos, node_size=0.001, node_color='b', edge_color='g', with_labels=False)
plt.savefig("Undirected_graph.png", dpi =600)
plt.show()


# This code below can be used to build directed graph based on the Upstream/ Downstream connections

# In[ ]:


import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Load NHD flowline data into a GeoDataFrame
gdf = gpd.read_file('/content/drive/MyDrive/Classroom/Shape/NHDFlowline.shp')

# Create an empty NetworkX DiGraph (Directed Graph)
G = nx.DiGraph()

# Add nodes to the graph from NHD flowline data
for idx, row in gdf.iterrows():
    point = row['geometry'].centroid
    G.add_node(idx, x=point.x, y=point.y)

# Create a spatial index for the GeoDataFrame to speed up intersection checks
gdf_sindex = gdf.sindex

# Add edges to the graph with directionality
for idx, row in gdf.iterrows():
    possible_neighbors = list(gdf_sindex.intersection(row['geometry'].bounds))
    neighbors = [n for n in possible_neighbors if n != idx and gdf.loc[n, 'geometry'].intersects(row['geometry'])]
    for neighbor in neighbors:
        # Check the flow direction of the stream to determine the edge direction
        if gdf.loc[idx, 'flowdir'] == 1:  # Flow is from node idx to neighbor
            G.add_edge(idx, neighbor)
        elif gdf.loc[idx, 'flowdir'] == 0:  # Flow is from neighbor to node idx
            G.add_edge(neighbor, idx)

# Draw the graph with dam nodes highlighted
pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes}
nx.draw(G, pos, node_size=0.01, node_color='b', edge_color='g', with_labels=False)

plt.show()


# In[20]:


# Draw the graph
pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes}
plt.figure(figsize=(10, 10))
nx.draw(G, pos, node_size=0.01, node_color='blue', edge_color='green', with_labels=False, arrows=True)
plt.title("Directed River Network Graph")

plt.savefig("directed+graph_LM.png", dpi =600)
plt.show()


# Visualizing the spatial variation of the Degree centrality for the Lower Mississipi region

# In[4]:


# Extract betweenness centrality values as a list
dc_values = [degree_centrality[node] for node in G.nodes]

plt.figure(figsize=(12, 10))
ax = plt.gca()

# Draw the graph
nc = nx.draw_networkx_nodes(G, pos, node_color=dc_values, node_size=1.30, cmap=plt.cm.rainbow, alpha=0.7)
nx.draw_networkx_edges(G, pos, edge_color='gray')
plt.axis('off')  # Hide the axes

plt.colorbar(nc)

plt.savefig("Degree_Centrality.png", dpi = 600)

#plt.title('NetworkX Graph with Node Color based on Betweenness Centrality')
plt.show()


# In[12]:


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

# Define the number of classes
num_classes = 5

# Assuming 'degree_centrality' is a dictionary with nodes as keys and centrality scores as values
# Calculate the percentile thresholds for each class
percentiles = np.linspace(0, 100, num_classes + 1)
thresholds = np.percentile(list(degree_centrality.values()), percentiles)

# Create a dictionary to map degree centrality values to classes
dc_classes = {}
for i in range(num_classes):
    lower_bound = thresholds[i]
    upper_bound = thresholds[i+1]
    dc_classes[i] = [node for node in G.nodes if lower_bound <= degree_centrality[node] <= upper_bound]

# Use a seaborn diverging palette
class_colors = sns.color_palette("RdYlGn", num_classes)  # 'RdYlGn' is a red-yellow-green palette

# Draw the graph with nodes colored based on degree centrality classes
plt.figure(figsize=(10, 8))
for i, class_nodes in dc_classes.items():
    nx.draw_networkx_nodes(G, pos, nodelist=class_nodes, node_size=0.5, node_color=class_colors[i])

nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.7)
plt.title('Network Graph with Node Color Based on Degree Centrality Classes')

# Creating a legend manually for better control
from matplotlib.patches import Patch
legend_labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']
legend_handles = [Patch(facecolor=color, label=label) for color, label in zip(class_colors, legend_labels)]
plt.legend(handles=legend_handles, title="Degree Centrality Classes")

plt.savefig("Degree_centrality_Classes.png", dpi = 600)

plt.show()


# In[16]:


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

# Define the number of classes
num_classes = 5
class_names = ["Very Low", "Low", "Medium", "High", "Very High"]

# Assuming 'degree_centrality' is a dictionary with nodes as keys and centrality scores as values
# Calculate the percentile thresholds for each class
percentiles = np.linspace(0, 100, num_classes + 1)
thresholds = np.percentile(list(degree_centrality.values()), percentiles)

# Create a dictionary to map degree centrality values to classes
dc_classes = {}
for i in range(num_classes):
    lower_bound = thresholds[i]
    upper_bound = thresholds[i + 1]
    dc_classes[i] = [node for node in G.nodes if lower_bound <= degree_centrality[node] <= upper_bound]

# Use a seaborn diverging palette
class_colors = sns.color_palette("RdYlGn", num_classes)  # 'RdYlGn' is a red-yellow-green palette

# Draw the graph with nodes colored based on degree centrality classes
plt.figure(figsize=(10, 8))
for i, class_nodes in dc_classes.items():
    nx.draw_networkx_nodes(G, pos, nodelist=class_nodes, node_size=0.5, node_color=[class_colors[i]])

nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.7)
plt.title('Network Graph with Node Color Based on Degree Centrality Classes')

# Create the legend with scientific notation ranges and class names
legend_labels = []
for i in range(num_classes):
    lower = thresholds[i]
    upper = thresholds[i + 1]
    label = f"{class_names[i]} ({lower:.1e} – {upper:.1e})"
    legend_labels.append(label)

legend_handles = [Patch(facecolor=color, label=label) for color, label in zip(class_colors, legend_labels)]
plt.legend(handles=legend_handles, title="Degree Centrality Ranges", loc='upper left')

plt.savefig("Degree_centrality_Classes.png", dpi=600)
plt.show()


# In[6]:


pip install jenkspy


# Visualizing the Spatial variation of the Closness Centrality for the LM

# In[15]:


# Extract betweenness centrality values as a list
cc_values = [closeness_centrality[node] for node in G.nodes]

plt.figure(figsize=(12, 10))
ax = plt.gca()

# Draw the graph
nc = nx.draw_networkx_nodes(G, pos, node_color=cc_values, node_size=1.30, cmap=plt.cm.rainbow, alpha=0.7)
nx.draw_networkx_edges(G, pos, edge_color='gray')
plt.axis('off')  # Hide the axes

plt.colorbar(nc)

#plt.title('NetworkX Graph with Node Color based on Betweenness Centrality')

plt.savefig('Closeness_Centrality.png', format='png', dpi=600)  # Save as PNG file with 300 dpi
plt.show()


# In[14]:


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

# Define the number of classes
num_classes = 5

# Calculate the percentile thresholds for each class
percentiles = np.linspace(0, 100, num_classes + 1)
thresholds = np.percentile(cc_values, percentiles)

# Create a dictionary to map closeness centrality values to classes
cc_classes = {}
for i in range(num_classes):
    lower_bound = thresholds[i]
    upper_bound = thresholds[i+1]
    cc_classes[i] = [node for node in G.nodes if lower_bound <= closeness_centrality[node] <= upper_bound]

# Use a seaborn diverging palette
class_colors = sns.color_palette("Spectral", num_classes)  # 'Spectral' or another diverging palette

# Draw the graph with nodes colored based on degree centrality classes
plt.figure(figsize=(10, 8))
for i, class_nodes in cc_classes.items():
    nx.draw_networkx_nodes(G, pos, nodelist=class_nodes, node_size=10, node_color=class_colors[i])

nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.7)
#plt.title('Network Graph with Node Color Based on Closeness Centrality Classes')

# Creating a legend manually
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=color, label=label) for color, label in zip(class_colors, ['Very low', 'Low', 'Medium', 'High', 'Very high'])]
plt.legend(handles=legend_handles, title="Closeness Centrality Classes")

# To save the figure
plt.savefig("Centrality_classes.png", dpi =600)
plt.show()



# Visualizing the Spatial variation of the Clustering co-efficient for the LM

# In[16]:


# Extract betweenness centrality values as a list
ce_values = [clustering_coefficient[node] for node in G.nodes]

plt.figure(figsize=(12, 10))
ax = plt.gca()

# Draw the graph
nc = nx.draw_networkx_nodes(G, pos, node_color=ce_values, node_size=1.30, cmap=plt.cm.rainbow, alpha=0.7)
nx.draw_networkx_edges(G, pos, edge_color='gray')
plt.axis('off')  # Hide the axes

plt.colorbar(nc)

plt.savefig("Clustering_coefficient.png", dpi = 600)

#plt.title('NetworkX Graph with Node Color based on Betweenness Centrality')
plt.show()


# In[17]:


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

# Assuming 'clustering_coefficient' is a dictionary from nx.clustering(G) with nodes as keys
# Define the number of classes
num_classes = 5

# Calculate the percentile thresholds for each class
percentiles = np.linspace(0, 100, num_classes + 1)
thresholds = np.percentile(list(clustering_coefficient.values()), percentiles)

# Create a dictionary to map clustering coefficient values to classes
clustering_classes = {}
for i in range(num_classes):
    lower_bound = thresholds[i]
    upper_bound = thresholds[i+1]
    clustering_classes[i] = [node for node in G.nodes if lower_bound <= clustering_coefficient[node] <= upper_bound]

# Use a seaborn diverging palette
class_colors = sns.color_palette("Spectral", num_classes)  # 'Blues' is good for showing progression in density

# Draw the graph with nodes colored based on clustering coefficient classes
plt.figure(figsize=(10, 8))
for i, class_nodes in clustering_classes.items():
    nx.draw_networkx_nodes(G, pos, nodelist=class_nodes, node_size=1.0, node_color=class_colors[i])

nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.7)
plt.title('Network Graph with Node Color Based on Clustering Coefficient')

# Creating a legend manually for better control
from matplotlib.patches import Patch
legend_labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']
legend_handles = [Patch(facecolor=color, label=label) for color, label in zip(class_colors, legend_labels)]
plt.legend(handles=legend_handles, title="Clustering Coefficient Classes")

plt.savefig("Clusteringco_classes.png", dpi =600)
plt.show()


# Spatial Variation of the eigen-vector centrality for the LM region

# In[11]:


# Extract betweenness centrality values as a list
ec_values = [eigenvector_centrality[node] for node in G.nodes]

plt.figure(figsize=(12, 10))
ax = plt.gca()

# Draw the graph
nc = nx.draw_networkx_nodes(G, pos, node_color=ec_values, node_size=0.30, cmap=plt.cm.rainbow, alpha=0.7)
nx.draw_networkx_edges(G, pos, edge_color='gray')
plt.axis('off')  # Hide the axes

plt.colorbar(nc)

#plt.title('NetworkX Graph with Node Color based on Betweenness Centrality')
plt.show()


# The code below visualizes the dams located in the LM region

# In[13]:


import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Load NHD flowline data into a GeoDataFrame
gdf = gpd.read_file('./Hydrography/NHDFlowline.shp')

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


# The code below can be used to remove the nodes that are impounded by the reservoirs by spatially joining them and removing the COMIDs that are impounded by them

# In[14]:


import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt

# Load the NHD flowline and waterbody (reservoir) data into GeoDataFrames
gdf_flowline = gpd.read_file('./Hydrography/NHDFlowline.shp')
gdf_reservoir = gpd.read_file('./Hydrography/NHDWaterbody.shp')

# Perform a spatial join between flowlines and reservoirs
gdf_impounded = gpd.sjoin(gdf_flowline, gdf_reservoir, how="inner", predicate="intersects")

# Check and print the columns of the joined GeoDataFrame
print("Impounded columns:", gdf_impounded.columns)

# Extract the COMIDs of the impounded flowlines using the correct column name
impounded_comids = gdf_impounded['COMID_left'].unique()

# Create an empty graph
G = nx.Graph()

# Precompute centroids to avoid recalculating them in the loop
centroids = gdf_flowline['geometry'].centroid

# Create a spatial index for the GeoDataFrame to speed up intersection checks
gdf_flowline_sindex = gdf_flowline.sindex

# Add nodes and edges to the graph
for idx, row in gdf_flowline.iterrows():
    point = centroids[idx]
    G.add_node(idx, x=point.x, y=point.y, comid=row['COMID'])
    
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


# In[15]:


# Optionally, plot the graph without the impounded nodes
pos = {idx: (row['geometry'].centroid.x, row['geometry'].centroid.y) for idx, row in gdf_flowline.iterrows() if idx in G.nodes}
plt.figure(figsize=(10, 10))
nx.draw(G, pos, node_size=0.5)
plt.title("Network Graph Without Impounded Flowlines")
plt.show()


# The code below can be efficiently used to calculate the Betweenness centrality for large graphs 

# In[22]:


import geopandas as gpd
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the NHD file into a GeoDataFrame
gdf = gpd.read_file('./Hydrography/NHDFlowline.shp')

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
betweenness_df.to_csv('./LM_BetweennessCentrality_Approx.csv', index=False)

print("Betweenness Centrality has been calculated and exported to Mid-Atlantic_BetweennessCentrality_Approx.csv")

# Visualize the graph
pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes}
nx.draw(G, pos, node_size=0.001, node_color='b', edge_color='g', with_labels=False)
plt.title("Stream Network Visualization")
plt.show()


# In[25]:


# Extract betweenness centrality values as a list
bc_values = [betweenness_centrality[node] for node in G.nodes]

plt.figure(figsize=(12, 10))
ax = plt.gca()

# Draw the graph
nc = nx.draw_networkx_nodes(G, pos, node_color=bc_values, node_size=2.30, cmap=plt.cm.rainbow, alpha=0.7)
nx.draw_networkx_edges(G, pos, edge_color='gray')
plt.axis('off')  # Hide the axes

plt.colorbar(nc)

#plt.title('NetworkX Graph with Node Color based on Betweenness Centrality')

plt.savefig('LM_Betweenness_Centrality.png', format='png', dpi=600)  # Save as PNG file with 300 dpi
plt.show()


# In[27]:


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

nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.7)
plt.title('Network Graph with Node Color Based on Betweenness Centrality Classes')

# Creating a legend manually for better control
from matplotlib.patches import Patch
legend_labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']
legend_handles = [Patch(facecolor=color, label=label) for color, label in zip(class_colors, legend_labels)]
plt.legend(handles=legend_handles, title="Betweenness Centrality Classes")

plt.savefig("Betweenness_Centrality_classes.png", dpi = 1000)

plt.show()


# In[ ]:




