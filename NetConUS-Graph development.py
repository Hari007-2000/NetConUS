#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This notebook models stream networks using NHD Flowline data and
explicitly incorporates fragmentation caused by reservoirs and dams.
The workflow includes:
1. Building stream connectivity graphs
2. Identifying impounded stream segments
3. Modifying networks based on dams
4. Visualizing stream connectivity classes
"""


import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
from geopy.distance import geodesic

## Load NHD Flowline shapefile
gdf_flowline = gpd.read_file('./Shape/NHDFlowline.shp')

## Inspect the attribute table
gdf_flowline.head()

## Load NHD Waterbody shapefile (includes reservoirs)
gdf_waterbody = gpd.read_file('./Shape/NHDWaterbody.shp')

## Plot both datasets together
fig, ax = plt.subplots(figsize=(15, 10))

gdf_flowline.plot(
    ax=ax,
    color='blue',
    linewidth=0.5,
    label='Flowlines'
)

gdf_waterbody.plot(
    ax=ax,
    color='green',
    alpha=0.8,
    label='Waterbodies / Reservoirs'
)

plt.title("NHD Flowlines and Waterbodies (HUC-4)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.savefig("Reservoirs.png", dpi=1000)
plt.show()

## Spatially join flowlines with waterbodies
gdf_impounded = gpd.sjoin(
    gdf_flowline,
    gdf_waterbody,
    how="inner",
    predicate="intersects"
)

## Extract COMIDs of impounded stream segments
impounded_comids = gdf_impounded['permanent__left'].unique()

print(f"Number of impounded stream segments: {len(impounded_comids)}")



## Initialize an undirected graph
G = nx.Graph()

### Precompute centroids for efficiency
centroids = gdf_flowline.geometry.centroid

### Spatial index for fast neighbor lookup
flowline_sindex = gdf_flowline.sindex

### Add nodes and edges
for idx, row in gdf_flowline.iterrows():
    centroid = centroids.loc[idx]
    
    # Add stream segment as a node
    G.add_node(
        idx,
        x=centroid.x,
        y=centroid.y,
        comid=row['permanent_']
    )
    
    # Identify intersecting neighbors
    candidate_neighbors = flowline_sindex.intersection(row.geometry.bounds)
    neighbors = [
        n for n in candidate_neighbors
        if n != idx and gdf_flowline.loc[n, 'geometry'].intersects(row.geometry)
    ]
    
    # Add edges between connected stream segments
    for neighbor in neighbors:
        G.add_edge(idx, neighbor)

## Identify nodes corresponding to impounded COMIDs
nodes_to_remove = [
    node for node, attr in G.nodes(data=True)
    if attr['comid'] in impounded_comids
]

## Remove impounded nodes
G.remove_nodes_from(nodes_to_remove)

## Plot the modified network
pos = {
    node: (attr['x'], attr['y'])
    for node, attr in G.nodes(data=True)
}

plt.figure(figsize=(10, 10))
nx.draw(G, pos, node_size=5, node_color='black')
plt.title("Stream Network with Impounded Segments Removed")
plt.show()

## Load dam dataset
dam_data = pd.read_csv('./nation.csv')

## Compute spatial extent of stream network
minx, miny, maxx, maxy = gdf_flowline.total_bounds

## Filter dams within the spatial extent
dam_data_filtered = dam_data[
    (dam_data['Latitude'] >= miny) &
    (dam_data['Latitude'] <= maxy) &
    (dam_data['Longitude'] >= minx) &
    (dam_data['Longitude'] <= maxx)
]

def find_nearest_stream_node(dam_latlon, graph_nodes):
    """Find the nearest stream node to a dam location."""
    min_dist = float('inf')
    nearest = None
    
    for node, data in graph_nodes:
        node_latlon = (data['y'], data['x'])
        dist = geodesic(dam_latlon, node_latlon).meters
        
        if dist < min_dist:
            min_dist = dist
            nearest = node
            
    return nearest


# Remove edges at dam locations
for _, dam in dam_data_filtered.iterrows():
    dam_point = (dam['Latitude'], dam['Longitude'])
    nearest_node = find_nearest_stream_node(dam_point, G.nodes(data=True))
    
    if nearest_node is not None:
        G.remove_edges_from(list(G.edges(nearest_node)))

plt.figure(figsize=(10, 10))
nx.draw(
    G,
    pos,
    node_size=0.5,
    node_color='blue',
    edge_color='green'
)
plt.title("Stream Network Fragmented by Dams")
plt.show()


df_network = pd.read_csv('./HUC4 network properties.csv')


df_network.loc[
    df_network['COMID'].isin(impounded_comids),
    'Stream Type Final'
] = 'Impounded'

df_network.to_csv('./HUC4 network properties.csv', index=False)


import umap

# Dimensionality reduction
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(features)

umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
umap_df['Cluster'] = data['Cluster']

plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=umap_df,
    x='UMAP1',
    y='UMAP2',
    hue='Cluster',
    palette='Spectral',
    s=10
)
plt.title('U-MAP Visualization of Stream Connectivity Clusters')
plt.savefig("Clusters.png", dpi=1000)
plt.show()

def plot_stream_type(stream_type, color):
    subset = data[data['Stream Type Final'] == stream_type]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(
        subset['X_Centroid_x'],
        subset['Y_Centroid_x'],
        s=6,
        color=color,
        label=stream_type
    )
    plt.xlabel('X Centroid')
    plt.ylabel('Y Centroid')
    plt.title(f'{stream_type} Visualization')
    plt.legend()
    plt.show()


plot_stream_type('Peripheral Streams', 'blue')
plot_stream_type('Bridge Streams', 'red')
plot_stream_type('High Centrality Streams', 'green')
plot_stream_type('Impounded', 'purple')
plot_stream_type('Cluster Streams', 'orange')
plot_stream_type('Mixed roles', 'brown')




