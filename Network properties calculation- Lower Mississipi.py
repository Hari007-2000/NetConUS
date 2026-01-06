#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This notebook focuses on computing network connectivity properties for a
stream network derived from NHD Flowline data, with a particular emphasis
on the Lower Mississippi (HUC-2) region.

The following network metrics are computed:
- Degree centrality
- Betweenness centrality (exact and approximate)
- Closeness centrality
- Eigenvector centrality
- PageRank
- Clustering coefficient

Given the large size of the river network, parallel and approximate
computations are used where appropriate.
"""
import pandas as pd
import geopandas as gpd
import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load NHD Flowline shapefile for the HUC-2 region
gdf = gpd.read_file('./Hydrography/NHDFlowline.shp')


gdf.head()

from mpl_toolkits.basemap import Basemap

# Compute centroids of flowline geometries
gdf['centroid'] = gdf.geometry.centroid
lats = gdf['centroid'].y.values
lons = gdf['centroid'].x.values

# Create a basemap of the continental U.S.
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

m.drawcoastlines()
m.drawcountries()

x, y = m(lons, lats)
m.scatter(x, y, s=0.5, color='blue', zorder=5)

plt.tight_layout()
plt.savefig("LM_USA.png", dpi=1000)
plt.show()

# Initialize an undirected graph
G = nx.Graph()

# Create a spatial index to accelerate intersection checks
gdf_sindex = gdf.sindex

# Add nodes and edges
for idx, row in gdf.iterrows():
    centroid = row.geometry.centroid
    G.add_node(idx, x=centroid.x, y=centroid.y)
    
    # Identify spatially intersecting neighbors
    candidate_neighbors = gdf_sindex.intersection(row.geometry.bounds)
    neighbors = [
        n for n in candidate_neighbors
        if n != idx and gdf.loc[n, 'geometry'].intersects(row.geometry)
    ]
    
    for neighbor in neighbors:
        G.add_edge(idx, neighbor)

import concurrent.futures

def compute_degree(G):
    return nx.degree_centrality(G)

def compute_closeness(G):
    return nx.closeness_centrality(G)

def compute_clustering(G):
    return nx.clustering(G)

def compute_eigenvector(G):
    return nx.eigenvector_centrality_numpy(G)

# Execute computations in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    f_degree = executor.submit(compute_degree, G)
    f_close  = executor.submit(compute_closeness, G)
    f_clust  = executor.submit(compute_clustering, G)
    f_eigen  = executor.submit(compute_eigenvector, G)

    degree_centrality = f_degree.result()
    closeness_centrality = f_close.result()
    clustering_coefficient = f_clust.result()
    eigenvector_centrality = f_eigen.result()

metrics_df = pd.DataFrame({
    'COMID': gdf['COMID'],
    'X_Centroid': [G.nodes[n]['x'] for n in G.nodes],
    'Y_Centroid': [G.nodes[n]['y'] for n in G.nodes],
    'Degree': pd.Series(degree_centrality),
    'Closeness': pd.Series(closeness_centrality),
    'Clustering': pd.Series(clustering_coefficient),
    'Eigenvector': pd.Series(eigenvector_centrality)
})

metrics_df.to_csv(
    './LowerMississippi_NetworkProperties.csv',
    index=False
)

# Initialize directed graph
DG = nx.DiGraph()

# Add nodes
for idx, row in gdf.iterrows():
    centroid = row.geometry.centroid
    DG.add_node(idx, x=centroid.x, y=centroid.y)

# Add directed edges based on flow direction
for idx, row in gdf.iterrows():
    candidate_neighbors = gdf_sindex.intersection(row.geometry.bounds)
    neighbors = [
        n for n in candidate_neighbors
        if n != idx and gdf.loc[n, 'geometry'].intersects(row.geometry)
    ]
    
    for neighbor in neighbors:
        if row['flowdir'] == 1:
            DG.add_edge(idx, neighbor)
        elif row['flowdir'] == 0:
            DG.add_edge(neighbor, idx)

pos = {n: (G.nodes[n]['x'], G.nodes[n]['y']) for n in G.nodes}
dc_values = [degree_centrality[n] for n in G.nodes]

plt.figure(figsize=(12, 10))
nodes = nx.draw_networkx_nodes(
    G, pos,
    node_color=dc_values,
    node_size=1.2,
    cmap=plt.cm.rainbow
)
nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3)
plt.colorbar(nodes)
plt.axis('off')

plt.savefig("Degree_Centrality.png", dpi=600)
plt.show()

num_classes = 5
percentiles = np.linspace(0, 100, num_classes + 1)
thresholds = np.percentile(dc_values, percentiles)

dc_classes = {
    i: [n for n in G.nodes if thresholds[i] <= degree_centrality[n] <= thresholds[i+1]]
    for i in range(num_classes)
}

colors = sns.color_palette("RdYlGn", num_classes)

plt.figure(figsize=(10, 8))
for i, nodes in dc_classes.items():
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=0.6, node_color=[colors[i]])

nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.4)

from matplotlib.patches import Patch
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
handles = [Patch(facecolor=c, label=l) for c, l in zip(colors, labels)]
plt.legend(handles=handles, title="Degree Centrality Classes")

plt.savefig("Degree_Centrality_Classes.png", dpi=600)
plt.show()


from tqdm import tqdm

print("Computing approximate betweenness centrality...")
betweenness_centrality = nx.betweenness_centrality(
    G,
    k=100,          # Number of sampled nodes
    normalized=True
)

betweenness_df = pd.DataFrame({
    'COMID': gdf['COMID'],
    'X_Centroid': [G.nodes[n]['x'] for n in G.nodes],
    'Y_Centroid': [G.nodes[n]['y'] for n in G.nodes],
    'Betweenness': pd.Series(betweenness_centrality)
})

betweenness_df.to_csv(
    './LM_BetweennessCentrality_Approx.csv',
    index=False
)


