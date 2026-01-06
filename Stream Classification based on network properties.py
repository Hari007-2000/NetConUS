#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This notebook implements a network-connectivity-based stream classification
framework for large river basins in the continental United States.

The workflow consists of:
1. Constructing stream networks from NHD Flowline data
2. Computing large-scale network connectivity metrics
3. Clustering streams using Gaussian Mixture Models (GMMs)
4. Selecting the optimal number of clusters using BIC/AIC
5. Interpreting clusters using fuzzy-logic-inspired semantic labels
6. Visualizing spatial patterns of stream classes across basins
"""

import geopandas as gpd
import pandas as pd
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
from tqdm import tqdm

# List of HUC-2 / HUC-3 basins included in the analysis
basin_paths = [
    './Ohio/Hydrography/NHDFlowline.shp',
    './Tennessee/Hydrography/NHDFlowline.shp',
    './Upper Mississipi/Hydrography/NHDFlowline.shp',
    './Lower Mississipi/Hydrography/NHDFlowline.shp',
    './Upper Missouri/Hydrography/NHDFlowline.shp',
    './Lower Missouri/Hydrography/NHDFlowline.shp',
    './Ark Red White/Hydrography/NHDFlowline.shp'
]

# Load and concatenate all basins
gdf_list = [gpd.read_file(path) for path in basin_paths]
gdf = pd.concat(gdf_list, ignore_index=True)

print(f"Total stream segments loaded: {len(gdf)}")

print("Computing approximate betweenness centrality...")
betweenness = nx.betweenness_centrality(G, k=100, normalized=True)

metrics_df = pd.read_csv('./HUC(1-3)Network_Properties.csv')

bc_df = pd.DataFrame({
    'COMID': gdf['COMID'],
    'Betweenness Centrality': pd.Series(betweenness)
})

data = metrics_df.merge(bc_df, on='COMID', how='inner')

features = data[
    [
        'Degree Centrality',
        'Betweenness Centrality',
        'Closeness Centrality',
        'Clustering Coefficient',
        'Eigenvector Centrality'
    ]
]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
features = imputer.fit_transform(features)

# Standardize
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

bic_scores, aic_scores = [], []
n_range = range(1, 11)

for n in n_range:
    gmm = GaussianMixture(n_components=n, random_state=0)
    gmm.fit(features_scaled)
    bic_scores.append(gmm.bic(features_scaled))
    aic_scores.append(gmm.aic(features_scaled))

plt.figure(figsize=(8, 6))
plt.plot(n_range, bic_scores, marker='o', label='BIC')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('GMM Model Selection using BIC')
plt.legend()
plt.savefig("Bayesian_Information_Criterion.png", dpi=1000)
plt.show()

gmm = GaussianMixture(n_components=5, random_state=0)
clusters = gmm.fit_predict(features_scaled)

data['Cluster'] = clusters
centroids = pd.DataFrame(gmm.means_, columns=features.columns)


centroid_df = pd.DataFrame(
    centroids,
    columns=[
        'Degree Centrality',
        'Betweenness Centrality',
        'Closeness Centrality',
        'Clustering Coefficient',
        'Eigenvector Centrality'
    ]
)

def fuzzy_level(value, low, high):
    """
    Assign fuzzy linguistic level based on relative position.
    """
    if value <= low:
        return 'Low'
    elif value >= high:
        return 'High'
    else:
        return 'Medium'

thresholds = {}
for col in centroid_df.columns:
    thresholds[col] = {
        'low': centroid_df[col].quantile(0.33),
        'high': centroid_df[col].quantile(0.66)
    }

def assign_stream_type_fuzzy(row, thresholds):
    deg = fuzzy_level(row['Degree Centrality'],
                      thresholds['Degree Centrality']['low'],
                      thresholds['Degree Centrality']['high'])

    bet = fuzzy_level(row['Betweenness Centrality'],
                      thresholds['Betweenness Centrality']['low'],
                      thresholds['Betweenness Centrality']['high'])

    clo = fuzzy_level(row['Closeness Centrality'],
                      thresholds['Closeness Centrality']['low'],
                      thresholds['Closeness Centrality']['high'])

    clu = fuzzy_level(row['Clustering Coefficient'],
                      thresholds['Clustering Coefficient']['low'],
                      thresholds['Clustering Coefficient']['high'])

    eig = fuzzy_level(row['Eigenvector Centrality'],
                      thresholds['Eigenvector Centrality']['low'],
                      thresholds['Eigenvector Centrality']['high'])

    # ---- FUZZY RULES ----

    # Peripheral Streams
    if deg == 'Low' and clo == 'Low':
        return 'Peripheral Streams'

    # Central Streams (globally important)
    if eig == 'High' and deg == 'High':
        return 'Central Streams'

    # Connector / Bridge Streams
    if bet == 'High':
        return 'Connector Streams'

    # Cluster Streams (locally dense)
    if clu == 'High':
        return 'Cluster Streams'

    # Convergent Streams (mixed functional roles)
    return 'Convergent Streams'

# Assign fuzzy labels to each cluster
cluster_labels = {}

for cluster_id, row in centroid_df.iterrows():
    cluster_labels[cluster_id] = assign_stream_type_fuzzy(row, thresholds)

cluster_labels

data['Stream Type'] = data['Cluster'].map(cluster_labels)

data.to_csv('./Stream_Classes_MidUS_revised.csv', index=False)

stream_data = pd.read_csv('./Stream_Classes_MidUS_revised.csv')

merged_gdf = gdf.merge(
    stream_data[['COMID', 'Stream Type']],
    on='COMID',
    how='left'
)

colors = {
    'Central Streams': 'blue',
    'Cluster Streams': 'green',
    'Convergent Streams': 'purple',
    'Peripheral Streams': 'orange',
    'Connector Streams': 'red'
}

fig, ax = plt.subplots(figsize=(15, 10))
for s_type, color in colors.items():
    merged_gdf[merged_gdf['Stream Type'] == s_type].plot(
        ax=ax,
        color=color,
        linewidth=1,
        label=s_type
    )

plt.title('Stream Classification Based on Network Connectivity')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(False)
plt.tight_layout()
plt.show()



