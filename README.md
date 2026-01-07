# NetConUS

NetConUS is a network-connectivity based stream classification workflow built on **NHDPlus V2** stream network data.  
It (i) builds a stream graph, (ii) computes network metrics, (iii) clusters and assigns **fuzzy stream class labels**, and (iv) validates the classification using **Bayesian Neural Networks (BNNs)** with epistemic uncertainty.

---

### 1) `NetConUS graph development.py`
Constructs the stream network graph from NHDPlus V2 flowlines and accounts for fragmentation:
- **Impoundments/reservoirs:** identifies stream segments intersecting `NHDWaterbody` and optionally removes/labels them  
- **Dams:** spatially filters dams to the region and severs connectivity at nearest stream segment(s)

### 2) `Network properties calculation - Lower Mississippi region.py`
Computes network properties per stream segment:
- Degree centrality  
- Betweenness centrality (approximate option for scalability)  
- Closeness centrality  
- Clustering coefficient  
- Eigenvector centrality  
- (Optional) PageRank

### 3) `Stream classification based on the network properties.py`
Clusters stream segments and assigns semantic labels:
- Fits **Gaussian Mixture Models (GMM)**
- Uses **BIC** (and optionally AIC) to select number of clusters
- Computes cluster centroids
- Assigns labels using **fuzzy logic rules** derived from centroid patterns (high/medium/low across metrics)

### 4) `Validation of the NetConUS using BNNs.py`
Validates stream class predictions with a Bayesian Neural Network (Pyro):
- learns distributions over weights (posterior)
- predicts class probabilities
- quantifies **epistemic uncertainty** (posterior predictive variance / std of logits)
- visualizes results: confusion matrix, ROC, PR, UMAP uncertainty

---

## Data sources (official links)

### NHDPlus Version 2 (NHDPlus V2)
Download NHD (HU4 staged products) from the USGS “The National Map” staging bucket:

- **Base directory (NHD products):**
  https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Hydrography/NHD/

- **HUC-4 downloads (HU4):**
  https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Hydrography/NHD/HU4/

You will need at minimum:
- `NHDFlowline.shp`
- `NHDWaterbody.shp`

> Tip: Keep each HUC region extracted into its own folder under `data/Hydrography/<RegionName>/Hydrography/`

### National Inventory of Dams (NID)
Official portal for dam data:

https://nid.sec.usace.army.mil/#/

Export/download and store in `data/dams/` as e.g.:
- `nation.csv` (your scripts assume this name in places)

Required columns (typical):
- `Latitude`, `Longitude`, `Dam Name` (or equivalent)

---

## Setup

### Python environment

pip install pandas numpy geopandas shapely rtree pygeos
pip install networkx scikit-learn matplotlib seaborn tqdm
pip install torch pyro-ppl umap-learn
pip install imbalanced-learn
