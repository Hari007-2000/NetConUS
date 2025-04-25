# NetConUS

The NetConUS code is organized into 4 files

1. NetConUS graph development
2. Network properties calculation -Lower Mississippi region
3. Stream classification based on the network properties
4. Validation of the NetConUS using BNNs


NetConUS graph development.py: Constructs the stream network graph using NHDPlus V2 data, incorporating dam and impoundment impacts to reflect real-world fragmentation.

Network properties calculation - Lower Mississippi region.py: Computes essential network metrics (e.g., degree, betweenness, closeness) for each stream segment, forming the basis for classification.

Stream classification based on the network properties.py: Clusters the stream nodes using Gaussian Mixture Models, and assigns fuzzy logic-based stream class labels.

Validation of the NetConUS using BNNs.py: Validates the classification using Bayesian Neural Networks (BNNs), quantifies epistemic uncertainty, and visualizes results with metrics like confusion matrix, ROC, and UMAP.
