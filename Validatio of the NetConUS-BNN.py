"""
This notebook validates the network-connectivity-based stream classification
using a Bayesian Neural Network (BNN).

The objective is to quantify epistemic uncertainty arising from:
- Limited structural network information
- Overlapping connectivity roles across stream classes

Bayesian inference is performed using Pyro with a No-U-Turn Sampler (NUTS),
allowing posterior distributions over model parameters and probabilistic
predictions for stream type classification.
"""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import MCMC, NUTS, Predictive

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

data = pd.read_csv('./HUC4_network_properties.csv')

feature_columns = [
    'Degree Centrality',
    'Betweenness Centrality',
    'Closeness Centrality',
    'Clustering Coefficient',
    'Eigenvector Centrality'
]

label_column = 'Stream Type Final'

# Extract features and labels
X = data[feature_columns].values
y = data[label_column].values

# Encode categorical stream labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test,  dtype=torch.long)

class BNN(PyroModule):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super().__init__()

        self.fc1 = PyroModule[nn.Linear](input_dim, hidden_dim1)
        self.fc1.weight = PyroSample(
            dist.Normal(0., 1.).expand([hidden_dim1, input_dim]).to_event(2)
        )
        self.fc1.bias = PyroSample(
            dist.Normal(0., 1.).expand([hidden_dim1]).to_event(1)
        )

        self.fc2 = PyroModule[nn.Linear](hidden_dim1, hidden_dim2)
        self.fc2.weight = PyroSample(
            dist.Normal(0., 1.).expand([hidden_dim2, hidden_dim1]).to_event(2)
        )
        self.fc2.bias = PyroSample(
            dist.Normal(0., 1.).expand([hidden_dim2]).to_event(1)
        )

        self.out = PyroModule[nn.Linear](hidden_dim2, output_dim)
        self.out.weight = PyroSample(
            dist.Normal(0., 1.).expand([output_dim, hidden_dim2]).to_event(2)
        )
        self.out.bias = PyroSample(
            dist.Normal(0., 1.).expand([output_dim]).to_event(1)
        )

    def forward(self, x, y=None):
        x = torch.tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)

        with pyro.plate("data", x.shape[0]):
            pyro.sample(
                "obs",
                dist.Categorical(logits=logits),
                obs=y
            )
        return logits

input_dim = X_train.shape[1]
hidden_dim1, hidden_dim2 = 16, 8
output_dim = len(np.unique(y))

bnn = BNN(input_dim, hidden_dim1, hidden_dim2, output_dim)

nuts_kernel = NUTS(bnn)
mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=200)

mcmc.run(X_train, y_train)

predictive = Predictive(
    bnn,
    posterior_samples=mcmc.get_samples(),
    return_sites=["_RETURN"]
)

predictions = predictive(X_test)
logits = predictions["_RETURN"]

# Mean prediction across posterior samples
mean_logits = logits.mean(dim=0)
std_logits  = logits.std(dim=0)

predicted_classes = mean_logits.argmax(dim=1)

print("Test Accuracy:", accuracy_score(y_test, predicted_classes))

conf_matrix = confusion_matrix(y_test, predicted_classes)

plt.figure(figsize=(10, 7))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("BNN Confusion Matrix")
plt.savefig("BNN_Confusion_Matrix.png", dpi=1000)
plt.show()

print(
    classification_report(
        y_test,
        predicted_classes,
        target_names=label_encoder.classes_
    )
)

uncertainty_df = pd.DataFrame({
    "True Class": label_encoder.inverse_transform(y_test.numpy()),
    "Predicted Class": label_encoder.inverse_transform(predicted_classes.numpy()),
    "Mean Logit": mean_logits.max(dim=1).values.numpy(),
    "Logit Variance": std_logits.max(dim=1).values.numpy()
})

plt.figure(figsize=(8, 6))
sns.boxplot(x="True Class", y="Mean Logit", data=uncertainty_df)
plt.xticks(rotation=45, ha="right")
plt.title("Posterior Predictive Confidence by Stream Class")
plt.ylabel("Mean Posterior Logit")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("BNN_Uncertainty_Boxplot.png", dpi=1000)
plt.show()

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from itertools import cycle

y_test_bin = label_binarize(y_test, classes=range(output_dim))

# ROC curves
plt.figure(figsize=(10, 7))
colors = cycle(["blue", "orange", "green", "red", "purple"])

for i, color in zip(range(output_dim), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], mean_logits[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Stream Classification")
plt.legend()
plt.savefig("BNN_ROC_Curves.png", dpi=1000)
plt.show()

results_out = results.copy()
results_out["True Label"] = y_test.numpy()
results_out["Predicted Label"] = predicted_classes.numpy()

results_out.to_csv(
    "./BNN_Predictions_With_Uncertainty.csv",
    index=False
)
