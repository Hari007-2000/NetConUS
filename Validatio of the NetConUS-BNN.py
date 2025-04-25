#!/usr/bin/env python
# coding: utf-8

# This notebook aims to validate the stream classification model using a Bayesian Neural Network (BNN) by quantifying epistemic uncertainty from limited network property information. 
# 
# BNNs provide probabilistic predictions by learning distributions over model parameters. This helps assess confidence in classifying stream types based on structural connectivity.
# 
# We begin by importing essential libraries for data processing, model building, and uncertainty analysis.
# 
# **Import the necessary libraries for the BNN model**
# 

# In[ ]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pyro
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# In[5]:


pip install imbalanced-learn


# In[6]:


pip install torch


# In[5]:


pip install torchsummary


# The data used here is typically the network properties for a given network graph

# In[1]:


import pandas as pd

data = pd.read_csv('./HUC4 network properties.csv')  # Replace with your actual file path 


# In[2]:


data


# This code below is for the BNN model development using Pyro
# 
# This code defines a Bayesian Neural Network (BNN) using Pyro to classify stream types based on five network centrality features. 
# 
# It preprocesses the data using standardization and label encoding, then defines a three-layer neural network where the weights and biases are treated as probabilistic distributions

# In[12]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pyro.nn import PyroModule, PyroSample
from pyro.infer import MCMC, NUTS, Predictive


data = pd.read_csv('./HUC4_network_properties.csv')  # Replace with your actual file path 

feature_columns = ['Degree Centrality', 'Betweenness Centrality', 'Closeness Centrality', 'Clustering Coefficient','Eigenvector Centrality']  # Replace with actual feature column names
label_column = 'Stream Type Final'  # Replace with the actual label column name

# Extract features and labels
X = data[feature_columns].values  # Features
y = data[label_column].values  # Labels

# Encode string labels as integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float)
X_test = torch.tensor(X_test, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Bayesian neural network
class BNN(PyroModule):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(BNN, self).__init__()
        self.fc1 = PyroModule[nn.Linear](input_dim, hidden_dim1)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim1, input_dim]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim1]).to_event(1))

        self.fc2 = PyroModule[nn.Linear](hidden_dim1, hidden_dim2)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim2, hidden_dim1]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim2]).to_event(1))

        self.out = PyroModule[nn.Linear](hidden_dim2, output_dim)
        self.out.weight = PyroSample(dist.Normal(0., 1.).expand([output_dim, hidden_dim2]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0., 1.).expand([output_dim]).to_event(1))

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, y=None):
        x = self.tanh(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.out(x)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits

# Instantiate the model
input_dim = X_train.shape[1]
hidden_dim1 = 16
hidden_dim2 = 8
output_dim = len(np.unique(y))

bnn = BNN(input_dim, hidden_dim1, hidden_dim2, output_dim)

# Define the NUTS sampler
nuts_kernel = NUTS(bnn)


# The code below sets up a No-U-Turn Sampler (NUTS), a type of Markov Chain Monte Carlo (MCMC) algorithm, to infer the posterior distributions of the model parameters.

# In[13]:


# Run MCMC to sample from the posterior
mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=200)
mcmc.run(X_train, y_train)


# This code below uses the posterior samples from the trained Bayesian Neural Network to generate predictive distributions for the test data and computes the average logits to determine class predictions.
# 
# It evaluates model performance using accuracy, a confusion matrix heatmap, and a detailed classification report. 
# 
# The results reflect the model’s predictive capability and help assess classification quality across stream types.

# In[14]:


#  predictive distribution to make predictions
predictive = Predictive(bnn, posterior_samples=mcmc.get_samples(), return_sites=["_RETURN"])
predictions = predictive(X_test)

# Extract logits
logits = predictions["_RETURN"]

# Compute mean of logits over posterior samples
mean_logits = logits.mean(dim=0)

# Compute predicted classes
predicted_classes = mean_logits.argmax(dim=1)

# Evaluate the model
print("Test Accuracy:", accuracy_score(y_test, predicted_classes))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("BNN_Confusion_matrix.png", dpi =1000, bbox_inches="tight", pad_inches=0.2)
plt.show()

# Classification report
class_report = classification_report(y_test, predicted_classes, target_names=label_encoder.classes_)
print("Classification Report:\n", class_report)


# In[20]:


# Define new class labels
new_labels = ["Connector Streams", "Cluster Streams", "Central Streams", "Convergent Streams", "Peripheral Streams"]

# Plot the confusion matrix with new labels
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=new_labels, yticklabels=new_labels)
plt.xlabel('Predicted', fontsize =15)
plt.ylabel('Actual', fontsize= 15)
plt.xticks(fontsize=15)  # Increase x-axis labels (Cluster names)
plt.yticks(fontsize=15)  # Increase y-axis labels (Network properties)

plt.title('Confusion Matrix')
plt.savefig("BNN_Confusion_matrix_renamed.png", dpi=1000, bbox_inches="tight", pad_inches=0.2)
plt.show()


# In[33]:


import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Extract logits from posterior samples (already done in your code)
logits = predictions["_RETURN"]

# Compute mean and variance of logits over posterior samples
mean_logits = logits.mean(dim=0).detach().numpy()  # Mean logits for each test sample
var_logits = logits.var(dim=0).detach().numpy()    # Variance (uncertainty) in logits for each test sample

# Compute predicted classes using mean logits
predicted_classes = mean_logits.argmax(axis=1)

# Prepare a DataFrame for plotting
plot_data = pd.DataFrame({
    'Mean Logits': mean_logits.max(axis=1),  # Take the max logit as the predicted class for simplicity
    'Variance Logits': var_logits.max(axis=1),  # Take the variance of the max logit
    'True Class': label_encoder.inverse_transform(y_test.numpy()),  # Actual class names
    'Predicted Class': label_encoder.inverse_transform(predicted_classes)  # Predicted class names
})

# Box Plot for predicted uncertainty with improved formatting
plt.figure(figsize=(8, 6))
sns.boxplot(x='True Class', y='Mean Logits', data=plot_data)

# Apply new xtick labels (edit as needed)
new_labels = ["Central Streams", "Peripheral Streams", "Connector Streams", "Cluster Streams", "Convergent Streams"]
plt.xticks(ticks=range(len(new_labels)), labels=new_labels, fontsize=14, rotation=45, ha='right')


# Improve Title and Labels
plt.title('Box Plot of Predicted Mean Logits (Uncertainty) by True Class', fontsize=18, fontweight='bold')
plt.xlabel('True Class', fontsize=16, fontweight='bold')
plt.ylabel('Mean Logits', fontsize=16, fontweight='bold')

# Improve Axis Labels
plt.xticks(fontsize=14, rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.yticks(fontsize=14)

# Add Grid for Better Visualization
plt.grid(True, linestyle="--", alpha=0.6)

plt.savefig("Uncertainity.png", dpi = 1000, bbox_inches="tight")
# Show Plot
plt.show()


# This code computes the **Receiver Operating Characteristic (ROC) curves** and **Area Under the Curve (AUC)** scores for each class in a multi-class Bayesian classification task:
# 
# 1. It iterates through each class to calculate the **false positive rate (FPR)** and **true positive rate (TPR)** using the predicted probabilities (`mean_logits`) and true binary labels (`y_test_binarized`).
# 2. It then **plots individual ROC curves** for each class using a different color and includes the AUC in the legend for visual performance comparison.
# 3. The plot helps assess the trade-off between sensitivity and specificity for each stream class and evaluates the **discriminative power** of the BNN classifier.

# In[46]:


# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(output_dim):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], mean_logits[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure(figsize=(10, 7))
colors = cycle(["blue", "orange", "green", "red", "purple", "brown"])

for i, color in zip(range(output_dim), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve for class {i} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel("False Positive Rate", fontsize = 15)
plt.ylabel("True Positive Rate", fontsize = 15)
plt.title("ROC curve for multi-class classification")
plt.legend(loc="lower right")
plt.savefig("ROCcurve.png", dpi =1000)
plt.show()


# This code generates **Precision-Recall (PR) curves** (sometimes referred to as **Cumulative Gains charts**) for each class in a multi-class classification problem using a Bayesian Neural Network:
# 
# 1. It computes the **precision-recall curve** for each class based on the predicted probabilities (`mean_logits`) and true binary labels (`y_test_binarized`).
# 2. It plots **precision vs. recall** for all classes in a single figure, allowing performance comparison between classes, especially in **imbalanced datasets**.
# 3. These curves are useful for assessing the classifier’s ability to retrieve relevant stream class instances with minimal false positives.

# In[43]:


from sklearn.metrics import precision_recall_curve

# Plot cumulative gains chart for each class
plt.figure(figsize=(10, 7))
for i, color in zip(range(output_dim), colors):
    precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], mean_logits[:, i])
    plt.plot(recall, precision, lw=2, color=color, label=f'Class {i}')
    
plt.xlabel('Recall',fontsize = 15)
plt.ylabel('Precision', fontsize = 15)
plt.title('Cumulative Gains Chart for Multi-Class Classification')
plt.legend(loc='lower right')
plt.show()


# In[44]:


from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_curve, roc_auc_score
from itertools import cycle

# Binarize the output labels for multi-class classification
y_test_binarized = label_binarize(y_test, classes=[i for i in range(output_dim)])

# Compute precision-recall curve and AUC for each class
precision = dict()
recall = dict()
average_precision = dict()

for i in range(output_dim):
    precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], mean_logits[:, i])
    average_precision[i] = auc(recall[i], precision[i])

# Plot Precision-Recall curve for each class
plt.figure(figsize=(10, 7))
colors = cycle(["blue", "orange", "green", "red", "purple", "brown"])

for i, color in zip(range(output_dim), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label=f'Precision-Recall curve for class {i} (area = {average_precision[i]:0.2f})')

plt.xlabel('Recall',fontsize = 15)
plt.ylabel('Precision', fontsize = 15)
plt.title("Precision-Recall curve for multi-class classification")
#plt.legend(loc="lower left")
plt.savefig("Precision-Recall curve.png", dpi = 600)
plt.show()


# In[12]:


# Use the predictive distribution to make predictions
predictive = Predictive(bnn, posterior_samples=mcmc.get_samples(), return_sites=["_RETURN"])
predictions = predictive(X_test)

# Extract logits
logits = predictions["_RETURN"]

# Compute mean and standard deviation of logits over posterior samples
mean_logits = logits.mean(dim=0)
std_logits = logits.std(dim=0)

# Compute predicted classes
predicted_classes = mean_logits.argmax(dim=1)

# Evaluate the model
test_accuracy = accuracy_score(y_test, predicted_classes)
print("Test Accuracy:", test_accuracy)

# Prepare data for saving to CSV
results = pd.DataFrame(X_test.numpy(), columns=feature_columns)
results['true_label'] = y_test.numpy()
results['predicted_label'] = predicted_classes.numpy()
results['uncertainty'] = std_logits.max(dim=1).values.numpy()

# Save the results to a CSV file
results.to_csv('./predictions_with_uncertainty.csv', index=False)


# In[27]:


pip install umap-learn


# This code performs **uncertainty visualization** of the Bayesian Neural Network's predictions using **UMAP (Uniform Manifold Approximation and Projection)** for dimensionality reduction:
# 
# 1. It prepares a `DataFrame` containing test features, predicted and true labels, and the model's **epistemic uncertainty** (approximated by the max standard deviation of logits).
# 2. It applies **UMAP** to project high-dimensional test features into **2D space**, preserving local structure.
# 3. The scatter plot visualizes uncertainty across the UMAP-reduced space, where **color intensity represents prediction uncertainty**—useful to identify regions with low confidence and possible misclassifications in stream class prediction.

# In[31]:


# Prepare data for plotting
results = pd.DataFrame(X_test.numpy(), columns=feature_columns)
results['true_label'] = y_test.numpy()
results['predicted_label'] = predicted_classes.numpy()
results['uncertainty'] = std_logits.max(dim=1).values.numpy()

# Apply UMAP to reduce dimensionality to 2D
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(X_test)

# Add UMAP components to the results DataFrame
results['UMAP1'] = embedding[:, 0]
results['UMAP2'] = embedding[:, 1]

# Plot the results in the UMAP-reduced space
plt.figure(figsize=(12, 8))
scatter = plt.scatter(results['UMAP1'], results['UMAP2'], 
                      c=results['uncertainty'], s=50, cmap='Spectral', 
                      alpha=0.6, edgecolors='w', linewidth=0.5)


plt.colorbar(scatter, label='Uncertainty')

# Add labels and title
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title('UMAP Projection: Uncertainty Visualization')
plt.show()


# In[ ]:




