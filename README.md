# 🛡️ Cyber Security Attacks Prediction Analysis

This repository contains a data analysis and machine learning project focused on predicting the severity level of cyber security attacks using network traffic data.

## 🎯 Project Goal

The primary goal of this project is to build and evaluate a classification model that can predict the **Severity Level** (`Low`, `Medium`, or `High`) of a detected cyber attack based on various features of the network traffic and security alerts.

## 🗃️ Dataset

The dataset used in this analysis is the **`cyber-security-attacks`** dataset from Kaggle.

### Data Loading (Notebook Snippet)

The initial data download and loading steps performed in the notebook are:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Download and load the dataset
import kagglehub
path = kagglehub.dataset_download("teamincribo/cyber-security-attacks")
csa = pd.read_csv('/kaggle/input/cyber-security-attacks/cybersecurity_attacks.csv')
```

## 🧹 Data Preprocessing & Feature Engineering

The initial dataset contained 40,000 entries and 25 columns, many of which were dropped or handled for the current model.

### 1\. Column Dropping (Initial Pass)

The following non-predictive or redundant columns were removed:

  * `Timestamp`, `Source IP Address`, `Destination IP Address`, `Source Port`, `Destination Port`, `Payload Data`, `Network Segment`, `Geo-location Data`, `Proxy Information`, `Firewall Logs`, `IDS/IPS Alerts`.

### 2\. Handling Missing Values

Rows with missing values (`NaN`) in the remaining columns were dropped, reducing the dataset size significantly.

```python
csa = csa.dropna(axis = 0) # Reduced to 9953 entries
csa = csa.reset_index()
csa = csa.drop('index',axis =1)
```

### 3\. Further Column Dropping (Traffic Features)

Additional columns related to general traffic were removed, focusing the model on the security-specific and anomaly features.

  * `Protocol`, `Packet Length`, `Packet Type`, `Traffic Type`.

### 4\. Encoding the Target Variable (`Severity Level`)

The categorical target variable (`Severity Level`) was converted into a numerical format using **One-Hot Encoding** with `pd.get_dummies()`. The `Low` and `Medium` levels were encoded, and the `High` level was used as the baseline (represented when both `Low=0` and `Medium=0`).

```python
# Create dummy variables for Severity Level, dropping the first (High is the baseline)
sev = pd.get_dummies(csa['Severity Level'], dtype=int, drop_first=True)
ncsa = pd.concat([csa, sev], axis=1)
```

## ⚙️ Model Development

A binary classification model was developed to predict whether an attack is **Low Severity** (`y = 'Low'`).

### 1\. Defining Features and Target

The model uses only two features:

  * **Features (X):** `Anomaly Scores` and the one-hot encoded column `Medium`.
  * **Target (y):** The one-hot encoded column `Low`.

<!-- end list -->

```python
X = ncsa[['Anomaly Scores', 'Medium']]
y = ncsa['Low']
```

### 2\. Data Splitting

The data was split into training and testing sets (70% train, 30% test).

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
```

### 3\. Keras Neural Network (Model Attempt)

A small Sequential model was defined and trained using the Adam optimizer and Mean Absolute Error (`mae`) loss.

```python
# Model Architecture
input_shape = [2]
model = keras.Sequential([
    layers.Dense(9953, activation='relu', input_shape=input_shape),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.3),
    layers.Dense(1, activation='sigmoid')
])

# Compile and Train
model.compile(optimizer='adam', loss='mae')

early_stopping = EarlyStopping(min_delta=0.001, patience=5, restore_best_weights=True)

# Data type conversion for Keras
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)

history = model.fit(
    X_train, y_train,
    batch_size=1000,
    epochs=100,
    callbacks=[early_stopping] # Note: Early stopping warning in notebook due to missing 'val_loss'
)

# Training Result: Minimum Loss: 0.3224
```

### 4\. Decision Tree Model (Evaluation)

Separately, a **Decision Tree Classifier** was trained and evaluated to provide a baseline or alternative classification result for the same prediction task (`Low` vs. Not-`Low`).

```python
dtree = DecisionTreeClassifier(max_depth=3, random_state=42)
dtree.fit(X_train, y_train)
predictor = dtree.predict(X_test)

# Classification Report
print(classification_report(y_test, predictor))
```

#### Decision Tree Performance (`Low` vs. Not-`Low`)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0** (Not Low) | 0.68 | 0.95 | 0.79 | 1988 |
| **1** (Low) | 0.54 | 0.12 | 0.20 | 998 |
| **Accuracy** | | | **0.67** | 2986 |

**Interpretation:** The model is relatively good at identifying attacks that are **Not Low** (Class 0, with 95% recall), but performs poorly at identifying attacks that **are Low** (Class 1, with only 12% recall). This indicates a significant class imbalance or insufficient features to distinguish the "Low" category.

### 5\. Decision Tree Visualization

The trained Decision Tree was visualized with a maximum depth of 3 for interpretability.

```python
feature_names = X.columns
target_names = ncsa['Severity Level'] # Note: Using original text labels for context

plt.figure(figsize=(20, 10))
plot_tree(dtree,
          feature_names=feature_names,
          class_names=target_names, # Note: This will show all original severity levels, though the model predicts only 'Low' (1) or 'Not-Low' (0)
          filled=True,
          rounded=True,
          impurity=True,
          fontsize=10)
plt.title("Decision Tree Visualization", fontsize=16)
plt.show()
```

## 🛠️ Requirements

To run this analysis, you will need the following Python libraries:

```
pandas
numpy
matplotlib
seaborn
kagglehub
scikit-learn
tensorflow (or tensorflow-cpu/gpu)
keras
```
