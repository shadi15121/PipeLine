import pandas as pd
import os
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans
from textblob import TextBlob

# === Load the dataset ===
# === add the file to work ===
df = pd.read_csv('data/PS_20174392719_1491204439457_log.csv')

# Preview first rows
print(df.head())

# DataFrame info
print(df.info())

# === Systems Stage: File Info ===
file_path = 'data/PS_20174392719_1491204439457_log.csv'

# File name
file_name = os.path.basename(file_path)
print(f"File Name: {file_name}")

# File size (MB)
file_size = os.path.getsize(file_path) / (1024 * 1024)
print(f"File Size: {file_size:.2f} MB")

# File type (extension)
file_type = os.path.splitext(file_path)[1]
print(f"File Type: {file_type}")


# === Stage 2: Meta Data ===

print("\n--- META DATA STAGE ---")

# 1️ Data size (rows, columns)
print(f"Data Shape (rows, columns): {df.shape}")

# 2️ Data types (already printed with info(), but here’s a clear version)
print("\nData Types:")
print(df.dtypes)

# 3️ Missing data
print("\nMissing Values Per Column:")
print(df.isnull().sum())

# 4️ Special values
# Example: check for negative amounts or balances (they shouldn’t exist here)
print("\nSpecial Values Check:")

# Amount < 0
neg_amounts = df[df['amount'] < 0]
print(f"Rows with negative amount: {len(neg_amounts)}")

# Any balance fields < 0
neg_balances = df[
    (df['oldbalanceOrg'] < 0) |
    (df['newbalanceOrig'] < 0) |
    (df['oldbalanceDest'] < 0) |
    (df['newbalanceDest'] < 0)
    ]
print(f"Rows with negative balances: {len(neg_balances)}")




# === Stage 3: DATA STATISTICS STAGE ===

print("\n--- DATA STATISTICS STAGE ---")

# 1️ Central Tendencies
print("\n=== Central Tendencies ===")
print(df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']].describe())

print("\nMode of transaction type:")
print(df['type'].mode())

# 2️ Correlation & Association
print("\n=== Correlation Matrix ===")
numeric_df = df.select_dtypes(include=['int64', 'float64'])
print(numeric_df.corr())

# 3️ Data Distribution
print("\n=== Value Counts for Transaction Type ===")
print(df['type'].value_counts())

print("\nFraudulent vs Non-Fraudulent Transactions:")
print(df['isFraud'].value_counts())

# 4️ Missing & Special Values — repeat to confirm
print("\n=== Re-check for Missing ===")
print(df.isnull().sum())

print("\nSpecial: Zero amount transactions:")
zero_amounts = df[df['amount'] == 0]
print(f"Rows with zero amount: {len(zero_amounts)}")

# 5⃣ Duplicates & Unique Values
print("\n=== Duplicate Rows ===")
print(f"Number of duplicate rows: {df.duplicated().sum()}")

print("\nUnique transaction types:")
print(df['type'].unique())

# 6⃣ Dimension Reduction (basic idea)
# For this stage: show numeric columns for PCA later
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("\n=== Numeric Columns for Possible Dimension Reduction ===")
print(numeric_cols)





# === Stage 4: Abnormality Detection ===

print("\n--- ABNORMALITY DETECTION STAGE ---")
print("\n MAD version")
# === 1⃣ Single Feature Outlier Detection (MAD) ===

# Calculate MAD for 'amount'
amount_median = df['amount'].median()
mad = np.median(np.abs(df['amount'] - amount_median))
threshold = 3 * mad

# Find outliers
mad_outliers = df[np.abs(df['amount'] - amount_median) > threshold]

print(f"Amount Median: {amount_median}")
print(f"MAD for amount: {mad}")
print(f"Number of amount outliers (MAD method): {len(mad_outliers)}")
# Apply IQR on 'amount'
Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

iqr_outliers = df[(df['amount'] < lower_bound) | (df['amount'] > upper_bound)]

print(f"\nIQR Method on 'amount':")
print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
print(f"Number of outliers detected by IQR: {len(iqr_outliers)}")


# === 1⃣c Z-Score (Mean ± k*STD) Outlier Detection ===

k = 3  # Number of standard deviations
mean_amount = df['amount'].mean()
std_amount = df['amount'].std()

zscore_outliers = df[(df['amount'] > mean_amount + k * std_amount) |
                     (df['amount'] < mean_amount - k * std_amount)]

print(f"\nZ-Score Method on 'amount':")
print(f"Mean: {mean_amount}, STD: {std_amount}")
print(f"Thresholds: {mean_amount - k * std_amount} to {mean_amount + k * std_amount}")
print(f"Number of outliers detected by Z-score method: {len(zscore_outliers)}")





# === 2⃣ Multi Feature Outlier Detection (Elliptic Envelope) ===

# Select numeric features — sample to keep it light
X = df[['amount', 'oldbalanceOrg']].sample(10000, random_state=42)

# Fit Elliptic Envelope
ee = EllipticEnvelope(contamination=0.01)  # 1% outliers assumed
ee.fit(X)

# Predict outliers
outlier_pred = ee.predict(X)  # -1 = outlier, 1 = inlier

# Extract outliers
ellipse_outliers = X[outlier_pred == -1]

print(f"Number of multi-feature outliers (Elliptic Envelope): {len(ellipse_outliers)}")




# === 3⃣ Multi Feature Outlier Detection (Isolation Forest) ===

from sklearn.ensemble import IsolationForest

print("\nIsolation Forest Outlier Detection on multi features...")

# Use the same numeric features sample
X_iso = df[['amount', 'oldbalanceOrg', 'newbalanceOrig']].sample(10000, random_state=42)

# Fit Isolation Forest
iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_preds = iso_forest.fit_predict(X_iso)  # -1 = outlier, 1 = inlier

# Extract outliers
iso_outliers = X_iso[iso_preds == -1]

print(f"Number of multi-feature outliers (Isolation Forest): {len(iso_outliers)}")




# === 4⃣ Multi Feature Outlier Detection (One-Class SVM) ===

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

print("\nOne-Class SVM Outlier Detection on multi features...")

# Sample and scale the data
X_svm = df[['amount', 'oldbalanceOrg', 'newbalanceOrig']].sample(5000, random_state=42)
scaler = StandardScaler()
X_svm_scaled = scaler.fit_transform(X_svm)

# Fit One-Class SVM
svm = OneClassSVM(nu=0.01, kernel="rbf", gamma="auto")  # nu = expected proportion of outliers
svm_preds = svm.fit_predict(X_svm_scaled)  # -1 = outlier, 1 = inlier

# Extract outliers
svm_outliers = X_svm[svm_preds == -1]

print(f"Number of multi-feature outliers (One-Class SVM): {len(svm_outliers)}")









# === Stage 5: clustering ===
print("\n--- CLUSTERING STAGE ---")

# === 1️ Grouping ===

# Use a small sample for clarity
X = df[['amount', 'oldbalanceOrg']].sample(5000, random_state=42)

# K-Means with 3 clusters (as an example)
kmeans = KMeans(n_clusters=3, random_state=42)
X['cluster'] = kmeans.fit_predict(X)

print(f"Cluster centers:\n{kmeans.cluster_centers_}")

# === 2️ Meaning of Groups ===
print("\nCluster sizes:")
print(X['cluster'].value_counts())

# === 3️ Do we have points outside? ===
# Calculate distance to cluster center
distances = kmeans.transform(X[['amount', 'oldbalanceOrg']])
min_distances = np.min(distances, axis=1)

# Define "outside" as being in the top 1% of distances in its cluster
threshold = np.percentile(min_distances, 99)
outside_points = X[min_distances > threshold]

print(f"Number of points far from cluster centers (possible cluster outliers): {len(outside_points)}")






# === Stage 6: Segment analysis ===
print("\n--- SEGMENT ANALYSIS STAGE ---")

# === 1️ Features: mean stats per cluster ===
# Use your X DataFrame with the clusters
feature_means = X.groupby('cluster').mean(numeric_only=True)
print("\nAverage features per cluster:")
print(feature_means)

# === 2️ Temporal: cluster size over time ===
# Join cluster labels back to full DataFrame by index (if you want)
# But here we show it on the sample
X['step'] = df.loc[X.index, 'step']  # Add step back

cluster_over_time = X.groupby(['cluster', 'step']).size().reset_index(name='count')
print("\nCluster size over time (sample):")
print(cluster_over_time.head())






# === Example only, my dataset does not have natural text ===
# === Stage 7: NLP ===
print("\n--- NLP STAGE (Demo Example) ---")

# Simulated small text sample
fake_texts = [
    "Customer reported issue with transfer delay.",
    "Payment completed successfully.",
    "Fraud alert: suspicious cash out detected."
]

# Simple sentiment check using TextBlob
from textblob import TextBlob

for text in fake_texts:
    blob = TextBlob(text)
    print(f"Text: '{text}' | Sentiment polarity: {blob.sentiment.polarity}")








# === Stage 8: Graph ===
print("\n--- GRAPHS STAGE ---")

import networkx as nx
import matplotlib.pyplot as plt

# Take a tiny sample to demo graph
graph_sample = df[['nameOrig', 'nameDest', 'amount']].sample(50, random_state=42)

# Create a directed graph: nodes = accounts, edges = transactions
G = nx.from_pandas_edgelist(graph_sample, 'nameOrig', 'nameDest', ['amount'], create_using=nx.DiGraph())

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Example: find node with highest degree (most connections)
degrees = G.degree()
highest_degree = max(degrees, key=lambda x: x[1])
print(f"Node with highest degree: {highest_degree}")

# Visualize tiny graph
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=300, font_size=8, arrows=True)
plt.title("Transaction Graph (Sample)")
plt.show()

# 2️ Account Hierarchy Tree — find top sender and plot its immediate connections
print("\n2) Account Hierarchy Tree")

# Find node with highest out-degree (sends the most)
out_degrees = G.out_degree()
top_sender = max(out_degrees, key=lambda x: x[1])[0]
print(f"Top sender node: {top_sender}")

# Build subgraph: top sender + direct receivers
receivers = list(G.successors(top_sender))
H = G.subgraph([top_sender] + receivers)

plt.figure(figsize=(6, 4))
pos = nx.spring_layout(H, seed=42)
nx.draw(H, pos, with_labels=True, node_size=500, node_color="lightblue", arrows=True)
plt.title(f"Hierarchy: {top_sender} → Receivers")
plt.show()




import seaborn as sns
import matplotlib.pyplot as plt

#⃣ Fraud Ratio by Transaction Type
print("\n1) Fraud Ratio by Transaction Type")
fraud_ratio = df.groupby('type').apply(
    lambda x: x['isFraud'].sum() / len(x)
).reset_index(name='fraud_ratio')

print(fraud_ratio)

plt.figure(figsize=(6, 4))
sns.barplot(data=fraud_ratio, x='type', y='fraud_ratio')
plt.title("Fraud Ratio by Transaction Type")
plt.ylabel("Fraud Ratio")
plt.show()

# 2️ Correlation Heatmap
print("\n2) Correlation Heatmap")
numeric_df = df.select_dtypes(include=['int64', 'float64'])
corr = numeric_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 3️ Transaction Amount Distribution
print("\n3) Transaction Amount Distribution")
plt.figure(figsize=(8, 4))
sns.histplot(df['amount'], bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.show()

# 4️ Cluster Scatter Plot
print("\n4) Cluster Scatter Plot")
# Re-use the clustering sample from before
sns.scatterplot(data=X, x='amount', y='oldbalanceOrg', hue='cluster', palette='Set1')
plt.title("Clusters: Amount vs Old Balance Origin")
plt.show()

# 5⃣ Degree Histogram of Transaction Graph
print("\n5) Degree Histogram of Graph")
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
plt.figure(figsize=(6, 4))
plt.hist(degree_sequence, bins=10)
plt.title("Network Degree Histogram")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.show()








# === Stage 9: Model ===
print("\n--- MODELS STAGE ---")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Use a small sample so it runs quickly
model_sample = df.sample(5000, random_state=42)

# Features and target
features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
X = model_sample[features]
y = model_sample['isFraud']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Coefficients for explainability
coeffs = dict(zip(features, model.coef_[0]))
print("\nFeature Coefficients (importance):")
for feat, coef in coeffs.items():
    print(f"{feat}: {coef:.4f}")



from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

print("\n--- GOODNESS OF FIT ---")

# ROC-AUC
probs = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, probs)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, probs)
plt.figure(figsize=(6, 4))
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# === LIFT ===
# Sort by predicted probability
df_lift = X_test.copy()
df_lift['y_true'] = y_test.values
df_lift['y_prob'] = probs
df_lift = df_lift.sort_values('y_prob', ascending=False)

# Divide into deciles
df_lift['decile'] = pd.qcut(df_lift['y_prob'], 10, duplicates='drop')

# Calculate lift per decile
lift_table = (
    df_lift.groupby('decile', observed=True)  # fixes the first warning
    .apply(
        lambda x: pd.Series({
            'n_obs': len(x),
            'fraud_rate': x['y_true'].mean()
        }),
        include_groups=False  # fixes the second warning
    )
    .reset_index()
)


# Baseline fraud rate
baseline = df_lift['y_true'].mean()
lift_table['lift'] = lift_table['fraud_rate'] / baseline

print("\nLIFT by decile:")
print(lift_table[['decile', 'fraud_rate', 'lift']])
