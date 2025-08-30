import openml
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import FeatureAgglomeration
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest
from scipy.stats import spearmanr

# Load the Musk dataset from OpenML
dataset = openml.datasets.get_dataset(46615)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute
)

# Convert to numpy arrays if they are pandas objects
if isinstance(X, pd.DataFrame):
    X = X.to_numpy()
if isinstance(y, pd.Series):
    y = y.to_numpy()

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Target distribution: {np.bincount(y.astype(int))}")

# Function to get top K features based on random forest importance
def get_top_features_rf(X, y, k=10, excluded_indices=None):
    # Create a mask for features to exclude
    mask = np.ones(X.shape[1], dtype=bool)
    if excluded_indices is not None:
        mask[excluded_indices] = False
    
    # Use only non-excluded features
    X_filtered = X[:, mask]
    
    # Train model and get feature importances
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_filtered, y)
    importances = rf.feature_importances_
    
    # Get indices of top k features (in the filtered space)
    filtered_indices = np.argsort(importances)[::-1][:k]
    
    # Map back to original feature indices
    original_indices = np.where(mask)[0][filtered_indices]
    
    return original_indices

# Function to get top K features based on XGBoost importance
def get_top_features_xgb(X, y, k=10, excluded_indices=None):
    # Create a mask for features to exclude
    mask = np.ones(X.shape[1], dtype=bool)
    if excluded_indices is not None:
        mask[excluded_indices] = False
    
    # Use only non-excluded features
    X_filtered = X[:, mask]
    
    # Train model and get feature importances
    xgb = XGBClassifier(n_estimators=100, random_state=42)
    xgb.fit(X_filtered, y)
    importances = xgb.feature_importances_
    
    # Get indices of top k features (in the filtered space)
    filtered_indices = np.argsort(importances)[::-1][:k]
    
    # Map back to original feature indices
    original_indices = np.where(mask)[0][filtered_indices]
    
    return original_indices

# Function to get top K features based on logistic regression coefficients
def get_top_features_lr(X, y, k=10, excluded_indices=None):
    # Create a mask for features to exclude
    mask = np.ones(X.shape[1], dtype=bool)
    if excluded_indices is not None:
        mask[excluded_indices] = False
    
    # Use only non-excluded features
    X_filtered = X[:, mask]
    
    # Train model and get feature importances
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_filtered, y)
    importances = np.abs(lr.coef_[0])
    
    # Get indices of top k features (in the filtered space)
    filtered_indices = np.argsort(importances)[::-1][:k]
    
    # Map back to original feature indices
    original_indices = np.where(mask)[0][filtered_indices]
    
    return original_indices

# Function to get top K features based on Feature Agglomeration
def get_top_features_fa(X, k=10, excluded_indices=None):
    # Create a mask for features to exclude
    mask = np.ones(X.shape[1], dtype=bool)
    if excluded_indices is not None:
        mask[excluded_indices] = False
    
    # Use only non-excluded features
    X_filtered = X[:, mask]
    
    # If we need more features than available, return all available
    if k >= X_filtered.shape[1]:
        filtered_indices = np.arange(X_filtered.shape[1])
        original_indices = np.where(mask)[0][filtered_indices]
        return original_indices[:k]
    
    # Set n_clusters to a reasonable value (we'll use 20% of features)
    n_clusters = max(1, int(X_filtered.shape[1] * 0.2))
    
    # Create and fit feature agglomeration (without random_state)
    fa = FeatureAgglomeration(n_clusters=n_clusters)
    fa.fit(X_filtered)
    
    # Compute cluster centers
    cluster_centers = np.zeros((n_clusters, X_filtered.shape[0]))
    for i in range(n_clusters):
        features_in_cluster = np.where(fa.labels_ == i)[0]
        if len(features_in_cluster) > 0:  # Avoid empty clusters
            cluster_centers[i] = np.mean(X_filtered[:, features_in_cluster], axis=1)
    
    # Calculate correlation with cluster center and variance for each feature
    feature_scores = np.zeros(X_filtered.shape[1])
    variances = np.var(X_filtered, axis=0)
    
    for i in range(X_filtered.shape[1]):
        feature_cluster = fa.labels_[i]
        feature_data = X_filtered[:, i]
        cluster_center = cluster_centers[feature_cluster]
        
        # Calculate correlation with cluster center
        correlation = np.abs(np.corrcoef(feature_data, cluster_center)[0, 1])
        if np.isnan(correlation):  # Handle case where correlation is NaN
            correlation = 0
            
        # Combine correlation and variance (normalized)
        variance_norm = variances[i] / np.max(variances) if np.max(variances) > 0 else 0
        feature_scores[i] = correlation * 0.5 + variance_norm * 0.5
    
    # Select top k features based on combined score
    filtered_indices = np.argsort(feature_scores)[::-1][:k]
    
    # Map back to original feature indices
    original_indices = np.where(mask)[0][filtered_indices]
    
    return original_indices

# Function to get top K features based on highly variable gene selection
def get_top_features_hvgs(X, k=10, excluded_indices=None):
    # Create a mask for features to exclude
    mask = np.ones(X.shape[1], dtype=bool)
    if excluded_indices is not None:
        mask[excluded_indices] = False
    
    # Use only non-excluded features
    X_filtered = X[:, mask]
    
    variances = np.var(X_filtered, axis=0)
    filtered_indices = np.argsort(variances)[::-1][:k]
    
    # Map back to original feature indices
    original_indices = np.where(mask)[0][filtered_indices]
    
    return original_indices

# Function to get top K features based on Spearman correlation with target
def get_top_features_spearman(X, y, k=10, excluded_indices=None):
    # Create a mask for features to exclude
    mask = np.ones(X.shape[1], dtype=bool)
    if excluded_indices is not None:
        mask[excluded_indices] = False
    
    # Use only non-excluded features
    X_filtered = X[:, mask]
    
    correlations = []
    
    for i in range(X_filtered.shape[1]):
        corr, _ = spearmanr(X_filtered[:, i], y)
        correlations.append(abs(corr))
    
    filtered_indices = np.argsort(correlations)[::-1][:k]
    
    # Map back to original feature indices
    original_indices = np.where(mask)[0][filtered_indices]
    
    return original_indices

# Function to evaluate model with cross-validation
def evaluate_model(X, y, feature_indices, model_type='rf'):
    X_selected = X[:, feature_indices]
    
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'xgb':
        model = XGBClassifier(n_estimators=100, random_state=42)
    elif model_type == 'lr':
        model = LogisticRegression(max_iter=1000, random_state=42)
    
    scores = cross_val_score(model, X_selected, y, cv=5, scoring='accuracy')
    return np.mean(scores)

print("\nSelecting top features and evaluating models...")

# 1. Random Forest feature selection and evaluation
print("\nRandom Forest Feature Selection:")
rf_top10_indices = get_top_features_rf(X, y, k=10)
rf_top10_features = [attribute_names[i] for i in rf_top10_indices]
rf_top10_accuracy = evaluate_model(X, y, rf_top10_indices, 'rf')

rf_top1_index = get_top_features_rf(X, y, k=1)[0]
rf_top1_feature = attribute_names[rf_top1_index]
rf_top1_accuracy = evaluate_model(X, y, [rf_top1_index], 'rf')

# Get top 9 features from dataset with top 1 feature removed
rf_top9_indices = get_top_features_rf(X, y, k=9, excluded_indices=[rf_top1_index])
rf_top9_features = [attribute_names[i] for i in rf_top9_indices]
rf_top9_accuracy = evaluate_model(X, y, rf_top9_indices, 'rf')

print(f"Top 10 features: {rf_top10_features}")
print(f"Top 10 accuracy: {rf_top10_accuracy:.4f}")
print(f"Top 1 feature: {rf_top1_feature}")
print(f"Top 1 accuracy: {rf_top1_accuracy:.4f}")
print(f"Top 9 features (from reduced dataset): {rf_top9_features}")
print(f"Top 9 accuracy: {rf_top9_accuracy:.4f}")

# 2. XGBoost feature selection and evaluation
print("\nXGBoost Feature Selection:")
xgb_top10_indices = get_top_features_xgb(X, y, k=10)
xgb_top10_features = [attribute_names[i] for i in xgb_top10_indices]
xgb_top10_accuracy = evaluate_model(X, y, xgb_top10_indices, 'xgb')

xgb_top1_index = get_top_features_xgb(X, y, k=1)[0]
xgb_top1_feature = attribute_names[xgb_top1_index]
xgb_top1_accuracy = evaluate_model(X, y, [xgb_top1_index], 'xgb')

# Get top 9 features from dataset with top 1 feature removed
xgb_top9_indices = get_top_features_xgb(X, y, k=9, excluded_indices=[xgb_top1_index])
xgb_top9_features = [attribute_names[i] for i in xgb_top9_indices]
xgb_top9_accuracy = evaluate_model(X, y, xgb_top9_indices, 'xgb')

print(f"Top 10 features: {xgb_top10_features}")
print(f"Top 10 accuracy: {xgb_top10_accuracy:.4f}")
print(f"Top 1 feature: {xgb_top1_feature}")
print(f"Top 1 accuracy: {xgb_top1_accuracy:.4f}")
print(f"Top 9 features (from reduced dataset): {xgb_top9_features}")
print(f"Top 9 accuracy: {xgb_top9_accuracy:.4f}")

# 3. Logistic Regression feature selection and evaluation
print("\nLogistic Regression Feature Selection:")
lr_top10_indices = get_top_features_lr(X, y, k=10)
lr_top10_features = [attribute_names[i] for i in lr_top10_indices]
lr_top10_accuracy = evaluate_model(X, y, lr_top10_indices, 'lr')

lr_top1_index = get_top_features_lr(X, y, k=1)[0]
lr_top1_feature = attribute_names[lr_top1_index]
lr_top1_accuracy = evaluate_model(X, y, [lr_top1_index], 'lr')

# Get top 9 features from dataset with top 1 feature removed
lr_top9_indices = get_top_features_lr(X, y, k=9, excluded_indices=[lr_top1_index])
lr_top9_features = [attribute_names[i] for i in lr_top9_indices]
lr_top9_accuracy = evaluate_model(X, y, lr_top9_indices, 'lr')

print(f"Top 10 features: {lr_top10_features}")
print(f"Top 10 accuracy: {lr_top10_accuracy:.4f}")
print(f"Top 1 feature: {lr_top1_feature}")
print(f"Top 1 accuracy: {lr_top1_accuracy:.4f}")
print(f"Top 9 features (from reduced dataset): {lr_top9_features}")
print(f"Top 9 accuracy: {lr_top9_accuracy:.4f}")

# 4. Feature Agglomeration feature selection and evaluation (with RF)
print("\nFeature Agglomeration Selection:")
fa_top10_indices = get_top_features_fa(X, k=10)
fa_top10_features = [attribute_names[i] for i in fa_top10_indices]
fa_top10_accuracy = evaluate_model(X, y, fa_top10_indices, 'rf')

fa_top1_index = get_top_features_fa(X, k=1)[0]
fa_top1_feature = attribute_names[fa_top1_index]
fa_top1_accuracy = evaluate_model(X, y, [fa_top1_index], 'rf')

# Get top 9 features from dataset with top 1 feature removed
fa_top9_indices = get_top_features_fa(X, k=9, excluded_indices=[fa_top1_index])
fa_top9_features = [attribute_names[i] for i in fa_top9_indices]
fa_top9_accuracy = evaluate_model(X, y, fa_top9_indices, 'rf')

print(f"Top 10 features: {fa_top10_features}")
print(f"Top 10 accuracy: {fa_top10_accuracy:.4f}")
print(f"Top 1 feature: {fa_top1_feature}")
print(f"Top 1 accuracy: {fa_top1_accuracy:.4f}")
print(f"Top 9 features (from reduced dataset): {fa_top9_features}")
print(f"Top 9 accuracy: {fa_top9_accuracy:.4f}")

# 5. Highly Variable Gene Selection and evaluation (with RF)
print("\nHighly Variable Gene Selection:")
hvgs_top10_indices = get_top_features_hvgs(X, k=10)
hvgs_top10_features = [attribute_names[i] for i in hvgs_top10_indices]
hvgs_top10_accuracy = evaluate_model(X, y, hvgs_top10_indices, 'rf')

hvgs_top1_index = get_top_features_hvgs(X, k=1)[0]
hvgs_top1_feature = attribute_names[hvgs_top1_index]
hvgs_top1_accuracy = evaluate_model(X, y, [hvgs_top1_index], 'rf')

# Get top 9 features from dataset with top 1 feature removed
hvgs_top9_indices = get_top_features_hvgs(X, k=9, excluded_indices=[hvgs_top1_index])
hvgs_top9_features = [attribute_names[i] for i in hvgs_top9_indices]
hvgs_top9_accuracy = evaluate_model(X, y, hvgs_top9_indices, 'rf')

print(f"Top 10 features: {hvgs_top10_features}")
print(f"Top 10 accuracy: {hvgs_top10_accuracy:.4f}")
print(f"Top 1 feature: {hvgs_top1_feature}")
print(f"Top 1 accuracy: {hvgs_top1_accuracy:.4f}")
print(f"Top 9 features (from reduced dataset): {hvgs_top9_features}")
print(f"Top 9 accuracy: {hvgs_top9_accuracy:.4f}")

# 6. Spearman Correlation feature selection and evaluation (with RF)
print("\nSpearman Correlation Selection:")
spearman_top10_indices = get_top_features_spearman(X, y, k=10)
spearman_top10_features = [attribute_names[i] for i in spearman_top10_indices]
spearman_top10_accuracy = evaluate_model(X, y, spearman_top10_indices, 'rf')

spearman_top1_index = get_top_features_spearman(X, y, k=1)[0]
spearman_top1_feature = attribute_names[spearman_top1_index]
spearman_top1_accuracy = evaluate_model(X, y, [spearman_top1_index], 'rf')

# Get top 9 features from dataset with top 1 feature removed
spearman_top9_indices = get_top_features_spearman(X, y, k=9, excluded_indices=[spearman_top1_index])
spearman_top9_features = [attribute_names[i] for i in spearman_top9_indices]
spearman_top9_accuracy = evaluate_model(X, y, spearman_top9_indices, 'rf')

print(f"Top 10 features: {spearman_top10_features}")
print(f"Top 10 accuracy: {spearman_top10_accuracy:.4f}")
print(f"Top 1 feature: {spearman_top1_feature}")
print(f"Top 1 accuracy: {spearman_top1_accuracy:.4f}")
print(f"Top 9 features (from reduced dataset): {spearman_top9_features}")
print(f"Top 9 accuracy: {spearman_top9_accuracy:.4f}")
