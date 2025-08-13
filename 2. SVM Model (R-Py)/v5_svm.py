#!/usr/bin/env python3
"""
SVM Model Implementation - Python Version of v4_svm.R
Converted from R to Python using scikit-learn
"""

# 1. PACKAGES
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("Starting SVM Analysis...")

# 2. DATA PREPARATION
print("\n2. Loading and preparing data...")

# Load dataset
df = pd.read_excel("ml_dataset_v5_final.xlsx")

# Remove Document ID to prevent data leakage
X = df.drop(columns=['document'])

print(f"Dataset shape: {X.shape}")
print(f"Features: {list(X.columns)}")

# 3. TRAIN/TEST SPLIT
print("\n3. Creating train/test split...")

# Set seed for reproducibility
np.random.seed(123)

# Separate features and target
X_features = X.drop(columns=['category'])
y = X['category']

# Split data into training and test data (80/20)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=123, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# 4. SCALING DATA
print("\n4. Scaling data...")

def get_scaling_params(X_df):
    """Get scaling parameters for non-binary features"""
    params = {}
    for col in X_df.columns:
        vals = X_df[col].values
        unique_vals = np.unique(vals)
        
        # Skip binary columns
        if len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
            continue
            
        mean_val = np.mean(vals)
        std_val = np.std(vals, ddof=1)  # Use sample std like R
        
        if std_val > 0:  # Only store if not zero variance
            params[col] = {'mean': mean_val, 'std': std_val}
    
    return params

def apply_scaling(X_df, params):
    """Apply scaling using provided parameters"""
    X_scaled = X_df.copy()
    for col in params:
        if col in X_scaled.columns:
            mean_val = params[col]['mean']
            std_val = params[col]['std']
            X_scaled[col] = (X_scaled[col] - mean_val) / std_val
    return X_scaled

# Scale features (preserve binary features)
scaling_params = get_scaling_params(X_train)
X_train_scaled = apply_scaling(X_train, scaling_params)
X_test_scaled = apply_scaling(X_test, scaling_params)

print(f"Scaled {len(scaling_params)} non-binary features")

# 5. CLASS WEIGHTS AND IMBALANCED DATA
print("\n5. Analyzing class distribution...")

# Check class distribution
class_counts = Counter(Y_train)
print("Class distribution:")
for cls, count in sorted(class_counts.items()):
    print(f"  Class {cls}: {count} ({count/len(Y_train)*100:.1f}%)")

# Calculate class weights (inverse frequency)
classes = np.array(list(class_counts.keys()))
class_weight_values = compute_class_weight(
    'balanced', classes=classes, y=Y_train
)
class_weights = dict(zip(classes, class_weight_values))

print("Class weights:", class_weights)

# Visualize class distribution
plt.figure(figsize=(10, 6))
plt.bar(class_counts.keys(), class_counts.values())
plt.title('Class Distribution of Target')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# 6. BASELINE SVM MODEL (LINEAR) & TEST
print("\n6. Training baseline SVM model...")

# Train baseline SVM with linear kernel
svm_benchmark = SVC(
    kernel='linear',
    C=1,
    class_weight=class_weights,
    random_state=123
)

svm_benchmark.fit(X_train_scaled, Y_train)

# Make predictions
predictions = svm_benchmark.predict(X_test_scaled)

# Confusion matrix
conf_matrix = confusion_matrix(Y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate accuracy
accuracy = accuracy_score(Y_test, predictions)
print(f"Test Accuracy: {accuracy:.4f}")

# Calculate macro F1 score
def calc_macro_f1(y_true, y_pred):
    """Calculate macro F1 score manually to match R implementation"""
    classes = np.unique(y_true)
    f1_scores = []
    
    for cls in classes:
        # One vs rest for each class
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        
        if (tp + fp) == 0 or (tp + fn) == 0:
            f1_scores.append(0)
            continue
            
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if (precision + recall) == 0:
            f1_scores.append(0)
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)
    
    return np.mean(f1_scores)

macro_f1 = calc_macro_f1(Y_test, predictions)
print(f"Macro F1 Score: {macro_f1:.4f}")

# 7. STRATIFIED K-FOLD CROSS-VALIDATION
print("\n7. Performing cross-validation with hyperparameter tuning...")

# Set seed for reproducibility
np.random.seed(111)

# Define hyperparameters
kernels = ['linear', 'rbf']
cost_values = [0.1, 1, 10, 100]
gamma_values = [0.001, 0.01, 0.1, 1]

# Create stratified k-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=111)

# Store results
results_cv = []

print("Running cross-validation...")
fold_num = 1

for train_idx, val_idx in skf.split(X_train_scaled, Y_train):
    print(f"  Processing fold {fold_num}/5...")
    
    # Split data for this fold
    X_train_cv = X_train_scaled.iloc[train_idx]
    Y_train_cv = Y_train.iloc[train_idx]
    X_val_cv = X_train_scaled.iloc[val_idx]
    Y_val_cv = Y_train.iloc[val_idx]
    
    # Calculate class weights for this fold
    fold_classes = np.unique(Y_train_cv)
    fold_weights = compute_class_weight(
        'balanced', classes=fold_classes, y=Y_train_cv
    )
    fold_class_weights = dict(zip(fold_classes, fold_weights))
    
    # Test different hyperparameters
    for kernel in kernels:
        for cost in cost_values:
            if kernel == 'linear':
                # Linear kernel (no gamma parameter)
                svm_model = SVC(
                    kernel=kernel,
                    C=cost,
                    class_weight=fold_class_weights,
                    random_state=123
                )
                svm_model.fit(X_train_cv, Y_train_cv)
                preds = svm_model.predict(X_val_cv)
                macro_f1_val = calc_macro_f1(Y_val_cv, preds)
                
                results_cv.append({
                    'kernel': kernel,
                    'cost': cost,
                    'gamma': 'linear',
                    'fold': fold_num,
                    'macroF1': macro_f1_val
                })
                
            else:  # RBF kernel
                for gamma in gamma_values:
                    svm_model = SVC(
                        kernel=kernel,
                        C=cost,
                        gamma=gamma,
                        class_weight=fold_class_weights,
                        random_state=123
                    )
                    svm_model.fit(X_train_cv, Y_train_cv)
                    preds = svm_model.predict(X_val_cv)
                    macro_f1_val = calc_macro_f1(Y_val_cv, preds)
                    
                    results_cv.append({
                        'kernel': kernel,
                        'cost': cost,
                        'gamma': gamma,
                        'fold': fold_num,
                        'macroF1': macro_f1_val
                    })
    
    fold_num += 1

# Convert results to DataFrame
results_df = pd.DataFrame(results_cv)
print("\nCross-validation results:")
print(results_df.head(10))

# Aggregate results by hyperparameters
agg_results = results_df.groupby(['kernel', 'cost', 'gamma'])['macroF1'].mean().reset_index()
agg_results = agg_results.sort_values('macroF1', ascending=False)

print("\nAggregated results (top 10):")
print(agg_results.head(10))

# Find best parameters
best_params = agg_results.iloc[0]
print(f"\nBest parameters:")
print(f"  Kernel: {best_params['kernel']}")
print(f"  Cost (C): {best_params['cost']}")
print(f"  Gamma: {best_params['gamma']}")
print(f"  CV Macro F1: {best_params['macroF1']:.4f}")

# 8. FINAL SVM WITH BEST PARAMETERS FROM CV + TEST
print("\n8. Training final SVM with best parameters...")

# Set seed
np.random.seed(123)

# Prepare parameters for final model
best_kernel = best_params['kernel']
best_cost = best_params['cost']
best_gamma = best_params['gamma'] if best_params['gamma'] != 'linear' else 'scale'

# Train final SVM
if best_kernel == 'linear':
    final_svm = SVC(
        kernel=best_kernel,
        C=best_cost,
        class_weight=class_weights,
        random_state=123
    )
else:
    final_svm = SVC(
        kernel=best_kernel,
        C=best_cost,
        gamma=best_gamma,
        class_weight=class_weights,
        random_state=123
    )

final_svm.fit(X_train_scaled, Y_train)

# Make final predictions
final_predictions = final_svm.predict(X_test_scaled)

# Final confusion matrix
final_conf_matrix = confusion_matrix(Y_test, final_predictions)
print("Final Confusion Matrix:")
print(final_conf_matrix)

# Calculate final metrics
final_accuracy = accuracy_score(Y_test, final_predictions)
final_macro_f1 = calc_macro_f1(Y_test, final_predictions)

print(f"Final Test Accuracy: {final_accuracy:.4f}")
print(f"Final Macro F1 Score: {final_macro_f1:.4f}")

# Performance comparison
perf_comparison = pd.DataFrame({
    'Model': ['Benchmark SVM (Linear)', 'Tuned Final SVM'],
    'Accuracy': [accuracy, final_accuracy],
    'MacroF1': [macro_f1, final_macro_f1]
})

print("\nPerformance Comparison:")
print(perf_comparison)

# 9. VISUALIZE FINAL SVM MODEL: HEATMAP
print("\n9. Creating confusion matrix heatmaps...")

def plot_confusion_matrix_heatmap(cm, title, normalize=True):
    """Plot confusion matrix as heatmap"""
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.1%'
        cm_plot = cm_norm
    else:
        fmt = 'd'
        cm_plot = cm
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=sorted(np.unique(Y_test)),
                yticklabels=sorted(np.unique(Y_test)))
    plt.title(f'{title} ({"row-normalized" if normalize else "raw counts"})')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_heatmap.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

# Plot confusion matrices
plot_confusion_matrix_heatmap(conf_matrix, "Benchmark SVM Confusion Matrix")
plot_confusion_matrix_heatmap(final_conf_matrix, "Tuned SVM Confusion Matrix")

# Calculate and plot difference heatmap
def plot_difference_heatmap(cm1, cm2, title):
    """Plot difference between two normalized confusion matrices"""
    cm1_norm = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
    cm2_norm = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]
    diff = cm2_norm - cm1_norm
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(diff, annot=True, fmt='.1%', cmap='RdBu_r', center=0,
                xticklabels=sorted(np.unique(Y_test)),
                yticklabels=sorted(np.unique(Y_test)))
    plt.title(title)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_difference.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

plot_difference_heatmap(conf_matrix, final_conf_matrix, 
                       "Change in Row-normalized CM (Tuned − Benchmark)")

# 10. FEATURE IMPORTANCE ANALYSIS
print("\n10. Calculating feature importance...")

def simple_perm_importance(model, X, y, metric_func=calc_macro_f1, nsim=5):
    """Calculate permutation importance"""
    base_pred = model.predict(X)
    base_score = metric_func(y, base_pred)
    
    feature_names = X.columns
    scores = {}
    
    print("  Calculating permutation importance for each feature...")
    for j, col in enumerate(feature_names):
        if j % 10 == 0:
            print(f"    Processing feature {j+1}/{len(feature_names)}")
            
        drops = []
        for i in range(nsim):
            X_perm = X.copy()
            X_perm[col] = np.random.permutation(X_perm[col].values)
            pred_perm = model.predict(X_perm)
            drop = base_score - metric_func(y, pred_perm)
            drops.append(drop)
        
        scores[col] = np.mean(drops)
    
    # Sort by importance (descending)
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    return scores

# Calculate importance for both models
np.random.seed(123)
print("Calculating importance for benchmark model...")
imp_bench = simple_perm_importance(svm_benchmark, X_test_scaled, Y_test, nsim=5)

np.random.seed(123)
print("Calculating importance for final model...")
imp_tuned = simple_perm_importance(final_svm, X_test_scaled, Y_test, nsim=5)

def plot_feature_importance(imp_dict, title, top_n=15):
    """Plot feature importance"""
    # Get top N features
    top_features = list(imp_dict.keys())[:top_n]
    top_importance = [imp_dict[f] for f in top_features]
    
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_importance)
    plt.yticks(y_pos, top_features)
    plt.xlabel('Δ Macro F1')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

# Plot feature importance for both models
plot_feature_importance(imp_bench, "Top 15 Feature Importances - Benchmark SVM")
plot_feature_importance(imp_tuned, "Top 15 Feature Importances - Final SVM")

# Calculate and plot importance differences
print("\nCalculating feature importance differences...")
all_features = set(imp_bench.keys()) | set(imp_tuned.keys())
importance_diff = {}

for feature in all_features:
    bench_imp = imp_bench.get(feature, 0)
    tuned_imp = imp_tuned.get(feature, 0)
    importance_diff[feature] = tuned_imp - bench_imp

# Sort by difference (descending)
importance_diff = dict(sorted(importance_diff.items(), key=lambda x: x[1], reverse=True))

plot_feature_importance(importance_diff, "Top 15 Feature Importance Gains (Final − Benchmark)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"Final Model Performance:")
print(f"  Accuracy: {final_accuracy:.4f}")
print(f"  Macro F1: {final_macro_f1:.4f}")
print(f"  Best Parameters: {best_kernel} kernel, C={best_cost}, gamma={best_gamma}")
print(f"\nTop 5 Most Important Features:")
for i, (feature, importance) in enumerate(list(imp_tuned.items())[:5]):
    print(f"  {i+1}. {feature}: {importance:.4f}")