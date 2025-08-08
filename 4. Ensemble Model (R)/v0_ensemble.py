#!/usr/bin/env python3
"""
XGBoost Email Classifier for German Banking Emails
Handles 12-class classification with significant class imbalance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_recall_fscore_support
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class XGBoostEmailClassifier:
    """XGBoost classifier for German banking email categorization"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.model = None
        self.best_params = None
        self.cv_results = None
        
    def load_data(self, filepath='ml_dataset_v5_full.xlsx'):
        """Load and prepare data from Excel file"""
        print(f"Loading data from {filepath}...")
        self.df = pd.read_excel(filepath)
        
        # Separate features and target
        self.X = self.df.drop(['document', 'category'], axis=1)
        self.y = self.df['category']
        
        # Encode categories
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        self.n_classes = len(np.unique(self.y_encoded))
        
        print(f"Dataset shape: {self.X.shape}")
        print(f"Number of classes: {self.n_classes}")
        print(f"Feature types: {self.X.shape[1]} features")
        
        # Class distribution analysis
        self._analyze_class_distribution()
        
    def _analyze_class_distribution(self):
        """Analyze and visualize class distribution"""
        class_counts = pd.Series(self.y).value_counts()
        
        print("\nClass Distribution:")
        for cat, count in class_counts.items():
            print(f"{cat}: {count} ({count/len(self.y)*100:.1f}%)")
        
        # Imbalance metrics
        max_class = class_counts.max()
        min_class = class_counts.min()
        imbalance_ratio = max_class / min_class
        
        print(f"\nImbalance Ratio: {imbalance_ratio:.2f}")
        print(f"Largest class: {max_class} samples")
        print(f"Smallest class: {min_class} samples")
        
    def split_data(self, test_size=0.2, val_size=0.15):
        """Create stratified train/validation/test splits"""
        # First split: train+val vs test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y_encoded,
            test_size=test_size,
            stratify=self.y_encoded,
            random_state=self.random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=self.random_state
        )
        
        print(f"\nData splits:")
        print(f"Train: {len(self.X_train)} samples ({len(self.X_train)/len(self.X)*100:.1f}%)")
        print(f"Validation: {len(self.X_val)} samples ({len(self.X_val)/len(self.X)*100:.1f}%)")
        print(f"Test: {len(self.X_test)} samples ({len(self.X_test)/len(self.X)*100:.1f}%)")
        
        # Calculate sample weights for class imbalance
        self.sample_weights = compute_sample_weight('balanced', self.y_train)
        
    def hyperparameter_tuning(self, cv_folds=5):
        """Perform hyperparameter tuning with cross-validation"""
        print("\nStarting hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [50, 100, 200],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2]
        }
        
        # Base model
        base_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=self.n_classes,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Grid search with sample weights
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=skf,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit with sample weights
        grid_search.fit(self.X_train, self.y_train, sample_weight=self.sample_weights)
        
        self.best_params = grid_search.best_params_
        self.cv_results = grid_search.cv_results_
        
        print(f"\nBest parameters found:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print(f"\nBest CV Macro-F1: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_
    
    def train_model(self, use_tuned_params=True):
        """Train XGBoost model with early stopping"""
        print("\nTraining XGBoost model...")
        
        if use_tuned_params and self.best_params:
            params = self.best_params.copy()
        else:
            # Default parameters
            params = {
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0
            }
        
        # Add fixed parameters
        params.update({
            'objective': 'multi:softprob',
            'num_class': self.n_classes,
            'random_state': self.random_state,
            'use_label_encoder': False,
            'eval_metric': 'mlogloss'
        })
        
        # Create model
        self.model = xgb.XGBClassifier(**params)
        
        # Train with early stopping
        self.model.fit(
            self.X_train, self.y_train,
            sample_weight=self.sample_weights,
            eval_set=[(self.X_val, self.y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        print(f"Training completed. Best iteration: {self.model.best_iteration}")
        
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("\nEvaluating model performance...")
        
        # Predictions
        y_pred_train = self.model.predict(self.X_train)
        y_pred_val = self.model.predict(self.X_val)
        y_pred_test = self.model.predict(self.X_test)
        
        # Metrics for each set
        results = {}
        for name, y_true, y_pred in [
            ('Train', self.y_train, y_pred_train),
            ('Validation', self.y_val, y_pred_val),
            ('Test', self.y_test, y_pred_test)
        ]:
            accuracy = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            weighted_f1 = f1_score(y_true, y_pred, average='weighted')
            
            results[name] = {
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'weighted_f1': weighted_f1,
                'y_true': y_true,
                'y_pred': y_pred
            }
            
            print(f"\n{name} Set Performance:")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Macro F1: {macro_f1:.3f}")
            print(f"  Weighted F1: {weighted_f1:.3f}")
        
        # Detailed test set report
        print("\nDetailed Test Set Classification Report:")
        print(classification_report(
            self.y_test, y_pred_test,
            target_names=self.label_encoder.classes_,
            digits=3
        ))
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, top_n=20):
        """Plot top N most important features"""
        importance = self.model.feature_importances_
        feature_names = self.X.columns
        
        # Create dataframe and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def cross_validate(self, cv_folds=5):
        """Perform detailed cross-validation analysis"""
        print(f"\nPerforming {cv_folds}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = {
            'accuracy': [],
            'macro_f1': [],
            'weighted_f1': [],
            'per_class_f1': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            # Split data
            X_fold_train = self.X_train.iloc[train_idx]
            X_fold_val = self.X_train.iloc[val_idx]
            y_fold_train = self.y_train[train_idx]
            y_fold_val = self.y_train[val_idx]
            
            # Calculate sample weights for this fold
            fold_weights = compute_sample_weight('balanced', y_fold_train)
            
            # Train model
            fold_model = xgb.XGBClassifier(**self.model.get_params())
            fold_model.fit(
                X_fold_train, y_fold_train,
                sample_weight=fold_weights,
                eval_set=[(X_fold_val, y_fold_val)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            # Evaluate
            y_pred = fold_model.predict(X_fold_val)
            
            cv_scores['accuracy'].append(accuracy_score(y_fold_val, y_pred))
            cv_scores['macro_f1'].append(f1_score(y_fold_val, y_pred, average='macro'))
            cv_scores['weighted_f1'].append(f1_score(y_fold_val, y_pred, average='weighted'))
            
            # Per-class F1
            _, _, f1_per_class, _ = precision_recall_fscore_support(
                y_fold_val, y_pred, average=None
            )
            cv_scores['per_class_f1'].append(f1_per_class)
        
        # Summary statistics
        print("\nCross-Validation Results:")
        print(f"Accuracy: {np.mean(cv_scores['accuracy']):.3f} (+/- {np.std(cv_scores['accuracy'])*2:.3f})")
        print(f"Macro F1: {np.mean(cv_scores['macro_f1']):.3f} (+/- {np.std(cv_scores['macro_f1'])*2:.3f})")
        print(f"Weighted F1: {np.mean(cv_scores['weighted_f1']):.3f} (+/- {np.std(cv_scores['weighted_f1'])*2:.3f})")
        
        return cv_scores
    
    def save_results(self, results, cv_scores):
        """Save model results and predictions"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save predictions
        test_predictions = pd.DataFrame({
            'document': self.df.iloc[self.X_test.index]['document'],
            'true_category': self.label_encoder.inverse_transform(self.y_test),
            'predicted_category': self.label_encoder.inverse_transform(results['Test']['y_pred']),
            'correct': self.y_test == results['Test']['y_pred']
        })
        test_predictions.to_csv(f'xgboost_predictions_{timestamp}.csv', index=False)
        
        # Save metrics summary
        metrics_summary = pd.DataFrame({
            'Dataset': ['Train', 'Validation', 'Test'],
            'Accuracy': [results[k]['accuracy'] for k in ['Train', 'Validation', 'Test']],
            'Macro_F1': [results[k]['macro_f1'] for k in ['Train', 'Validation', 'Test']],
            'Weighted_F1': [results[k]['weighted_f1'] for k in ['Train', 'Validation', 'Test']]
        })
        metrics_summary.to_csv(f'xgboost_metrics_{timestamp}.csv', index=False)
        
        print(f"\nResults saved with timestamp: {timestamp}")
        
        return test_predictions, metrics_summary


def main():
    """Main execution function"""
    # Initialize classifier
    classifier = XGBoostEmailClassifier(random_state=42)
    
    # Load data
    classifier.load_data('ml_dataset_v5_full.xlsx')
    
    # Split data
    classifier.split_data(test_size=0.2, val_size=0.15)
    
    # Hyperparameter tuning (optional - takes time)
    # best_model = classifier.hyperparameter_tuning(cv_folds=5)
    
    # Train model (without tuning for quick results)
    classifier.train_model(use_tuned_params=False)
    
    # Evaluate model
    results = classifier.evaluate_model()
    
    # Visualizations
    classifier.plot_confusion_matrix(
        results['Test']['y_true'], 
        results['Test']['y_pred'],
        title="Test Set Confusion Matrix"
    )
    
    # Feature importance
    importance_df = classifier.plot_feature_importance(top_n=20)
    
    # Cross-validation
    cv_scores = classifier.cross_validate(cv_folds=5)
    
    # Save results
    test_predictions, metrics_summary = classifier.save_results(results, cv_scores)
    
    print("\nXGBoost Email Classification Complete!")
    print(f"Final Test Macro-F1: {results['Test']['macro_f1']:.3f}")
    print(f"Final Test Accuracy: {results['Test']['accuracy']:.3f}")


if __name__ == "__main__":
    main()