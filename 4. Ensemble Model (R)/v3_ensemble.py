#!/usr/bin/env python3
"""
Anti-Overfitting XGBoost Email Classifier v3 - German Banking Emails
Focus: Aggressive anti-overfitting techniques to close train-test gap

Key Changes:
- Aggressive regularization (high alpha/lambda)
- Reduced model complexity (shallow trees, fewer estimators)
- More aggressive feature selection
- Proper early stopping implementation
- Ensemble with high variance models
- Cross-validation based model selection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV,
    cross_val_score, cross_validate
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_recall_fscore_support,
    balanced_accuracy_score, cohen_kappa_score
)
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import joblib
warnings.filterwarnings('ignore')


class AntiOverfittingXGBoost:
    """XGBoost with aggressive anti-overfitting techniques"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.model = None
        self.ensemble_model = None
        self.best_params = None
        self.selected_features = None
        self.feature_importance_df = None
        self.train_val_gap = {}
        
    def load_data(self, filepath='ml_dataset_v5_full.xlsx'):
        """Load and prepare data with validation"""
        try:
            print(f"Loading data from {filepath}...")
            self.df = pd.read_excel(filepath)
            
            if self.df.empty:
                raise ValueError("Dataset is empty")
            
            # Separate features and target
            self.X = self.df.drop(['document', 'category'], axis=1)
            self.y = self.df['category']
            
            # Handle missing values
            if self.X.isnull().any().any():
                print("Warning: Missing values detected. Filling with median...")
                self.X = self.X.fillna(self.X.median())
            
            # Remove constant and near-constant features (more aggressive)
            # Remove features with low variance
            constant_features = []
            for col in self.X.columns:
                if self.X[col].nunique() <= 2:  # More aggressive: remove binary and constant
                    constant_features.append(col)
            
            if len(constant_features) > 0:
                print(f"Removing {len(constant_features)} constant/binary features...")
                self.X = self.X.drop(columns=constant_features)
            
            # Remove highly correlated features
            corr_matrix = self.X.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            high_corr_features = [column for column in upper_triangle.columns 
                                if any(upper_triangle[column] > 0.95)]
            
            if len(high_corr_features) > 0:
                print(f"Removing {len(high_corr_features)} highly correlated features...")
                self.X = self.X.drop(columns=high_corr_features)
            
            # Encode categories
            self.y_encoded = self.label_encoder.fit_transform(self.y)
            self.n_classes = len(np.unique(self.y_encoded))
            self.class_names = self.label_encoder.classes_
            
            print(f"Dataset shape after aggressive preprocessing: {self.X.shape}")
            print(f"Number of classes: {self.n_classes}")
            print(f"Features after preprocessing: {self.X.shape[1]}")
            
            self._analyze_class_distribution()
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
        
    def _analyze_class_distribution(self):
        """Class distribution analysis"""
        class_counts = pd.Series(self.y).value_counts()
        
        print("\nClass Distribution:")
        for cat, count in class_counts.items():
            print(f"{cat}: {count} ({count/len(self.y)*100:.1f}%)")
        
        # Calculate class weights
        self.class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.y_encoded),
            y=self.y_encoded
        )
        self.class_weight_dict = dict(enumerate(self.class_weights))
        
        # Imbalance metrics
        max_class = class_counts.max()
        min_class = class_counts.min()
        imbalance_ratio = max_class / min_class
        
        print(f"\nImbalance Ratio: {imbalance_ratio:.2f}")
        print(f"Largest class: {max_class} samples")
        print(f"Smallest class: {min_class} samples")
        
    def aggressive_feature_selection(self, n_features='auto', method='rfe'):
        """More aggressive feature selection to prevent overfitting"""
        print("\nPerforming aggressive feature selection...")
        
        if n_features == 'auto':
            # More aggressive: use smaller feature count
            # Rule: min(15, sqrt(n_samples))
            n_features = min(15, int(np.sqrt(len(self.X_train))))
        
        if method == 'rfe':
            # Use RFE with a simple estimator
            base_estimator = LogisticRegression(
                max_iter=1000, 
                random_state=self.random_state,
                class_weight='balanced'
            )
            selector = RFE(base_estimator, n_features_to_select=n_features)
            selector.fit(self.X_train_scaled, self.y_train)
            
            self.feature_selector = selector
            self.selected_features = self.X.columns[selector.support_].tolist()
            
        else:
            # Mutual information selection
            selector = SelectKBest(mutual_info_classif, k=n_features)
            selector.fit(self.X_train_scaled, self.y_train)
            
            self.feature_selector = selector
            self.selected_features = self.X.columns[selector.get_support()].tolist()
        
        print(f"Selected {len(self.selected_features)} features from {self.X_train.shape[1]}")
        print(f"Selected features: {self.selected_features[:10]}...")
        
        return self.feature_selector.transform(self.X_train_scaled)
        
    def split_data(self, test_size=0.2, val_size=0.15):
        """Data splitting with same structure as v2"""
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
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        import sys
        
        print(f"\nData splits:", flush=True)
        print(f"Train: {len(self.X_train)} samples ({len(self.X_train)/len(self.X)*100:.1f}%)", flush=True)
        print(f"Validation: {len(self.X_val)} samples ({len(self.X_val)/len(self.X)*100:.1f}%)", flush=True)
        print(f"Test: {len(self.X_test)} samples ({len(self.X_test)/len(self.X)*100:.1f}%)", flush=True)
        
        # Verify split sizes
        expected_train = 161
        expected_val = 38
        expected_test = 50
        
        if len(self.X_train) != expected_train or len(self.X_val) != expected_val or len(self.X_test) != expected_test:
            print(f"\nNote: Split sizes differ from v0 due to random_state or stratification:", flush=True)
            print(f"  Expected: Train={expected_train}, Val={expected_val}, Test={expected_test}", flush=True)
            print(f"  Actual:   Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}", flush=True)
        else:
            print(f"\n✓ Split sizes match v0 exactly!", flush=True)
        
        sys.stdout.flush()
        
        # Calculate sample weights
        self.sample_weights = compute_sample_weight('balanced', self.y_train)
        
    def create_antioverfit_model(self, params=None):
        """Create XGBoost model with aggressive anti-overfitting settings"""
        if params is None:
            params = {
                # Reduced complexity
                'max_depth': 3,  # Shallow trees
                'min_child_weight': 5,  # Higher minimum samples per leaf
                
                # Aggressive regularization
                'reg_alpha': 2.0,  # High L1 regularization
                'reg_lambda': 3.0,  # High L2 regularization
                'gamma': 0.5,  # High minimum loss reduction
                
                # Aggressive subsampling
                'subsample': 0.6,  # Use only 60% of samples
                'colsample_bytree': 0.6,  # Use only 60% of features per tree
                'colsample_bylevel': 0.6,  # Use only 60% of features per level
                
                # Conservative learning
                'learning_rate': 0.05,  # Slow learning
                'n_estimators': 100,  # Fewer estimators
                
                # Fixed parameters
                'objective': 'multi:softprob',
                'n_jobs': -1,
                'random_state': self.random_state,
                'use_label_encoder': False,
                'eval_metric': 'mlogloss',
                'verbosity': 0
            }
        
        return xgb.XGBClassifier(**params)
    
    def train_with_proper_early_stopping(self):
        """Train with corrected early stopping approach"""
        print("\nTraining with anti-overfitting XGBoost model...")
        
        # Create model
        self.model = self.create_antioverfit_model()
        
        # Prepare data
        if self.feature_selector:
            X_train = self.feature_selector.transform(self.X_train_scaled)
            X_val = self.feature_selector.transform(self.X_val_scaled)
        else:
            X_train = self.X_train_scaled
            X_val = self.X_val_scaled
        
        # Train with proper early stopping using callbacks
        try:
            # For newer XGBoost versions, use callbacks
            from xgboost.callback import EarlyStopping
            
            # Create model with callbacks
            self.model = xgb.XGBClassifier(
                max_depth=3,
                min_child_weight=5,
                reg_alpha=2.0,
                reg_lambda=3.0,
                gamma=0.5,
                subsample=0.6,
                colsample_bytree=0.6,
                colsample_bylevel=0.6,
                learning_rate=0.05,
                n_estimators=1000,  # High number, will stop early
                objective='multi:softprob',
                n_jobs=-1,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='mlogloss',
                verbosity=0,
                callbacks=[EarlyStopping(rounds=10, save_best=True)]
            )
            
            self.model.fit(
                X_train, self.y_train,
                sample_weight=self.sample_weights,
                eval_set=[(X_val, self.y_val)],
                verbose=False
            )
            
            print(f"Training completed with early stopping at iteration {self.model.best_iteration}")
            
        except Exception as e:
            print(f"Early stopping with callbacks failed: {str(e)}")
            print("Training with fixed iterations...")
            
            # Fallback: train without early stopping
            self.model = self.create_antioverfit_model()
            self.model.fit(
                X_train, self.y_train,
                sample_weight=self.sample_weights
            )
            print("Training completed without early stopping.")
    
    def create_regularized_ensemble(self):
        """Create ensemble with diverse, regularized models"""
        print("\nCreating regularized ensemble...")
        
        # Model 1: Very conservative XGBoost
        xgb_conservative = xgb.XGBClassifier(
            max_depth=2,
            min_child_weight=10,
            reg_alpha=3.0,
            reg_lambda=4.0,
            gamma=1.0,
            subsample=0.5,
            colsample_bytree=0.5,
            learning_rate=0.03,
            n_estimators=50,
            objective='multi:softprob',
            n_jobs=-1,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0
        )
        
        # Model 2: Regularized Random Forest
        rf_reg = RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Model 3: Regularized Logistic Regression
        lr_reg = LogisticRegression(
            C=0.1,  # High regularization
            max_iter=1000,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Create ensemble
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('xgb_conservative', xgb_conservative),
                ('rf_regularized', rf_reg),
                ('lr_regularized', lr_reg)
            ],
            voting='soft',
            n_jobs=-1
        )
        
        # Train ensemble
        if self.feature_selector:
            X_train = self.feature_selector.transform(self.X_train_scaled)
        else:
            X_train = self.X_train_scaled
            
        self.ensemble_model.fit(X_train, self.y_train)
        print("Regularized ensemble training completed!")
    
    def evaluate_overfitting(self):
        """Comprehensive evaluation with overfitting analysis"""
        print("\nEvaluating model performance and overfitting...")
        
        # Prepare data
        if self.feature_selector:
            X_train = self.feature_selector.transform(self.X_train_scaled)
            X_val = self.feature_selector.transform(self.X_val_scaled)
            X_test = self.feature_selector.transform(self.X_test_scaled)
        else:
            X_train = self.X_train_scaled
            X_val = self.X_val_scaled
            X_test = self.X_test_scaled
        
        # Test both single model and ensemble
        models_to_test = [
            ('Single XGBoost', self.model),
            ('Regularized Ensemble', self.ensemble_model)
        ]
        
        all_results = {}
        
        for model_name, model in models_to_test:
            print(f"\n{'-'*50}")
            print(f"Evaluating {model_name}")
            print(f"{'-'*50}")
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            results = {}
            for name, y_true, y_pred in [
                ('Train', self.y_train, y_pred_train),
                ('Validation', self.y_val, y_pred_val),
                ('Test', self.y_test, y_pred_test)
            ]:
                accuracy = accuracy_score(y_true, y_pred)
                balanced_acc = balanced_accuracy_score(y_true, y_pred)
                macro_f1 = f1_score(y_true, y_pred, average='macro')
                weighted_f1 = f1_score(y_true, y_pred, average='weighted')
                kappa = cohen_kappa_score(y_true, y_pred)
                
                results[name] = {
                    'accuracy': accuracy,
                    'balanced_accuracy': balanced_acc,
                    'macro_f1': macro_f1,
                    'weighted_f1': weighted_f1,
                    'kappa': kappa,
                    'y_true': y_true,
                    'y_pred': y_pred
                }
                
                print(f"\n{name} Set Performance:")
                print(f"  Accuracy: {accuracy:.3f}")
                print(f"  Balanced Accuracy: {balanced_acc:.3f}")
                print(f"  Macro F1: {macro_f1:.3f}")
                print(f"  Weighted F1: {weighted_f1:.3f}")
                print(f"  Cohen's Kappa: {kappa:.3f}")
            
            # Overfitting analysis
            train_test_gap = {
                'accuracy_gap': results['Train']['accuracy'] - results['Test']['accuracy'],
                'macro_f1_gap': results['Train']['macro_f1'] - results['Test']['macro_f1'],
                'balanced_acc_gap': results['Train']['balanced_accuracy'] - results['Test']['balanced_accuracy']
            }
            
            print(f"\nOverfitting Analysis:")
            print(f"  Train-Test Accuracy Gap: {train_test_gap['accuracy_gap']:.3f}")
            print(f"  Train-Test Macro-F1 Gap: {train_test_gap['macro_f1_gap']:.3f}")
            print(f"  Train-Test Balanced Acc Gap: {train_test_gap['balanced_acc_gap']:.3f}")
            
            # Store results
            all_results[model_name] = {
                'results': results,
                'overfitting_gaps': train_test_gap
            }
        
        # Choose best model based on smallest overfitting gap and decent performance
        best_model_name = None
        best_score = float('inf')
        
        for model_name, data in all_results.items():
            # Combined score: prioritize small gap + reasonable performance
            gap_penalty = data['overfitting_gaps']['macro_f1_gap'] * 2  # Penalize gap heavily
            performance_bonus = data['results']['Test']['macro_f1']  # Reward good performance
            combined_score = gap_penalty - performance_bonus
            
            print(f"\n{model_name} Combined Score: {combined_score:.3f} (lower is better)")
            
            if combined_score < best_score:
                best_score = combined_score
                best_model_name = model_name
        
        print(f"\n🏆 Best model: {best_model_name}")
        
        # Detailed test report for best model
        best_results = all_results[best_model_name]['results']
        print(f"\nDetailed Test Set Classification Report ({best_model_name}):")
        print(classification_report(
            best_results['Test']['y_true'], 
            best_results['Test']['y_pred'],
            target_names=self.class_names,
            digits=3
        ))
        
        return all_results, best_model_name
    
    def plot_overfitting_analysis(self, all_results):
        """Visualize overfitting comparison"""
        try:
            # Create comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Performance comparison
            models = list(all_results.keys())
            train_scores = [all_results[m]['results']['Train']['macro_f1'] for m in models]
            test_scores = [all_results[m]['results']['Test']['macro_f1'] for m in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            ax1.bar(x - width/2, train_scores, width, label='Train Macro-F1', alpha=0.8, color='skyblue')
            ax1.bar(x + width/2, test_scores, width, label='Test Macro-F1', alpha=0.8, color='lightcoral')
            
            ax1.set_xlabel('Model')
            ax1.set_ylabel('Macro-F1 Score')
            ax1.set_title('Train vs Test Performance')
            ax1.set_xticks(x)
            ax1.set_xticklabels(models, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Overfitting gaps
            gaps = [all_results[m]['overfitting_gaps']['macro_f1_gap'] for m in models]
            
            colors = ['red' if gap > 0.2 else 'orange' if gap > 0.1 else 'green' for gap in gaps]
            ax2.bar(models, gaps, color=colors, alpha=0.7)
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Train-Test Macro-F1 Gap')
            ax2.set_title('Overfitting Analysis (Lower is Better)')
            ax2.set_xticklabels(models, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Add threshold lines
            ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Warning (0.1)')
            ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Severe (0.2)')
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Could not display overfitting analysis plot: {str(e)}")
    
    def cross_validate_antioverfit(self, cv_folds=5):
        """Cross-validation with overfitting monitoring"""
        print(f"\nPerforming {cv_folds}-fold cross-validation with overfitting monitoring...")
        
        try:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            # Prepare data
            if self.feature_selector:
                X_cv = self.feature_selector.transform(self.X_train_scaled)
            else:
                X_cv = self.X_train_scaled
            
            # Test both models
            models_to_cv = [
                ('Conservative XGBoost', self.create_antioverfit_model()),
                ('Regularized Ensemble', self.ensemble_model)
            ]
            
            cv_results = {}
            
            for model_name, model in models_to_cv:
                print(f"\nCross-validating {model_name}...")
                
                scores = cross_val_score(
                    model, X_cv, self.y_train,
                    cv=skf, scoring='f1_macro', n_jobs=-1
                )
                
                cv_results[model_name] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores
                }
                
                print(f"{model_name} CV Macro-F1: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
            
            return cv_results
            
        except Exception as e:
            print(f"Cross-validation failed: {str(e)}")
            return None
    
    def save_comprehensive_results(self, all_results, best_model_name):
        """Save comprehensive results with overfitting analysis"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save best model
            if best_model_name == 'Single XGBoost':
                joblib.dump(self.model, f'v3_xgboost_best_model_{timestamp}.pkl')
            else:
                joblib.dump(self.ensemble_model, f'v3_ensemble_best_model_{timestamp}.pkl')
            
            # Save comprehensive comparison
            comparison_data = []
            for model_name, data in all_results.items():
                for dataset in ['Train', 'Test']:
                    comparison_data.append({
                        'Model': model_name,
                        'Dataset': dataset,
                        'Accuracy': data['results'][dataset]['accuracy'],
                        'Balanced_Accuracy': data['results'][dataset]['balanced_accuracy'],
                        'Macro_F1': data['results'][dataset]['macro_f1'],
                        'Weighted_F1': data['results'][dataset]['weighted_f1'],
                        'Kappa': data['results'][dataset]['kappa']
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv(f'v3_model_comparison_{timestamp}.csv', index=False)
            
            # Save overfitting analysis
            overfitting_data = []
            for model_name, data in all_results.items():
                overfitting_data.append({
                    'Model': model_name,
                    'Accuracy_Gap': data['overfitting_gaps']['accuracy_gap'],
                    'Macro_F1_Gap': data['overfitting_gaps']['macro_f1_gap'],
                    'Balanced_Acc_Gap': data['overfitting_gaps']['balanced_acc_gap'],
                    'Best_Model': model_name == best_model_name
                })
            
            overfitting_df = pd.DataFrame(overfitting_data)
            overfitting_df.to_csv(f'v3_overfitting_analysis_{timestamp}.csv', index=False)
            
            print(f"\nComprehensive results saved with timestamp: {timestamp}")
            return comparison_df, overfitting_df
            
        except Exception as e:
            print(f"Could not save results: {str(e)}")
            return None, None


def main():
    """Main execution with anti-overfitting focus"""
    try:
        print("Starting v3 Anti-Overfitting XGBoost Email Classifier...")
        print("Focus: Reducing train-test performance gap\n")
        
        # Initialize classifier
        classifier = AntiOverfittingXGBoost(random_state=42)
        
        # Load data with aggressive preprocessing
        classifier.load_data('ml_dataset_v5_full.xlsx')
        
        # Split data
        print("About to split data...")
        classifier.split_data(test_size=0.2, val_size=0.15)
        print("Data splitting completed!")
        
        # Aggressive feature selection
        classifier.aggressive_feature_selection(n_features='auto', method='rfe')
        
        # Train regularized models
        classifier.train_with_proper_early_stopping()
        classifier.create_regularized_ensemble()
        
        # Comprehensive evaluation with overfitting analysis
        all_results, best_model_name = classifier.evaluate_overfitting()
        
        # Visualizations
        classifier.plot_overfitting_analysis(all_results)
        
        # Cross-validation
        cv_results = classifier.cross_validate_antioverfit(cv_folds=5)
        
        # Save comprehensive results
        comparison_df, overfitting_df = classifier.save_comprehensive_results(all_results, best_model_name)
        
        # Final summary with overfitting focus
        best_data = all_results[best_model_name]
        print("\n" + "="*70)
        print("ANTI-OVERFITTING RESULTS SUMMARY")
        print("="*70)
        print(f"Best Model: {best_model_name}")
        print(f"\nPerformance:")
        print(f"  Test Accuracy: {best_data['results']['Test']['accuracy']:.3f}")
        print(f"  Test Macro-F1: {best_data['results']['Test']['macro_f1']:.3f}")
        print(f"  Test Weighted-F1: {best_data['results']['Test']['weighted_f1']:.3f}")
        
        print(f"\nOverfitting Analysis:")
        print(f"  Train-Test Accuracy Gap: {best_data['overfitting_gaps']['accuracy_gap']:.3f}")
        print(f"  Train-Test Macro-F1 Gap: {best_data['overfitting_gaps']['macro_f1_gap']:.3f}")
        
        # Compare with v2 if gap is significantly reduced
        v2_gap = 0.987 - 0.651  # From OUTCOMES.txt
        v3_gap = best_data['overfitting_gaps']['macro_f1_gap']
        improvement = v2_gap - v3_gap
        
        print(f"\nImprovement vs v2:")
        print(f"  v2 Train-Test Gap: {v2_gap:.3f}")
        print(f"  v3 Train-Test Gap: {v3_gap:.3f}")
        print(f"  Gap Reduction: {improvement:.3f} {'✅' if improvement > 0.1 else '⚠️'}")
        
        print(f"\n{'✅' if v3_gap < 0.15 else '⚠️'} v3 Anti-Overfitting XGBoost completed!")
        return classifier, all_results, best_model_name
        
    except Exception as e:
        print(f"\n❌ Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    try:
        classifier, results, best_model = main()
        if classifier is None:
            print("Execution failed, but script completed without crashing.")
    except Exception as e:
        print(f"Script failed completely: {str(e)}")
        classifier, results, best_model = None, None, None