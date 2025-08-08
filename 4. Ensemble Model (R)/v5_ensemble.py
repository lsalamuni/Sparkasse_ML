#!/usr/bin/env python3
"""
Advanced XGBoost Email Classifier v5 - German Banking Emails
Strategy: Focus on better generalization and reaching 80%+ accuracy

Key Changes from v4:
- Stacking ensemble instead of voting (better for small datasets)
- Pseudo-labeling for semi-supervised learning
- More sophisticated feature engineering
- Advanced cross-validation with different strategies
- Focus on reducing overfitting while boosting performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
    cross_val_score, cross_validate
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_recall_fscore_support,
    balanced_accuracy_score, cohen_kappa_score
)
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, 
    StackingClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import joblib
import os
# Comprehensive warning suppression for clean output
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['LIGHTGBM_VERBOSITY'] = '-1'
# Suppress sklearn warnings
import sklearn
sklearn.set_config(assume_finite=True)


class AdvancedStackingClassifierV5:
    """V5: Advanced stacking with pseudo-labeling and sophisticated feature engineering"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.poly_features = None
        self.feature_selector = None
        self.base_models = {}
        self.meta_model = None
        self.stacking_model = None
        self.selected_features = None
        self.feature_importance_df = None
        self.pseudo_labeled_data = None
        
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
            
            # Remove only constant features
            constant_features = self.X.columns[self.X.nunique() == 1]
            if len(constant_features) > 0:
                print(f"Removing {len(constant_features)} constant features...")
                self.X = self.X.drop(columns=constant_features)
            
            # Encode categories
            self.y_encoded = self.label_encoder.fit_transform(self.y)
            self.n_classes = len(np.unique(self.y_encoded))
            self.class_names = self.label_encoder.classes_
            
            print(f"Dataset shape: {self.X.shape}")
            print(f"Number of classes: {self.n_classes}")
            
            self._analyze_class_distribution()
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
        
    def _analyze_class_distribution(self):
        """Enhanced class distribution analysis"""
        class_counts = pd.Series(self.y).value_counts()
        
        print("\nClass Distribution:")
        self.minority_classes = []
        for cat, count in class_counts.items():
            print(f"{cat}: {count} ({count/len(self.y)*100:.1f}%)")
            if count < 20:
                self.minority_classes.append(self.label_encoder.transform([cat])[0])
        
        # Calculate class weights
        self.class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.y_encoded),
            y=self.y_encoded
        )
        self.class_weight_dict = dict(enumerate(self.class_weights))
        
        print(f"\nMinority classes (<20 samples): {len(self.minority_classes)}")
        
    def advanced_feature_engineering(self):
        """Create sophisticated features"""
        print("\nPerforming advanced feature engineering...")
        
        X_eng = self.X.copy()
        
        # 1. Statistical aggregations
        numeric_cols = X_eng.select_dtypes(include=[np.number]).columns
        
        # Group features by type for better aggregations
        length_features = [col for col in numeric_cols if 'count' in col or 'length' in col]
        punct_features = [col for col in numeric_cols if 'punct' in col or 'mark' in col]
        keyword_features = [col for col in numeric_cols if 'keyword' in col]
        
        if len(length_features) > 1:
            X_eng['length_features_mean'] = X_eng[length_features].mean(axis=1)
            X_eng['length_features_std'] = X_eng[length_features].std(axis=1)
            X_eng['length_features_max'] = X_eng[length_features].max(axis=1)
        
        if len(punct_features) > 1:
            X_eng['punct_features_sum'] = X_eng[punct_features].sum(axis=1)
            X_eng['punct_features_max'] = X_eng[punct_features].max(axis=1)
        
        if len(keyword_features) > 1:
            X_eng['keyword_features_sum'] = X_eng[keyword_features].sum(axis=1)
            X_eng['keyword_features_nonzero'] = (X_eng[keyword_features] > 0).sum(axis=1)
        
        # 2. Ratio and interaction features
        if 'char_count' in X_eng.columns and 'word_count' in X_eng.columns:
            X_eng['char_per_word'] = X_eng['char_count'] / (X_eng['word_count'] + 1)
            X_eng['word_density'] = X_eng['word_count'] / (X_eng['char_count'] + 1)
        
        if 'sentence_count' in X_eng.columns and 'word_count' in X_eng.columns:
            X_eng['words_per_sentence'] = X_eng['word_count'] / (X_eng['sentence_count'] + 1)
        
        # 3. Percentile-based features
        if 'char_count' in X_eng.columns:
            char_percentiles = X_eng['char_count'].quantile([0.25, 0.5, 0.75])
            X_eng['is_very_short'] = (X_eng['char_count'] < char_percentiles[0.25]).astype(int)
            X_eng['is_medium_length'] = ((X_eng['char_count'] >= char_percentiles[0.25]) & 
                                        (X_eng['char_count'] <= char_percentiles[0.75])).astype(int)
            X_eng['is_very_long'] = (X_eng['char_count'] > char_percentiles[0.75]).astype(int)
        
        # 4. Banking domain sophistication
        banking_patterns = [
            ('urgent_pattern', ['exclamation_marks', 'all_caps_words']),
            ('formal_pattern', ['sentence_count', 'avg_sentence_length']),
            ('complex_pattern', ['total_punctuation', 'capital_words'])
        ]
        
        for pattern_name, cols in banking_patterns:
            available_cols = [col for col in cols if col in X_eng.columns]
            if len(available_cols) > 1:
                X_eng[pattern_name] = X_eng[available_cols].sum(axis=1)
        
        print(f"Created {len(X_eng.columns) - len(self.X.columns)} new engineered features")
        self.X = X_eng
        
    def create_polynomial_features(self, degree=2, n_features=15):
        """Create polynomial features for non-linear patterns"""
        print(f"\nCreating polynomial features (degree={degree})...")
        
        # Select top features for polynomial transformation
        if hasattr(self, 'X_train_scaled'):
            selector = SelectKBest(mutual_info_classif, k=n_features)
            X_selected = selector.fit_transform(self.X_train_scaled, self.y_train)
            selected_idx = selector.get_support(indices=True)
            selected_cols = self.X.columns[selected_idx]
            
            # Create polynomial features on selected features only
            self.poly_features = PolynomialFeatures(
                degree=degree, 
                interaction_only=True, 
                include_bias=False
            )
            
            X_poly = self.poly_features.fit_transform(X_selected)
            
            # Add polynomial features to training data
            poly_feature_names = [f"poly_{i}" for i in range(X_poly.shape[1] - len(selected_cols))]
            X_poly_new = X_poly[:, len(selected_cols):]  # Only new features
            
            X_poly_df = pd.DataFrame(X_poly_new, columns=poly_feature_names, index=self.X_train.index)
            
            # Combine with original features
            self.X_train_with_poly = pd.concat([
                pd.DataFrame(self.X_train_scaled, columns=self.X.columns, index=self.X_train.index),
                X_poly_df
            ], axis=1)
            
            print(f"Added {len(poly_feature_names)} polynomial features")
            return self.X_train_with_poly.values
        else:
            print("No training data available for polynomial features")
            return None
        
    def split_data(self, test_size=0.2, val_size=0.15):
        """Data splitting with enhanced validation"""
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
        
        sys.stdout.flush()
        
        # Enhanced sample weights with class-specific boosts
        sample_weights = compute_sample_weight('balanced', self.y_train)
        
        # Extra boost for smallest classes
        for class_idx in self.minority_classes:
            mask = self.y_train == class_idx
            sample_weights[mask] *= 1.8  # 80% extra weight
        
        self.sample_weights = sample_weights
        
    def intelligent_feature_selection(self, n_features=30):
        """Multi-stage feature selection"""
        print(f"\nPerforming intelligent feature selection (target: {n_features} features)...")
        
        # Stage 1: Remove low-variance features
        from sklearn.feature_selection import VarianceThreshold
        var_selector = VarianceThreshold(threshold=0.01)
        X_var = var_selector.fit_transform(self.X_train_scaled)
        
        # Stage 2: Mutual information
        mi_selector = SelectKBest(mutual_info_classif, k=min(50, X_var.shape[1]))
        X_mi = mi_selector.fit_transform(X_var, self.y_train)
        
        # Stage 3: Model-based selection with XGBoost
        xgb_selector = SelectFromModel(
            xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                random_state=self.random_state,
                verbosity=0
            ),
            max_features=n_features
        )
        
        # Combine selectors
        var_mask = var_selector.get_support()
        mi_mask = mi_selector.get_support()
        
        # Apply variance selector first
        X_temp = self.X_train_scaled[:, var_mask]
        # Then apply MI selector
        X_temp = X_temp[:, mi_mask]
        # Finally apply model-based selector
        xgb_selector.fit(X_temp, self.y_train)
        
        # Get final feature indices
        final_mask = var_mask.copy()
        final_mask[var_mask] = mi_mask
        temp_mask = final_mask.copy()
        temp_mask[final_mask] = xgb_selector.get_support()
        
        self.selected_features = self.X.columns[temp_mask].tolist()
        
        # Store the mask for later use instead of lambda
        self.feature_mask = temp_mask
        
        # Create a simple method for feature selection
        def apply_feature_mask(X):
            return X[:, temp_mask] if X.ndim > 1 else X[temp_mask]
        
        self.feature_selector = apply_feature_mask
        
        print(f"Selected {len(self.selected_features)} features through multi-stage selection")
        print(f"Top 10 selected features: {self.selected_features[:10]}")
        
        return self.X_train_scaled[:, temp_mask]
    
    def apply_advanced_sampling(self, X_train, y_train):
        """Advanced sampling strategy combining oversampling and undersampling"""
        print("\nApplying advanced sampling strategy...")
        
        try:
            # First apply SMOTEENN (combines SMOTE + Edited Nearest Neighbours)
            smoteenn = SMOTEENN(
                smote=SMOTE(random_state=self.random_state, k_neighbors=3),
                enn=EditedNearestNeighbours(n_neighbors=3),
                random_state=self.random_state
            )
            X_resampled, y_resampled = smoteenn.fit_resample(X_train, y_train)
            print(f"SMOTEENN applied. Shape: {X_resampled.shape}")
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"SMOTEENN failed ({str(e)}), trying ADASYN...")
            try:
                adasyn = ADASYN(random_state=self.random_state, n_neighbors=3)
                X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
                print(f"ADASYN applied. Shape: {X_resampled.shape}")
                return X_resampled, y_resampled
            except Exception as e2:
                print(f"All sampling methods failed ({str(e2)}). Using original data...")
                return X_train, y_train
    
    def _create_bagging_classifier(self):
        """Create BaggingClassifier with version compatibility"""
        base_lr = LogisticRegression(
            C=0.5, 
            class_weight='balanced', 
            max_iter=500,
            random_state=self.random_state
        )
        
        # Try new parameter name first
        try:
            return BaggingClassifier(
                estimator=base_lr,
                n_estimators=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        except TypeError:
            # Fallback to old parameter name
            try:
                return BaggingClassifier(
                    base_estimator=base_lr,
                    n_estimators=10,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            except TypeError:
                # If both fail, use simple LogisticRegression
                print("BaggingClassifier failed, using LogisticRegression...")
                return LogisticRegression(
                    C=0.5,
                    class_weight='balanced',
                    max_iter=500,
                    random_state=self.random_state,
                    n_jobs=-1
                )
    
    def create_base_models(self):
        """Create diverse base models for stacking"""
        print("\nCreating diverse base models for stacking...")
        
        self.base_models = {
            # Gradient Boosting Models (use v3 conservative settings)
            'xgb': xgb.XGBClassifier(
                n_estimators=150,
                max_depth=3,
                min_child_weight=5,
                learning_rate=0.08,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=2.0,
                reg_lambda=3.0,
                gamma=0.5,
                random_state=self.random_state,
                verbosity=0
            ),
            
            'lgb': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=2.0,
                class_weight='balanced',
                random_state=self.random_state,
                verbosity=-1,
                force_row_wise=True,  # Suppress warnings
                feature_pre_filter=False  # Prevent feature name issues
            ),
            
            'cat': CatBoostClassifier(
                iterations=100,
                depth=4,
                learning_rate=0.1,
                l2_leaf_reg=3.0,
                bootstrap_type='Bernoulli',
                subsample=0.8,
                class_weights=self.class_weight_dict,
                random_state=self.random_state,
                verbose=False
            ),
            
            # Tree-based Models
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'et': ExtraTreesClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            # Linear Models
            'lr': LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'svm': self._create_bagging_classifier(),
            
            # Distance-based Model
            'knn': KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                n_jobs=-1
            ),
            
            # Probabilistic Model
            'nb': GaussianNB()
        }
        
        print(f"Created {len(self.base_models)} diverse base models")
        
    def train_stacking_ensemble(self):
        """Train stacking ensemble with cross-validation"""
        print("\nTraining stacking ensemble...")
        
        # Prepare data using feature mask
        if hasattr(self, 'feature_mask'):
            X_train = self.X_train_scaled[:, self.feature_mask]
            X_val = self.X_val_scaled[:, self.feature_mask]
        else:
            X_train = self.X_train_scaled
            X_val = self.X_val_scaled
        
        # Apply advanced sampling
        X_train_sampled, y_train_sampled = self.apply_advanced_sampling(X_train, self.y_train)
        
        # Create meta-learner (simple but effective)
        self.meta_model = LogisticRegression(
            C=2.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=self.random_state
        )
        
        # Create stacking classifier
        self.stacking_model = StackingClassifier(
            estimators=list(self.base_models.items()),
            final_estimator=self.meta_model,
            cv=5,  # 5-fold CV for generating meta-features
            stack_method='predict_proba',
            n_jobs=-1,
            verbose=0
        )
        
        # Train the stacking ensemble
        print("Training stacking classifier with 5-fold CV...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.stacking_model.fit(X_train_sampled, y_train_sampled)
            print("Stacking ensemble training completed!")
        except Exception as e:
            print(f"Stacking training failed: {str(e)}")
            print("Falling back to single XGBoost model...")
            
            # Fallback to single model
            self.stacking_model = self.base_models['xgb']
            self.stacking_model.fit(X_train_sampled, y_train_sampled)
            print("Fallback model training completed!")
        
    def pseudo_labeling(self, confidence_threshold=0.9):
        """Implement pseudo-labeling for semi-supervised learning"""
        print(f"\nImplementing pseudo-labeling (confidence threshold: {confidence_threshold})...")
        
        # Prepare validation data using feature mask
        if hasattr(self, 'feature_mask'):
            X_val = self.X_val_scaled[:, self.feature_mask]
        else:
            X_val = self.X_val_scaled
        
        # Get predictions with probabilities
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y_val_proba = self.stacking_model.predict_proba(X_val)
            y_val_pred = self.stacking_model.predict(X_val)
        
        # Find high-confidence predictions
        max_proba = np.max(y_val_proba, axis=1)
        high_confidence_mask = max_proba >= confidence_threshold
        
        if np.sum(high_confidence_mask) > 0:
            print(f"Found {np.sum(high_confidence_mask)} high-confidence pseudo-labels")
            
            # Get high-confidence pseudo-labels
            X_pseudo = X_val[high_confidence_mask]
            y_pseudo = y_val_pred[high_confidence_mask]
            
            # Combine with original training data
            if hasattr(self, 'feature_mask'):
                X_train_orig = self.X_train_scaled[:, self.feature_mask]
            else:
                X_train_orig = self.X_train_scaled
            
            X_combined = np.vstack([X_train_orig, X_pseudo])
            y_combined = np.hstack([self.y_train, y_pseudo])
            
            # Retrain the model on combined data
            print("Retraining model with pseudo-labeled data...")
            
            # Apply sampling to combined data
            X_final, y_final = self.apply_advanced_sampling(X_combined, y_combined)
            
            # Retrain stacking model
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.stacking_model.fit(X_final, y_final)
            
            print("Pseudo-labeling completed!")
            return True
        else:
            print("No high-confidence predictions found for pseudo-labeling")
            return False
    
    def evaluate_comprehensive(self):
        """Enhanced evaluation with detailed analysis"""
        print("\nEvaluating model performance...")
        
        # Prepare data using feature mask
        if hasattr(self, 'feature_mask'):
            X_train = self.X_train_scaled[:, self.feature_mask]
            X_val = self.X_val_scaled[:, self.feature_mask]
            X_test = self.X_test_scaled[:, self.feature_mask]
        else:
            X_train = self.X_train_scaled
            X_val = self.X_val_scaled
            X_test = self.X_test_scaled
        
        # Predictions
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y_pred_train = self.stacking_model.predict(X_train)
            y_pred_val = self.stacking_model.predict(X_val)
            y_pred_test = self.stacking_model.predict(X_test)
            
            # Get prediction probabilities for confidence analysis
            y_proba_test = self.stacking_model.predict_proba(X_test)
        
        # Calculate comprehensive metrics
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
            
            # Per-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average=None, labels=range(self.n_classes)
            )
            
            results[name] = {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc,
                'macro_f1': macro_f1,
                'weighted_f1': weighted_f1,
                'kappa': kappa,
                'y_true': y_true,
                'y_pred': y_pred,
                'per_class_f1': f1,
                'per_class_precision': precision,
                'per_class_recall': recall
            }
            
            print(f"\n{name} Set Performance:")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Balanced Accuracy: {balanced_acc:.3f}")
            print(f"  Macro F1: {macro_f1:.3f}")
            print(f"  Weighted F1: {weighted_f1:.3f}")
            print(f"  Cohen's Kappa: {kappa:.3f}")
        
        # Overfitting analysis
        train_test_gap = results['Train']['macro_f1'] - results['Test']['macro_f1']
        print(f"\nOverfitting Analysis:")
        print(f"  Train-Test Macro-F1 Gap: {train_test_gap:.3f}")
        
        # Detailed test set report
        print("\nDetailed Test Set Classification Report:")
        print(classification_report(
            self.y_test, y_pred_test,
            target_names=self.class_names,
            digits=3
        ))
        
        # Confidence analysis
        avg_confidence = np.mean(np.max(y_proba_test, axis=1))
        print(f"\nAverage prediction confidence: {avg_confidence:.3f}")
        
        # Model ensemble analysis
        self._analyze_base_model_contributions()
        
        return results
    
    def _analyze_base_model_contributions(self):
        """Analyze how much each base model contributes"""
        print("\nAnalyzing base model contributions...")
        
        try:
            # Get meta-learner coefficients
            if hasattr(self.stacking_model.final_estimator_, 'coef_'):
                coeffs = self.stacking_model.final_estimator_.coef_
                
                # For multiclass, average across classes
                if coeffs.ndim > 1:
                    avg_coeffs = np.mean(np.abs(coeffs), axis=0)
                else:
                    avg_coeffs = np.abs(coeffs)
                
                # Map to base models (each model contributes n_classes coefficients)
                n_base_models = len(self.base_models)
                model_contributions = []
                
                for i, (model_name, _) in enumerate(self.base_models.items()):
                    start_idx = i * self.n_classes
                    end_idx = (i + 1) * self.n_classes
                    if end_idx <= len(avg_coeffs):
                        contribution = np.mean(avg_coeffs[start_idx:end_idx])
                        model_contributions.append((model_name, contribution))
                
                # Sort by contribution
                model_contributions.sort(key=lambda x: x[1], reverse=True)
                
                print("Base model contributions (higher = more important):")
                for model_name, contribution in model_contributions:
                    print(f"  {model_name}: {contribution:.3f}")
                    
        except Exception as e:
            print(f"Could not analyze model contributions: {str(e)}")
    
    def advanced_cross_validation(self, cv_folds=5, repeats=3):
        """Advanced cross-validation with multiple strategies"""
        print(f"\nPerforming advanced cross-validation ({repeats}x{cv_folds} folds)...")
        
        try:
            # Prepare data using feature mask
            if hasattr(self, 'feature_mask'):
                X_cv = self.X_train_scaled[:, self.feature_mask]
            else:
                X_cv = self.X_train_scaled
            
            # Repeated Stratified K-Fold
            rskf = RepeatedStratifiedKFold(
                n_splits=cv_folds, 
                n_repeats=repeats, 
                random_state=self.random_state
            )
            
            # Multiple scoring metrics
            scoring = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted']
            
            cv_results = {}
            for score in scoring:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    scores = cross_val_score(
                        self.stacking_model, X_cv, self.y_train,
                        cv=rskf, scoring=score, n_jobs=-1
                    )
                cv_results[score] = scores
                print(f"{score}: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
            
            return cv_results
            
        except Exception as e:
            print(f"Advanced cross-validation failed: {str(e)}")
            return None
    
    def save_comprehensive_results(self, results):
        """Save comprehensive results with all details"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save the stacking ensemble
            joblib.dump(self.stacking_model, f'v5_stacking_model_{timestamp}.pkl')
            if hasattr(self, 'feature_mask'):
                joblib.dump(self.feature_mask, f'v5_feature_mask_{timestamp}.pkl')
            joblib.dump(self.scaler, f'v5_scaler_{timestamp}.pkl')
            
            # Save detailed predictions
            if hasattr(self, 'feature_mask'):
                X_test = self.X_test_scaled[:, self.feature_mask]
            else:
                X_test = self.X_test_scaled
                
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                y_proba_test = self.stacking_model.predict_proba(X_test)
            
            test_predictions = pd.DataFrame({
                'document': self.df.iloc[self.X_test.index]['document'],
                'true_category': self.label_encoder.inverse_transform(self.y_test),
                'predicted_category': self.label_encoder.inverse_transform(results['Test']['y_pred']),
                'confidence': np.max(y_proba_test, axis=1),
                'correct': self.y_test == results['Test']['y_pred']
            })
            
            # Add individual base model predictions for analysis
            for model_name, model in self.base_models.items():
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model_pred = model.predict(X_test) if hasattr(model, 'predict') else np.zeros(len(X_test))
                    test_predictions[f'{model_name}_prediction'] = self.label_encoder.inverse_transform(model_pred)
            
            test_predictions.to_csv(f'v5_detailed_predictions_{timestamp}.csv', index=False)
            
            # Save version comparison
            version_comparison = pd.DataFrame({
                'Version': ['v3', 'v4', 'v5'],
                'Test_Accuracy': [0.740, 0.740, results['Test']['accuracy']],
                'Test_Macro_F1': [0.713, 0.705, results['Test']['macro_f1']],
                'Train_Test_Gap': [0.124, 0.270, results['Train']['macro_f1'] - results['Test']['macro_f1']],
                'Method': ['Regularized Ensemble', 'Voting Ensemble', 'Stacking + Pseudo-labeling']
            })
            version_comparison.to_csv(f'v5_version_comparison_{timestamp}.csv', index=False)
            
            print(f"\nComprehensive results saved with timestamp: {timestamp}")
            return test_predictions, version_comparison
            
        except Exception as e:
            print(f"Could not save results: {str(e)}")
            return None, None


def main():
    """Main execution for V5"""
    try:
        print("Starting v5 Advanced Stacking Email Classifier...")
        print("Strategy: Stacking + Pseudo-labeling + Advanced Feature Engineering\n")
        
        # Initialize classifier
        classifier = AdvancedStackingClassifierV5(random_state=42)
        
        # Load data
        classifier.load_data('ml_dataset_v5_full.xlsx')
        
        # Advanced feature engineering
        classifier.advanced_feature_engineering()
        
        # Split data
        print("\nAbout to split data...")
        classifier.split_data(test_size=0.2, val_size=0.15)
        print("Data splitting completed!")
        
        # Intelligent feature selection (fewer features for better generalization)
        classifier.intelligent_feature_selection(n_features=25)
        
        # Create polynomial features (optional)
        # classifier.create_polynomial_features(degree=2, n_features=10)
        
        # Create diverse base models
        classifier.create_base_models()
        
        # Train stacking ensemble
        classifier.train_stacking_ensemble()
        
        # Apply pseudo-labeling with higher threshold for better quality
        pseudo_success = classifier.pseudo_labeling(confidence_threshold=0.90)
        
        # Comprehensive evaluation
        results = classifier.evaluate_comprehensive()
        
        # Advanced cross-validation
        cv_results = classifier.advanced_cross_validation(cv_folds=5, repeats=2)
        
        # Save comprehensive results
        test_predictions, version_comparison = classifier.save_comprehensive_results(results)
        
        # Final summary
        print("\n" + "="*70)
        print("V5 ADVANCED STACKING RESULTS SUMMARY")
        print("="*70)
        print(f"Test Accuracy: {results['Test']['accuracy']:.3f}")
        print(f"Test Balanced Accuracy: {results['Test']['balanced_accuracy']:.3f}")
        print(f"Test Macro-F1: {results['Test']['macro_f1']:.3f}")
        print(f"Test Weighted-F1: {results['Test']['weighted_f1']:.3f}")
        print(f"Test Cohen's Kappa: {results['Test']['kappa']:.3f}")
        
        print(f"\nOverfitting:")
        print(f"  Train-Test Macro-F1 Gap: {results['Train']['macro_f1'] - results['Test']['macro_f1']:.3f}")
        
        # Version improvements
        v4_accuracy = 0.740
        v5_accuracy = results['Test']['accuracy']
        improvement = v5_accuracy - v4_accuracy
        
        print(f"\nImprovement over v4:")
        print(f"  v4 Accuracy: {v4_accuracy:.3f}")
        print(f"  v5 Accuracy: {v5_accuracy:.3f}")
        print(f"  Improvement: {improvement:+.3f} ({improvement/v4_accuracy*100:+.1f}%)")
        
        target_reached = v5_accuracy >= 0.80
        print(f"\n{'🎯' if target_reached else '📊'} Target 80% accuracy: {'ACHIEVED!' if target_reached else f'Progress: {v5_accuracy:.1%} ({(0.80 - v5_accuracy)*100:.1f}% to go)'}")
        
        if pseudo_success:
            print(f"✅ Pseudo-labeling applied successfully")
        
        print(f"\n✅ v5 Advanced Stacking XGBoost completed!")
        return classifier, results
        
    except Exception as e:
        print(f"\n❌ Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    try:
        classifier, results = main()
        if classifier is None:
            print("Execution failed, but script completed without crashing.")
    except Exception as e:
        print(f"Script failed completely: {str(e)}")
        classifier, results = None, None