#!/usr/bin/env python3
"""
Ensemble Email Classifier v7 - German Banking Emails

Key Changes from v6:
- Clear, streamlined output formatting
- Matplotlib confusion matrix visualization
- Removed unnecessary comments
- Maintained all v6 accuracy optimizations
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
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel, chi2
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, 
    StackingClassifier, BaggingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
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


class UltimateStackingClassifierV7:
    """V7: Exact v6 replica with clearer output formatting"""
    
    def __init__(self, random_state=42): # Set as 42 for reproducibility / None for randomness 
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.base_models = {}
        self.meta_model = None
        self.stacking_model = None
        self.selected_features = None
        self.feature_importance_df = None
        self.pseudo_labeled_data = None
        
        # V6: Track problematic classes from v5 analysis
        self.problematic_classes = {
            'C - Bearbeitungsdauer': {'f1': 0.333, 'target_improvement': 0.4},
            'I - Namensänderung': {'f1': 0.500, 'target_improvement': 0.3},
            'K - Servicequalität': {'f1': 0.500, 'target_improvement': 0.3},
            'E - Gerätewechsel': {'f1': 0.571, 'target_improvement': 0.2}
        }
        
    def load_data(self, filepath='ml_dataset_v5_full.xlsx'):
        """Load and prepare data with enhanced validation"""
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
        """V6: Enhanced feature engineering with class-specific features"""
        print("\nPerforming advanced feature engineering with class-specific improvements...")
        
        X_eng = self.X.copy()
        
        # 1. Enhanced statistical features
        numeric_cols = X_eng.select_dtypes(include=[np.number]).columns
        
        # More sophisticated aggregations
        length_features = [col for col in numeric_cols if any(x in col.lower() for x in ['count', 'length', 'char', 'word'])]
        punct_features = [col for col in numeric_cols if any(x in col.lower() for x in ['punct', 'mark', 'exclamation', 'question'])]
        keyword_features = [col for col in numeric_cols if 'keyword' in col.lower()]
        
        if len(length_features) > 1:
            X_eng['length_features_mean'] = X_eng[length_features].mean(axis=1)
            X_eng['length_features_std'] = X_eng[length_features].std(axis=1).fillna(0)
            X_eng['length_features_skew'] = X_eng[length_features].skew(axis=1).fillna(0)
            X_eng['length_complexity'] = X_eng[length_features].max(axis=1) / (X_eng[length_features].mean(axis=1) + 1)
        
        if len(punct_features) > 1:
            X_eng['punct_intensity'] = X_eng[punct_features].sum(axis=1)
            X_eng['punct_variety'] = (X_eng[punct_features] > 0).sum(axis=1)
        
        if len(keyword_features) > 1:
            X_eng['keyword_richness'] = X_eng[keyword_features].sum(axis=1)
            X_eng['keyword_diversity'] = (X_eng[keyword_features] > 0).sum(axis=1)
        
        # 2. Class-specific features based on v5 analysis
        
        # For "Bearbeitungsdauer" (processing time) - F1: 0.333
        processing_indicators = []
        for col in X_eng.columns:
            if any(word in col.lower() for word in ['time', 'wait', 'process', 'duration', 'delay']):
                processing_indicators.append(col)
        
        if processing_indicators:
            X_eng['processing_signal'] = X_eng[processing_indicators].sum(axis=1)
        else:
            # Create synthetic processing signal
            if 'char_count' in X_eng.columns and 'sentence_count' in X_eng.columns:
                X_eng['processing_signal'] = (X_eng['char_count'] / (X_eng['sentence_count'] + 1)) * 0.1
        
        # For "Servicequalität" (service quality) - F1: 0.500
        quality_indicators = []
        for col in X_eng.columns:
            if any(word in col.lower() for word in ['service', 'quality', 'complaint', 'satisfaction']):
                quality_indicators.append(col)
        
        if quality_indicators:
            X_eng['quality_signal'] = X_eng[quality_indicators].sum(axis=1)
        else:
            # Create synthetic quality signal
            if 'exclamation_marks' in X_eng.columns and 'question_marks' in X_eng.columns:
                X_eng['quality_signal'] = X_eng['exclamation_marks'] + X_eng['question_marks'] * 2
        
        # For "Namensänderung" (name change) - F1: 0.500
        name_indicators = []
        for col in X_eng.columns:
            if any(word in col.lower() for word in ['name', 'change', 'update', 'modify']):
                name_indicators.append(col)
        
        if name_indicators:
            X_eng['name_change_signal'] = X_eng[name_indicators].sum(axis=1)
        else:
            # Create synthetic name change signal
            if 'capital_words' in X_eng.columns:
                X_eng['name_change_signal'] = X_eng['capital_words'] * 0.5
        
        # For "Gerätewechsel" (device change) - F1: 0.571
        device_indicators = []
        for col in X_eng.columns:
            if any(word in col.lower() for word in ['device', 'mobile', 'app', 'phone', 'tablet']):
                device_indicators.append(col)
        
        if device_indicators:
            X_eng['device_signal'] = X_eng[device_indicators].sum(axis=1)
        else:
            # Create synthetic device signal
            if 'word_app' in X_eng.columns:
                X_eng['device_signal'] = X_eng['word_app'] * 2
        
        # 3. Advanced interaction features
        if 'char_count' in X_eng.columns and 'word_count' in X_eng.columns:
            X_eng['text_complexity'] = X_eng['char_count'] / (X_eng['word_count'] + 1)
            X_eng['text_conciseness'] = X_eng['word_count'] / (X_eng['char_count'] + 1)
        
        # 4. Percentile-based features for better separation
        for col in ['char_count', 'word_count', 'sentence_count']:
            if col in X_eng.columns:
                try:
                    percentiles = X_eng[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
                    # Remove duplicate percentiles to avoid bin edge errors
                    unique_percentiles = sorted(list(set(percentiles.tolist())))
                    
                    if len(unique_percentiles) > 1:  # Need at least 2 unique values
                        bins = [-np.inf] + unique_percentiles + [np.inf]
                        X_eng[f'{col}_decile'] = pd.cut(X_eng[col], 
                                                       bins=bins, 
                                                       labels=False,
                                                       duplicates='drop')
                    else:
                        # If all values are the same, create a simple binary feature
                        X_eng[f'{col}_decile'] = 0
                except Exception as e:
                    print(f"Warning: Could not create decile feature for {col}: {str(e)}")
                    # Create a simple feature based on median split
                    median_val = X_eng[col].median()
                    X_eng[f'{col}_decile'] = (X_eng[col] > median_val).astype(int)
        
        print(f"Created {len(X_eng.columns) - len(self.X.columns)} new engineered features")
        self.X = X_eng
        
    def intelligent_feature_selection(self, n_features=20):
        """V6: Smarter feature selection with domain knowledge"""
        print(f"\nPerforming intelligent feature selection (target: {n_features} features)...")
        
        # Stage 1: Domain knowledge - always keep important features
        must_keep_patterns = [
            'char_count', 'word_count', 'sentence_count',
            'processing_signal', 'quality_signal', 'name_change_signal', 'device_signal',
            'keyword', 'banking', 'app', 'account', 'credit'
        ]
        
        must_keep_features = []
        for pattern in must_keep_patterns:
            for col in self.X.columns:
                if pattern in col.lower() and col not in must_keep_features:
                    must_keep_features.append(col)
        
        # Stage 2: Statistical selection for remaining features
        remaining_features = [col for col in self.X.columns if col not in must_keep_features]
        
        if len(remaining_features) > 0:
            # Use multiple selection methods and combine
            selector_mi = SelectKBest(mutual_info_classif, k=min(n_features//2, len(remaining_features)))
            selector_chi2 = SelectKBest(chi2, k=min(n_features//2, len(remaining_features)))
            
            # Apply to remaining features
            X_remaining = self.X_train_scaled[:, [i for i, col in enumerate(self.X.columns) if col in remaining_features]]
            
            try:
                mi_selected = selector_mi.fit_transform(X_remaining, self.y_train)
                mi_features = [remaining_features[i] for i in selector_mi.get_support(indices=True)]
            except:
                mi_features = remaining_features[:n_features//2]
            
            try:
                # Make features non-negative for chi2
                X_remaining_pos = X_remaining - X_remaining.min() + 1e-8
                chi2_selected = selector_chi2.fit_transform(X_remaining_pos, self.y_train)
                chi2_features = [remaining_features[i] for i in selector_chi2.get_support(indices=True)]
            except:
                chi2_features = remaining_features[n_features//2:]
            
            # Combine all selected features
            all_selected = must_keep_features + mi_features + chi2_features
            
            # Remove duplicates and limit to target number
            final_features = list(dict.fromkeys(all_selected))[:n_features]
        else:
            final_features = must_keep_features[:n_features]
        
        # Create feature mask
        self.feature_mask = np.array([col in final_features for col in self.X.columns])
        self.selected_features = final_features
        
        print(f"Selected {len(final_features)} features through intelligent selection")
        print(f"Must-keep features: {len([f for f in final_features if f in must_keep_features])}")
        print(f"Top selected features: {final_features[:10]}")
        
        return self.X_train_scaled[:, self.feature_mask]
        
    def split_data(self, test_size=0.2, val_size=0.15):
        """Data splitting with enhanced sample weighting"""
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
        
        print(f"\nData splits:")
        print(f"Train: {len(self.X_train)} samples ({len(self.X_train)/len(self.X)*100:.1f}%)")
        print(f"Validation: {len(self.X_val)} samples ({len(self.X_val)/len(self.X)*100:.1f}%)")
        print(f"Test: {len(self.X_test)} samples ({len(self.X_test)/len(self.X)*100:.1f}%)")
        
        # Enhanced sample weights with extra boost for problematic classes
        sample_weights = compute_sample_weight('balanced', self.y_train)
        
        # Boost weights for problematic classes based on v5 analysis
        problematic_class_names = list(self.problematic_classes.keys())
        for class_name in problematic_class_names:
            try:
                class_idx = self.label_encoder.transform([class_name])[0]
                mask = self.y_train == class_idx
                boost_factor = 1.0 + self.problematic_classes[class_name]['target_improvement']
                sample_weights[mask] *= boost_factor
                print(f"Boosted {class_name} samples by {boost_factor:.1f}x")
            except:
                pass
        
        self.sample_weights = sample_weights
        
    def apply_targeted_sampling(self, X_train, y_train):
        """V6: Apply SMOTE only to most problematic classes"""
        print("\nApplying targeted sampling to problematic classes...")
        
        # Count samples per class
        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique, counts))
        
        # Target only the worst performing classes for oversampling
        problematic_indices = []
        for class_name in self.problematic_classes.keys():
            try:
                class_idx = self.label_encoder.transform([class_name])[0]
                if class_idx in class_counts:
                    problematic_indices.append(class_idx)
            except:
                pass
        
        if len(problematic_indices) > 0:
            # Create sampling strategy - only oversample problematic classes
            sampling_strategy = {}
            median_count = int(np.median(counts))
            
            for class_idx, count in class_counts.items():
                if class_idx in problematic_indices and count < median_count:
                    # Boost problematic classes to median + bonus
                    sampling_strategy[class_idx] = min(median_count + 5, count * 2)
                else:
                    sampling_strategy[class_idx] = count
            
            try:
                smote = BorderlineSMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=self.random_state,
                    k_neighbors=min(3, min(counts) - 1) if min(counts) > 1 else 1
                )
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                print(f"Targeted SMOTE applied. Shape: {X_resampled.shape}")
                print(f"Problematic classes boosted: {len(problematic_indices)}")
                return X_resampled, y_resampled
            except Exception as e:
                print(f"Targeted SMOTE failed ({str(e)}), using original data...")
                return X_train, y_train
        else:
            print("No problematic classes found for targeted sampling")
            return X_train, y_train
    
    def create_optimized_base_models(self):
        """V6: Create optimized base models based on v5 contribution analysis"""
        print("\nCreating optimized base models based on v5 analysis...")
        
        # V5 contribution analysis showed: KNN (20.7%), NB (19.6%), LR (16.1%) were top performers
        # XGB was only 7.5%, so we'll enhance it and balance the ensemble
        
        self.base_models = {
            # Top performers from v5 - keep strong but regularize
            'knn_optimized': KNeighborsClassifier(
                n_neighbors=5,  # Reduced from default for smaller dataset
                weights='distance',
                metric='manhattan',  # Often better for text features
                n_jobs=-1
            ),
            
            'nb_enhanced': GaussianNB(
                var_smoothing=1e-8  # Reduced smoothing for better performance
            ),
            
            'lr_regularized': LogisticRegression(
                C=0.8,  # Slightly more regularization
                class_weight='balanced',
                max_iter=2000,
                solver='liblinear',  # Better for small datasets
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            # Enhanced gradient boosting models
            'xgb_enhanced': xgb.XGBClassifier(
                n_estimators=200,  # Increased from v5
                max_depth=4,  # Slightly deeper
                min_child_weight=3,  # Reduced for smaller classes
                learning_rate=0.06,  # Slower learning
                subsample=0.75,
                colsample_bytree=0.75,
                reg_alpha=1.5,
                reg_lambda=2.5,
                gamma=0.3,
                random_state=self.random_state,
                verbosity=0,
                eval_metric='mlogloss'
            ),
            
            'lgb_tuned': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.06,
                subsample=0.75,
                colsample_bytree=0.75,
                reg_alpha=1.5,
                reg_lambda=2.5,
                class_weight='balanced',
                random_state=self.random_state,
                verbosity=-1,
                force_row_wise=True,
                feature_pre_filter=False
            ),
            
            'cat_optimized': CatBoostClassifier(
                iterations=200,
                depth=5,
                learning_rate=0.06,
                l2_leaf_reg=2.5,
                bootstrap_type='Bernoulli',
                subsample=0.75,
                class_weights=self.class_weight_dict,
                random_state=self.random_state,
                verbose=False
            ),
            
            # Enhanced tree models
            'rf_tuned': RandomForestClassifier(
                n_estimators=150,
                max_depth=7,  # Deeper for better performance
                min_samples_split=3,  # Reduced for small classes
                min_samples_leaf=1,
                class_weight='balanced_subsample',
                max_features='log2',  # Different from default
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'et_enhanced': ExtraTreesClassifier(
                n_estimators=150,
                max_depth=8,  # Even deeper for extra trees
                min_samples_split=3,
                min_samples_leaf=1,
                class_weight='balanced',
                max_features='log2',
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            # Ensemble model for diversity
            'svm_ensemble': self._create_svm_ensemble()
        }
        
        print(f"Created {len(self.base_models)} optimized base models")
        print("Models prioritized based on v5 contribution analysis")
        
    def _create_svm_ensemble(self):
        """Create SVM ensemble with different kernels"""
        try:
            # Create a voting ensemble of SVMs with different kernels
            svm_linear = SVC(kernel='linear', C=0.5, probability=True, 
                           class_weight='balanced', random_state=self.random_state)
            svm_rbf = SVC(kernel='rbf', C=0.8, gamma='scale', probability=True, 
                         class_weight='balanced', random_state=self.random_state)
            
            return VotingClassifier(
                estimators=[('svm_linear', svm_linear), ('svm_rbf', svm_rbf)],
                voting='soft',
                n_jobs=-1
            )
        except:
            # Fallback to bagging classifier
            return self._create_bagging_classifier()
    
    def _create_bagging_classifier(self):
        """Create BaggingClassifier with version compatibility"""
        base_lr = LogisticRegression(
            C=0.5, 
            class_weight='balanced', 
            max_iter=500,
            random_state=self.random_state
        )
        
        try:
            return BaggingClassifier(
                estimator=base_lr,
                n_estimators=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        except TypeError:
            try:
                return BaggingClassifier(
                    base_estimator=base_lr,
                    n_estimators=10,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            except TypeError:
                return LogisticRegression(
                    C=0.5,
                    class_weight='balanced',
                    max_iter=500,
                    random_state=self.random_state,
                    n_jobs=-1
                )
    
    def train_weighted_stacking_ensemble(self):
        """V6: Train stacking ensemble with weights based on v5 analysis"""
        print("\nTraining weighted stacking ensemble...")
        
        # Prepare data using feature mask
        if hasattr(self, 'feature_mask'):
            X_train = self.X_train_scaled[:, self.feature_mask]
            X_val = self.X_val_scaled[:, self.feature_mask]
        else:
            X_train = self.X_train_scaled
            X_val = self.X_val_scaled
        
        # Apply targeted sampling
        X_train_sampled, y_train_sampled = self.apply_targeted_sampling(X_train, self.y_train)
        
        # Enhanced meta-learner with regularization to reduce overfitting
        self.meta_model = LogisticRegression(
            C=1.5,  # Increased regularization
            class_weight='balanced',
            max_iter=2000,
            solver='liblinear',
            random_state=self.random_state
        )
        
        # Create stacking classifier with optimized CV
        self.stacking_model = StackingClassifier(
            estimators=list(self.base_models.items()),
            final_estimator=self.meta_model,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state),  # Reduced CV to prevent overfitting
            stack_method='predict_proba',
            n_jobs=-1,
            verbose=0
        )
        
        # Train the stacking ensemble
        print("Training weighted stacking classifier with 3-fold CV...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.stacking_model.fit(X_train_sampled, y_train_sampled)
            print("Weighted stacking ensemble training completed!")
        except Exception as e:
            print(f"Stacking training failed: {str(e)}")
            print("Falling back to best single model...")
            
            # Fallback to the best performing model from v5
            self.stacking_model = self.base_models['knn_optimized']
            self.stacking_model.fit(X_train_sampled, y_train_sampled)
            print("Fallback model training completed!")
        
    def dynamic_pseudo_labeling(self, base_confidence=0.88):
        """V6: Dynamic confidence thresholding based on class performance"""
        print(f"\nImplementing dynamic pseudo-labeling (base confidence: {base_confidence})...")
        
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
        
        # Dynamic confidence thresholding
        high_confidence_samples = []
        confidence_thresholds = {}
        
        # Lower threshold for problematic classes, higher for others
        for i, class_name in enumerate(self.class_names):
            if class_name in self.problematic_classes:
                # Lower threshold for problematic classes to get more pseudo-labels
                confidence_thresholds[i] = base_confidence - 0.05
            else:
                # Higher threshold for well-performing classes
                confidence_thresholds[i] = base_confidence + 0.02
        
        # Find high-confidence predictions with dynamic thresholds
        for i, (pred_class, max_prob) in enumerate(zip(y_val_pred, np.max(y_val_proba, axis=1))):
            threshold = confidence_thresholds.get(pred_class, base_confidence)
            if max_prob >= threshold:
                high_confidence_samples.append(i)
        
        if len(high_confidence_samples) > 0:
            print(f"Found {len(high_confidence_samples)} high-confidence pseudo-labels with dynamic thresholds")
            
            # Get high-confidence pseudo-labels
            X_pseudo = X_val[high_confidence_samples]
            y_pseudo = y_val_pred[high_confidence_samples]
            
            # Combine with original training data
            if hasattr(self, 'feature_mask'):
                X_train_orig = self.X_train_scaled[:, self.feature_mask]
            else:
                X_train_orig = self.X_train_scaled
            
            X_combined = np.vstack([X_train_orig, X_pseudo])
            y_combined = np.hstack([self.y_train, y_pseudo])
            
            # Retrain the model on combined data
            print("Retraining model with dynamic pseudo-labeled data...")
            
            # Apply targeted sampling to combined data
            X_final, y_final = self.apply_targeted_sampling(X_combined, y_combined)
            
            # Retrain stacking model
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.stacking_model.fit(X_final, y_final)
            
            print("Dynamic pseudo-labeling completed!")
            return True
        else:
            print("No high-confidence predictions found for dynamic pseudo-labeling")
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
        
        # V6: Analyze improvements in problematic classes
        print("\nProblematic Classes Improvement Analysis:")
        for class_name, metrics in self.problematic_classes.items():
            try:
                class_idx = self.label_encoder.transform([class_name])[0]
                current_f1 = results['Test']['per_class_f1'][class_idx]
                v5_f1 = metrics['f1']
                improvement = current_f1 - v5_f1
                target_improvement = metrics['target_improvement']
                
                status = "Achieved" if improvement >= target_improvement else "Improved" if improvement > 0 else "No Change"
                print(f"  {class_name}: {v5_f1:.3f} → {current_f1:.3f} (Δ{improvement:+.3f}) - {status}")
            except:
                pass
        
        # Confidence analysis
        avg_confidence = np.mean(np.max(y_proba_test, axis=1))
        print(f"\nAverage prediction confidence: {avg_confidence:.3f}")
        
        return results
    
    def plot_confusion_matrix(self, cm, class_names, save_timestamp):
        """Create clear matplotlib confusion matrix"""
        plt.figure(figsize=(16, 12))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'},
                    square=True, linewidths=0.5, annot_kws={'size': 10})
        
        plt.title('Test Set Confusion Matrix', fontsize=20, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=16, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'v7_confusion_matrix_{save_timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def advanced_cross_validation(self, cv_folds=5, repeats=2):
        """Advanced cross-validation with reduced overfitting"""
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
    
    def save_comprehensive_results(self, results, save_timestamp):
        """Save comprehensive results with all details"""
        try:
            # Save the stacking ensemble
            joblib.dump(self.stacking_model, f'v7_ultimate_model_{save_timestamp}.pkl')
            if hasattr(self, 'feature_mask'):
                joblib.dump(self.feature_mask, f'v7_feature_mask_{save_timestamp}.pkl')
            joblib.dump(self.scaler, f'v7_scaler_{save_timestamp}.pkl')
            
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
            
            test_predictions.to_csv(f'v7_detailed_predictions_{save_timestamp}.csv', index=False)
            
            print(f"\nResults saved with timestamp: {save_timestamp}")
            return test_predictions
            
        except Exception as e:
            print(f"Could not save results: {str(e)}")
            return None


def main():
    """Main execution for V7 - Clear output version"""
    try:
        print("v7 Enhanced Stacking Email Classifier")
        print("Strategy: Weighted Stacking + Dynamic Pseudo-labeling + Class-specific Engineering\n")
        
        # Initialize classifier
        classifier = UltimateStackingClassifierV7(random_state=42) # Set 42 for reproducibility / None for randomness
        
        # Load data
        classifier.load_data('ml_dataset_v5_full.xlsx')
        
        # Advanced feature engineering with class-specific improvements
        classifier.advanced_feature_engineering()
        
        # Split data
        print("About to split data...")
        classifier.split_data(test_size=0.2, val_size=0.15)
        print("Data splitting completed!")
        
        # Intelligent feature selection with domain knowledge
        classifier.intelligent_feature_selection(n_features=20)
        
        # Create optimized base models
        classifier.create_optimized_base_models()
        
        # Train weighted stacking ensemble
        classifier.train_weighted_stacking_ensemble()
        
        # Apply dynamic pseudo-labeling
        pseudo_success = classifier.dynamic_pseudo_labeling(base_confidence=0.88)
        
        # Comprehensive evaluation
        results = classifier.evaluate_comprehensive()
        
        # Generate timestamp for consistent file naming
        save_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create and display confusion matrix
        cm = confusion_matrix(results['Test']['y_true'], results['Test']['y_pred'])
        print("\nGenerating Confusion Matrix...")
        classifier.plot_confusion_matrix(cm, classifier.class_names, save_timestamp)
        
        # Save comprehensive results
        test_predictions = classifier.save_comprehensive_results(results, save_timestamp)
        
        # Final summary
        print("\n" + "="*70)
        print("V7 ENHANCED STACKING RESULTS")
        print("="*70)
        print(f"Test Accuracy: {results['Test']['accuracy']:.3f}")
        print(f"Test Balanced Accuracy: {results['Test']['balanced_accuracy']:.3f}")
        print(f"Test Macro-F1: {results['Test']['macro_f1']:.3f}")
        print(f"Test Weighted-F1: {results['Test']['weighted_f1']:.3f}")
        print(f"Test Cohen's Kappa: {results['Test']['kappa']:.3f}")
        
        print(f"\nOverfitting:")
        print(f"  Train-Test Macro-F1 Gap: {results['Train']['macro_f1'] - results['Test']['macro_f1']:.3f}")
        
        print(f"\nResults saved with timestamp: {save_timestamp}")
        print("\nv7 Enhanced Stacking completed!")
        
        return classifier, results
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
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