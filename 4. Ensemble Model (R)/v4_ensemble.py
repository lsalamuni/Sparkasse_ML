#!/usr/bin/env python3
"""
Enhanced XGBoost Email Classifier v4 - German Banking Emails
Goal: Boost accuracy to 80%+ while maintaining low overfitting

Key Improvements:
- Optimized feature selection (20-25 features)
- Smart SMOTE for minority classes only
- Enhanced ensemble with CatBoost and LightGBM
- Advanced feature engineering
- Class-specific optimization
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
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import joblib
warnings.filterwarnings('ignore')


class OptimizedXGBoostV4:
    """V4: Optimized for higher accuracy with maintained low overfitting"""
    
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
        self.engineered_features = None
        
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
            
            # Remove only constant features (not binary - they might be useful)
            constant_features = self.X.columns[self.X.nunique() == 1]
            if len(constant_features) > 0:
                print(f"Removing {len(constant_features)} constant features...")
                self.X = self.X.drop(columns=constant_features)
            
            # Remove only extremely correlated features (>0.98)
            corr_matrix = self.X.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            high_corr_features = [column for column in upper_triangle.columns 
                                if any(upper_triangle[column] > 0.98)]
            
            if len(high_corr_features) > 0:
                print(f"Removing {len(high_corr_features)} highly correlated features...")
                self.X = self.X.drop(columns=high_corr_features)
            
            # Encode categories
            self.y_encoded = self.label_encoder.fit_transform(self.y)
            self.n_classes = len(np.unique(self.y_encoded))
            self.class_names = self.label_encoder.classes_
            
            print(f"Dataset shape after preprocessing: {self.X.shape}")
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
            if count < 20:  # Classes with less than 20 samples
                self.minority_classes.append(self.label_encoder.transform([cat])[0])
        
        # Calculate class weights
        self.class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.y_encoded),
            y=self.y_encoded
        )
        self.class_weight_dict = dict(enumerate(self.class_weights))
        
        # Identify problematic classes from v3 results
        self.problematic_classes = {
            'C - Bearbeitungsdauer': 2,
            'E - Gerätewechsel': 4,
            'K - Servicequalität': 10
        }
        
        print(f"\nMinority classes (<20 samples): {len(self.minority_classes)}")
        print(f"Problematic classes (low F1 in v3): {list(self.problematic_classes.keys())}")
        
    def engineer_advanced_features(self):
        """Create advanced features based on existing ones"""
        print("\nEngineering advanced features...")
        
        X_eng = self.X.copy()
        
        # 1. Ratio features
        if 'char_count' in X_eng.columns and 'word_count' in X_eng.columns:
            X_eng['char_per_word'] = X_eng['char_count'] / (X_eng['word_count'] + 1)
        
        if 'total_punctuation' in X_eng.columns and 'char_count' in X_eng.columns:
            X_eng['punct_density'] = X_eng['total_punctuation'] / (X_eng['char_count'] + 1)
        
        if 'capital_words' in X_eng.columns and 'word_count' in X_eng.columns:
            X_eng['capital_ratio'] = X_eng['capital_words'] / (X_eng['word_count'] + 1)
        
        # 2. Banking keyword combinations
        banking_cols = [col for col in X_eng.columns if 'keywords' in col]
        if len(banking_cols) > 1:
            X_eng['total_banking_keywords'] = X_eng[banking_cols].sum(axis=1)
            X_eng['banking_keyword_diversity'] = (X_eng[banking_cols] > 0).sum(axis=1)
        
        # 3. Urgency indicators
        if 'exclamation_marks' in X_eng.columns and 'question_marks' in X_eng.columns:
            X_eng['urgency_score'] = X_eng['exclamation_marks'] + X_eng['question_marks'] * 2
        
        # 4. Length-based features
        if 'char_count' in X_eng.columns:
            X_eng['is_short_email'] = (X_eng['char_count'] < X_eng['char_count'].quantile(0.25)).astype(int)
            X_eng['is_long_email'] = (X_eng['char_count'] > X_eng['char_count'].quantile(0.75)).astype(int)
        
        # 5. Category-specific features for problematic classes
        # For "Bearbeitungsdauer" (processing time)
        if 'word_count' in X_eng.columns:
            time_words = ['dauer', 'zeit', 'warten', 'lange']  # German time-related words
            X_eng['time_related_score'] = 0  # Initialize
            
        # For "Servicequalität" (service quality)
        quality_indicators = ['service', 'qualität', 'beschwerde', 'unzufrieden']
        X_eng['quality_complaint_score'] = 0  # Initialize
        
        print(f"Created {len(X_eng.columns) - len(self.X.columns)} new features")
        self.X = X_eng
        
    def optimized_feature_selection(self, n_features=25):
        """Optimized feature selection - less aggressive than v3"""
        print(f"\nPerforming optimized feature selection (target: {n_features} features)...")
        
        # Use RFECV for optimal number of features
        base_estimator = ExtraTreesClassifier(
            n_estimators=100,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Try RFECV with cross-validation
        try:
            selector = RFECV(
                base_estimator,
                min_features_to_select=15,
                cv=3,
                scoring='f1_macro',
                n_jobs=-1
            )
            selector.fit(self.X_train_scaled, self.y_train)
            optimal_features = selector.n_features_
            print(f"RFECV selected optimal features: {optimal_features}")
            
            # If RFECV selects too many/few, use fixed number
            if optimal_features < 20 or optimal_features > 30:
                print(f"Adjusting to target {n_features} features...")
                selector = SelectKBest(mutual_info_classif, k=n_features)
                selector.fit(self.X_train_scaled, self.y_train)
        except Exception as e:
            # Fallback to SelectKBest
            print(f"RFECV failed ({str(e)}), using SelectKBest...")
            selector = SelectKBest(mutual_info_classif, k=n_features)
            selector.fit(self.X_train_scaled, self.y_train)
        
        self.feature_selector = selector
        self.selected_features = self.X.columns[selector.get_support()].tolist()
        
        print(f"Selected {len(self.selected_features)} features")
        print(f"Selected features include: {self.selected_features[:10]}...")
        
        return selector.transform(self.X_train_scaled)
        
    def split_data(self, test_size=0.2, val_size=0.15):
        """Data splitting with same structure as previous versions"""
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
        
        # Calculate sample weights with extra weight for problematic classes
        sample_weights = compute_sample_weight('balanced', self.y_train)
        
        # Boost weights for problematic classes
        for class_name, class_idx in self.problematic_classes.items():
            mask = self.y_train == class_idx
            sample_weights[mask] *= 1.5  # 50% extra weight
        
        self.sample_weights = sample_weights
        
    def apply_smart_smote(self, X_train, y_train):
        """Apply SMOTE only to minority classes"""
        print("\nApplying smart SMOTE to minority classes...")
        
        # Count samples per class
        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique, counts))
        
        # Determine sampling strategy - only oversample minority classes
        sampling_strategy = {}
        target_samples = int(np.median(counts))  # Target: median class size
        
        for class_idx, count in class_counts.items():
            if class_idx in self.minority_classes and count < target_samples:
                sampling_strategy[class_idx] = target_samples
            else:
                sampling_strategy[class_idx] = count
        
        # Apply BorderlineSMOTE (better for edge cases)
        try:
            smote = BorderlineSMOTE(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                k_neighbors=3
            )
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print(f"BorderlineSMOTE applied. New shape: {X_resampled.shape}")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"BorderlineSMOTE failed ({str(e)}), using regular SMOTE...")
            try:
                smote = SMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=self.random_state,
                    k_neighbors=3
                )
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                print(f"Regular SMOTE applied. New shape: {X_resampled.shape}")
                return X_resampled, y_resampled
            except Exception as e2:
                print(f"All SMOTE methods failed ({str(e2)}). Using original data...")
                return X_train, y_train
    
    def create_advanced_ensemble(self):
        """Create ensemble with multiple gradient boosting algorithms"""
        print("\nCreating advanced ensemble...")
        
        # Prepare data - keep everything as numpy arrays for consistency
        if self.feature_selector:
            X_train = self.feature_selector.transform(self.X_train_scaled)
            X_val = self.feature_selector.transform(self.X_val_scaled)
        else:
            X_train = self.X_train_scaled
            X_val = self.X_val_scaled
        
        # Apply SMOTE
        X_train_smote, y_train_smote = self.apply_smart_smote(X_train, self.y_train)
        
        print(f"Data shapes: X_train={X_train.shape}, X_train_smote={X_train_smote.shape}")
        print(f"Data types: X_train={type(X_train)}, X_train_smote={type(X_train_smote)}")
        
        # Model 1: XGBoost with moderate regularization
        xgb_model = xgb.XGBClassifier(
            max_depth=4,
            min_child_weight=3,
            reg_alpha=1.0,
            reg_lambda=2.0,
            gamma=0.3,
            subsample=0.7,
            colsample_bytree=0.7,
            learning_rate=0.1,
            n_estimators=200,
            objective='multi:softprob',
            n_jobs=-1,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0
        )
        
        # Model 2: LightGBM (handles imbalanced data well)
        lgb_model = lgb.LGBMClassifier(
            boosting_type='dart',  # DART for better generalization
            num_leaves=31,
            max_depth=5,
            learning_rate=0.1,
            n_estimators=200,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1
        )
        
        # Model 3: CatBoost (robust to overfitting)
        cat_model = CatBoostClassifier(
            iterations=200,
            depth=4,
            learning_rate=0.1,
            l2_leaf_reg=3.0,
            border_count=32,
            bootstrap_type='Bernoulli',  # Required for subsample
            subsample=0.8,
            random_strength=1.0,
            class_weights=self.class_weight_dict,
            random_state=self.random_state,
            verbose=False
        )
        
        # Model 4: Random Forest with balanced settings
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Model 5: Extra Trees for diversity
        et_model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Train models on SMOTE data
        print("Training individual models...")
        models = [
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('cat', cat_model),
            ('rf', rf_model),
            ('et', et_model)
        ]
        
        trained_models = []
        for name, model in models:
            print(f"Training {name}...")
            
            # Train all models on numpy arrays for consistency
            model.fit(X_train_smote, y_train_smote)
            
            # Calibrate probabilities for better ensemble performance
            try:
                calibrated = CalibratedClassifierCV(
                    model, method='isotonic', cv=3, n_jobs=-1
                )
            except TypeError:
                # Some sklearn versions don't support n_jobs in CalibratedClassifierCV
                calibrated = CalibratedClassifierCV(
                    model, method='isotonic', cv=3
                )
            
            # Calibrate on original training data
            calibrated.fit(X_train, self.y_train)
            trained_models.append((name + '_cal', calibrated))
        
        # Create ensemble with calibrated models
        self.ensemble_model = VotingClassifier(
            estimators=trained_models,
            voting='soft',
            n_jobs=-1
        )
        
        # Fit the VotingClassifier on original training data
        # Even though individual models are trained, VotingClassifier needs to be fitted
        print("Fitting ensemble...")
        try:
            self.ensemble_model.fit(X_train, self.y_train)
            print("Advanced ensemble training completed!")
        except Exception as e:
            print(f"Ensemble fitting failed: {str(e)}")
            print("Creating simple ensemble without VotingClassifier...")
            # Fallback: use first model as main model
            self.ensemble_model = trained_models[0][1]
            print("Using XGBoost as fallback model")
    
    def evaluate_comprehensive(self):
        """Comprehensive evaluation with detailed analysis"""
        print("\nEvaluating model performance...")
        
        # Prepare data
        if self.feature_selector:
            X_train = self.feature_selector.transform(self.X_train_scaled)
            X_val = self.feature_selector.transform(self.X_val_scaled)
            X_test = self.feature_selector.transform(self.X_test_scaled)
        else:
            X_train = self.X_train_scaled
            X_val = self.X_val_scaled
            X_test = self.X_test_scaled
        
        # Predictions
        y_pred_train = self.ensemble_model.predict(X_train)
        y_pred_val = self.ensemble_model.predict(X_val)
        y_pred_test = self.ensemble_model.predict(X_test)
        
        # Get prediction probabilities for analysis
        y_proba_test = self.ensemble_model.predict_proba(X_test)
        
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
        
        # Analyze problematic classes
        print("\nProblematic Classes Analysis:")
        for class_name, class_idx in self.problematic_classes.items():
            f1 = results['Test']['per_class_f1'][class_idx]
            improvement = f1 - 0.4  # Baseline from v3 results
            print(f"{class_name}: F1={f1:.3f} (improvement: {improvement:+.3f})")
        
        # Confidence analysis
        avg_confidence = np.mean(np.max(y_proba_test, axis=1))
        print(f"\nAverage prediction confidence: {avg_confidence:.3f}")
        
        return results
    
    def plot_enhanced_analysis(self, results):
        """Enhanced visualizations for v4"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Confusion Matrix
            cm = confusion_matrix(results['Test']['y_true'], results['Test']['y_pred'])
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(
                cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=axes[0, 0]
            )
            axes[0, 0].set_title('Test Set Confusion Matrix (Normalized)')
            axes[0, 0].set_ylabel('True Label')
            axes[0, 0].set_xlabel('Predicted Label')
            
            # Plot 2: Per-Class F1 Scores
            per_class_df = pd.DataFrame({
                'Class': self.class_names,
                'F1-Score': results['Test']['per_class_f1']
            }).sort_values('F1-Score', ascending=True)
            
            colors = ['red' if f1 < 0.5 else 'orange' if f1 < 0.7 else 'green' 
                     for f1 in per_class_df['F1-Score']]
            
            axes[0, 1].barh(per_class_df['Class'], per_class_df['F1-Score'], color=colors)
            axes[0, 1].set_xlabel('F1-Score')
            axes[0, 1].set_title('Per-Class Performance')
            axes[0, 1].axvline(x=0.7, color='black', linestyle='--', alpha=0.5)
            
            # Plot 3: Train vs Test Performance
            metrics = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1']
            train_scores = [results['Train'][m] for m in metrics]
            test_scores = [results['Test'][m] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, train_scores, width, label='Train', alpha=0.8)
            axes[1, 0].bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
            axes[1, 0].set_xlabel('Metric')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title('Train vs Test Performance')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(metrics, rotation=45)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Version Comparison
            versions_data = {
                'v2': {'accuracy': 0.680, 'macro_f1': 0.651, 'gap': 0.336},
                'v3': {'accuracy': 0.740, 'macro_f1': 0.713, 'gap': 0.124},
                'v4': {
                    'accuracy': results['Test']['accuracy'],
                    'macro_f1': results['Test']['macro_f1'],
                    'gap': results['Train']['macro_f1'] - results['Test']['macro_f1']
                }
            }
            
            versions = list(versions_data.keys())
            accuracies = [versions_data[v]['accuracy'] for v in versions]
            f1_scores = [versions_data[v]['macro_f1'] for v in versions]
            
            x = np.arange(len(versions))
            axes[1, 1].bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
            axes[1, 1].bar(x + width/2, f1_scores, width, label='Macro-F1', color='lightcoral')
            axes[1, 1].set_xlabel('Version')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Version Performance Comparison')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(versions)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Could not display plots: {str(e)}")
    
    def save_v4_results(self, results):
        """Save comprehensive results for v4"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model
            joblib.dump(self.ensemble_model, f'v4_ensemble_model_{timestamp}.pkl')
            joblib.dump(self.feature_selector, f'v4_feature_selector_{timestamp}.pkl')
            joblib.dump(self.scaler, f'v4_scaler_{timestamp}.pkl')
            
            # Save predictions with confidence
            y_proba_test = self.ensemble_model.predict_proba(
                self.feature_selector.transform(self.X_test_scaled) 
                if self.feature_selector else self.X_test_scaled
            )
            
            test_predictions = pd.DataFrame({
                'document': self.df.iloc[self.X_test.index]['document'],
                'true_category': self.label_encoder.inverse_transform(self.y_test),
                'predicted_category': self.label_encoder.inverse_transform(results['Test']['y_pred']),
                'confidence': np.max(y_proba_test, axis=1),
                'correct': self.y_test == results['Test']['y_pred']
            })
            
            # Add top 3 predictions
            top3_indices = np.argsort(y_proba_test, axis=1)[:, -3:][:, ::-1]
            for i in range(3):
                test_predictions[f'top{i+1}_prediction'] = self.label_encoder.inverse_transform(top3_indices[:, i])
                test_predictions[f'top{i+1}_confidence'] = y_proba_test[np.arange(len(y_proba_test)), top3_indices[:, i]]
            
            test_predictions.to_csv(f'v4_predictions_{timestamp}.csv', index=False)
            
            # Save metrics
            metrics_summary = pd.DataFrame({
                'Version': ['v2', 'v3', 'v4'],
                'Test_Accuracy': [0.680, 0.740, results['Test']['accuracy']],
                'Test_Macro_F1': [0.651, 0.713, results['Test']['macro_f1']],
                'Train_Test_Gap': [0.336, 0.124, results['Train']['macro_f1'] - results['Test']['macro_f1']],
                'Features_Used': ['25', '12', str(len(self.selected_features))]
            })
            metrics_summary.to_csv(f'v4_version_comparison_{timestamp}.csv', index=False)
            
            print(f"\nResults saved with timestamp: {timestamp}")
            return test_predictions, metrics_summary
            
        except Exception as e:
            print(f"Could not save results: {str(e)}")
            return None, None


def main():
    """Main execution for v4"""
    try:
        print("Starting v4 Enhanced XGBoost Email Classifier...")
        print("Goal: 80%+ accuracy with maintained low overfitting\n")
        
        # Initialize classifier
        classifier = OptimizedXGBoostV4(random_state=42)
        
        # Load data
        classifier.load_data('ml_dataset_v5_full.xlsx')
        
        # Engineer advanced features
        classifier.engineer_advanced_features()
        
        # Split data
        print("\nAbout to split data...")
        classifier.split_data(test_size=0.2, val_size=0.15)
        print("Data splitting completed!")
        
        # Optimized feature selection (25 features)
        classifier.optimized_feature_selection(n_features=25)
        
        # Create and train advanced ensemble
        classifier.create_advanced_ensemble()
        
        # Comprehensive evaluation
        results = classifier.evaluate_comprehensive()
        
        # Enhanced visualizations
        classifier.plot_enhanced_analysis(results)
        
        # Save results
        test_predictions, metrics_summary = classifier.save_v4_results(results)
        
        # Final summary
        print("\n" + "="*70)
        print("V4 ENHANCED RESULTS SUMMARY")
        print("="*70)
        print(f"Test Accuracy: {results['Test']['accuracy']:.3f}")
        print(f"Test Balanced Accuracy: {results['Test']['balanced_accuracy']:.3f}")
        print(f"Test Macro-F1: {results['Test']['macro_f1']:.3f}")
        print(f"Test Weighted-F1: {results['Test']['weighted_f1']:.3f}")
        print(f"Test Cohen's Kappa: {results['Test']['kappa']:.3f}")
        
        print(f"\nOverfitting:")
        print(f"  Train-Test Macro-F1 Gap: {results['Train']['macro_f1'] - results['Test']['macro_f1']:.3f}")
        
        # Version improvements
        v3_accuracy = 0.740
        v4_accuracy = results['Test']['accuracy']
        improvement = v4_accuracy - v3_accuracy
        
        print(f"\nImprovement over v3:")
        print(f"  v3 Accuracy: {v3_accuracy:.3f}")
        print(f"  v4 Accuracy: {v4_accuracy:.3f}")
        print(f"  Improvement: {improvement:+.3f} ({improvement/v3_accuracy*100:+.1f}%)")
        
        target_reached = v4_accuracy >= 0.80
        print(f"\n{'🎯' if target_reached else '📊'} Target 80% accuracy: {'ACHIEVED!' if target_reached else f'Not yet ({(0.80 - v4_accuracy)*100:.1f}% to go)'}")
        
        print(f"\n✅ v4 Enhanced XGBoost completed successfully!")
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