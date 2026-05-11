"""
Wildfire Survival Analysis Models
WiDS Global Datathon 2026

This module implements survival models for wildfire threat prediction:
1. XGBoost with survival objectives (Cox and AFT)
2. Random Survival Forests
3. Custom evaluation metrics (Hybrid Score)
4. Probability calibration
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Try to import scikit-survival for Random Survival Forests
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import concordance_index_censored, brier_score, integrated_brier_score
    SKSURV_AVAILABLE = True
except ImportError:
    print("Warning: scikit-survival not available. Some features will be limited.")
    SKSURV_AVAILABLE = False

class WildfireSurvivalModels:
    def __init__(self, model_type='xgb_cox', random_state=42):
        """
        Initialize survival models
        
        Args:
            model_type: Type of model ('xgb_cox', 'xgb_aft', 'rsf')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.calibrator = None
        
    def prepare_survival_data(self, df):
        """
        Prepare data in survival format
        
        Args:
            df: DataFrame with features and survival targets
            
        Returns:
            X: Feature matrix
            y: Survival data (time, event)
        """
        # Extract features (exclude target and identifier columns)
        exclude_cols = ['event_id', 'time_to_hit_hours', 'event']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()
        
        # Prepare survival targets
        y_time = df['time_to_hit_hours'].values
        y_event = df['event'].values
        
        return X, y_time, y_event, feature_cols
    
    def train_xgb_cox(self, X, y_time, y_event, **params):
        """Train XGBoost with Cox proportional hazards objective"""
        print("Training XGBoost Cox model...")
        
        # Default parameters for Cox survival
        default_params = {
            'objective': 'survival:cox',
            'eval_metric': 'cox-nloglik',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        }
        
        # Update with provided parameters
        params = {**default_params, **params}
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X, label=y_time)
        
        # Set up evaluation data (use a small holdout for early stopping)
        eval_size = min(50, len(X) // 5)
        eval_indices = np.random.choice(len(X), eval_size, replace=False)
        train_indices = np.setdiff1d(np.arange(len(X)), eval_indices)
        
        dtrain_full = xgb.DMatrix(X.iloc[train_indices], label=y_time[train_indices])
        deval = xgb.DMatrix(X.iloc[eval_indices], label=y_time[eval_indices])
        
        # Train model with early stopping
        self.model = xgb.train(
            params,
            dtrain_full,
            num_boost_round=params['n_estimators'],
            evals=[(deval, 'eval')],
            early_stopping_rounds=20,
            verbose_eval=50
        )
        
        print(f"XGBoost Cox trained. Best iteration: {self.model.best_iteration}")
        return self.model
    
    def train_xgb_aft(self, X, y_time, y_event, **params):
        """Train XGBoost with Accelerated Failure Time objective"""
        print("Training XGBoost AFT model...")
        
        # Default parameters for AFT survival
        default_params = {
            'objective': 'survival:aft',
            'eval_metric': 'aft-nloglik',
            'aft_loss_distribution': 'normal',
            'aft_sigma': 1.0,
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        }
        
        # Update with provided parameters
        params = {**default_params, **params}
        
        # Create DMatrix with survival data
        dtrain = xgb.DMatrix(X, label=y_time)
        
        # Set up evaluation data
        eval_size = min(50, len(X) // 5)
        eval_indices = np.random.choice(len(X), eval_size, replace=False)
        train_indices = np.setdiff1d(np.arange(len(X)), eval_indices)
        
        dtrain_full = xgb.DMatrix(X.iloc[train_indices], label=y_time[train_indices])
        deval = xgb.DMatrix(X.iloc[eval_indices], label=y_time[eval_indices])
        
        # Train model
        self.model = xgb.train(
            params,
            dtrain_full,
            num_boost_round=params['n_estimators'],
            evals=[(deval, 'eval')],
            early_stopping_rounds=20,
            verbose_eval=50
        )
        
        print(f"XGBoost AFT trained. Best iteration: {self.model.best_iteration}")
        return self.model
    
    def train_rsf(self, X, y_time, y_event, **params):
        """Train Random Survival Forest (requires scikit-survival)"""
        if not SKSURV_AVAILABLE:
            raise ImportError("scikit-survival is required for Random Survival Forests")
        
        print("Training Random Survival Forest...")
        
        # Convert to structured array for scikit-survival
        y_structured = np.array([(bool(event), time) for event, time in zip(y_event, y_time)],
                               dtype=[('event', bool), ('time', float)])
        
        # Default parameters for RSF
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 10,
            'min_samples_leaf': 15,
            'max_features': 'sqrt',
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        # Update with provided parameters
        params = {**default_params, **params}
        
        # Train model
        self.model = RandomSurvivalForest(**params)
        self.model.fit(X, y_structured)
        
        print("Random Survival Forest trained")
        return self.model
    
    def predict_risk_scores(self, X):
        """Predict risk scores (higher = higher risk)"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.model_type in ['xgb_cox', 'xgb_aft']:
            # XGBoost models
            dmatrix = xgb.DMatrix(X)
            if self.model_type == 'xgb_cox':
                # Cox: higher risk score = higher hazard
                risk_scores = self.model.predict(dmatrix)
            else:
                # AFT: need to convert to risk scores
                # For AFT, negative predicted time = higher risk
                risk_scores = -self.model.predict(dmatrix)
        
        elif self.model_type == 'rsf':
            # Random Survival Forest
            risk_scores = self.model.predict(X)
        
        return risk_scores
    
    def predict_survival_probabilities(self, X, time_points=[12, 24, 48, 72]):
        """
        Predict survival probabilities at specific time points
        
        Args:
            X: Feature matrix
            time_points: Time points in hours for probability prediction
            
        Returns:
            survival_probs: Array of survival probabilities (n_samples, n_time_points)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        n_samples = len(X)
        n_time_points = len(time_points)
        survival_probs = np.zeros((n_samples, n_time_points))
        
        if self.model_type in ['xgb_cox', 'xgb_aft']:
            # For XGBoost, we need to estimate survival curves
            # This is a simplified approach - in practice you'd use more sophisticated methods
            risk_scores = self.predict_risk_scores(X)
            
            # Normalize risk scores to create survival probabilities
            # This is a simplified baseline approach
            for i, time_point in enumerate(time_points):
                if self.model_type == 'xgb_cox':
                    # Cox: S(t) = exp(-cumulative_hazard * risk_score)
                    # Simplified: use exponential decay based on risk
                    baseline_survival = np.exp(-time_point / 48)  # Baseline survival at 48h
                    survival_probs[:, i] = baseline_survival ** (risk_scores / np.median(risk_scores + 1e-6))
                else:
                    # AFT: simplified approach
                    # Higher risk = lower survival probability
                    normalized_risk = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min() + 1e-6)
                    survival_probs[:, i] = 1 - normalized_risk * (time_point / 72)
        
        elif self.model_type == 'rsf':
            if SKSURV_AVAILABLE:
                # Random Survival Forest can predict survival functions
                survival_functions = self.model.predict_survival_function(X)
                
                for i, time_point in enumerate(time_points):
                    # Extract survival probability at each time point
                    for j, surv_func in enumerate(survival_functions):
                        # Find closest time point in the survival function
                        times = surv_func.x
                        if len(times) > 0:
                            # Interpolate to get survival at desired time point
                            if time_point <= times[-1]:
                                survival_probs[j, i] = np.interp(time_point, times, surv_func.y)
                            else:
                                survival_probs[j, i] = surv_func.y[-1]  # Use last known value
                        else:
                            survival_probs[j, i] = 1.0  # Default to full survival
            else:
                # Fallback if scikit-survival not available
                risk_scores = self.predict_risk_scores(X)
                normalized_risk = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min() + 1e-6)
                for i, time_point in enumerate(time_points):
                    survival_probs[:, i] = 1 - normalized_risk * (time_point / 72)
        
        # Ensure probabilities are in valid range [0, 1]
        survival_probs = np.clip(survival_probs, 0, 1)
        
        return survival_probs
    
    def calculate_concordance_index(self, X, y_time, y_event):
        """Calculate concordance index (C-index)"""
        risk_scores = self.predict_risk_scores(X)
        
        if SKSURV_AVAILABLE:
            # Use scikit-survival implementation
            cindex, _ = concordance_index_censored(y_event.astype(bool), y_time, -risk_scores)
            return cindex
        else:
            # Simple concordance calculation
            n_correct = 0
            n_comparable = 0
            
            for i in range(len(y_time)):
                for j in range(i+1, len(y_time)):
                    if y_time[i] != y_time[j]:
                        # Check if pairs are comparable
                        if (y_event[i] and y_time[i] <= y_time[j]) or (y_event[j] and y_time[j] <= y_time[i]):
                            n_comparable += 1
                            
                            # Check if ordering is correct
                            if y_event[i] and not y_event[j]:
                                if risk_scores[i] > risk_scores[j]:
                                    n_correct += 1
                            elif not y_event[i] and y_event[j]:
                                if risk_scores[i] < risk_scores[j]:
                                    n_correct += 1
                            elif y_event[i] and y_event[j]:
                                if (y_time[i] < y_time[j] and risk_scores[i] > risk_scores[j]) or \
                                   (y_time[i] > y_time[j] and risk_scores[i] < risk_scores[j]):
                                    n_correct += 1
            
            return n_correct / n_comparable if n_comparable > 0 else 0.5
    
    def calculate_brier_score(self, X, y_time, y_event, time_points=[24, 48, 72], weights=[0.3, 0.4, 0.3]):
        """
        Calculate weighted Brier score for specific time points
        
        Args:
            X: Feature matrix
            y_time: Survival times
            y_event: Event indicators
            time_points: Time points to evaluate
            weights: Weights for each time point
            
        Returns:
            weighted_brier_score: Weighted Brier score
        """
        survival_probs = self.predict_survival_probabilities(X, time_points)
        
        brier_scores = []
        
        for i, time_point in enumerate(time_points):
            # Get survival probability at this time point
            surv_prob = survival_probs[:, i]
            
            # Calculate Brier score at this time point
            # Brier score = (1/N) * sum((predicted - actual)^2)
            # where actual = 1 if event occurred before time_point, 0 otherwise
            
            # Determine actual outcomes at this time point
            actual = ((y_time <= time_point) & y_event).astype(float)
            
            # Handle censoring: exclude censored cases that occur after time_point
            # For proper Brier score calculation, we'd need inverse probability weighting
            # This is a simplified version
            mask = (y_time > time_point) | y_event  # Include all uncensored and events before time_point
            
            if mask.sum() > 0:
                brier = np.mean((surv_prob[mask] - actual[mask]) ** 2)
                brier_scores.append(brier)
            else:
                brier_scores.append(0.5)  # Default to neutral score
        
        # Calculate weighted Brier score
        weighted_brier = np.average(brier_scores, weights=weights)
        
        return weighted_brier, brier_scores
    
    def calculate_hybrid_score(self, X, y_time, y_event, time_points=[24, 48, 72], 
                              weights=[0.3, 0.4, 0.3], cindex_weight=0.3, brier_weight=0.7):
        """
        Calculate Hybrid Score: 0.3 * C-index + 0.7 * (1 - Weighted Brier Score)
        
        Args:
            X: Feature matrix
            y_time: Survival times
            y_event: Event indicators
            time_points: Time points for Brier score calculation
            weights: Weights for Brier score time points
            cindex_weight: Weight for C-index component
            brier_weight: Weight for Brier score component
            
        Returns:
            hybrid_score: Final hybrid score
            components: Dictionary with individual components
        """
        # Calculate C-index
        cindex = self.calculate_concordance_index(X, y_time, y_event)
        
        # Calculate weighted Brier score
        weighted_brier, individual_brier_scores = self.calculate_brier_score(
            X, y_time, y_event, time_points, weights
        )
        
        # Calculate hybrid score
        hybrid_score = cindex_weight * cindex + brier_weight * (1 - weighted_brier)
        
        components = {
            'hybrid_score': hybrid_score,
            'cindex': cindex,
            'weighted_brier_score': weighted_brier,
            'individual_brier_scores': dict(zip(time_points, individual_brier_scores)),
            'brier_score_component': 1 - weighted_brier
        }
        
        return hybrid_score, components
    
    def calibrate_probabilities(self, X, y_time, y_event, time_points=[24, 48, 72]):
        """
        Calibrate survival probabilities using isotonic regression
        
        Args:
            X: Feature matrix
            y_time: Survival times
            y_event: Event indicators
            time_points: Time points to calibrate for
        """
        print("Calibrating probabilities...")
        
        # Get predicted probabilities
        predicted_probs = self.predict_survival_probabilities(X, time_points)
        
        # Create calibrators for each time point
        self.calibrators = {}
        calibrated_probs = np.zeros_like(predicted_probs)
        
        for i, time_point in enumerate(time_points):
            # Determine actual outcomes
            actual = ((y_time <= time_point) & y_event).astype(float)
            
            # Fit isotonic regression
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(predicted_probs[:, i], actual)
            
            # Apply calibration
            calibrated_probs[:, i] = calibrator.transform(predicted_probs[:, i])
            
            # Store calibrator
            self.calibrators[time_point] = calibrator
        
        print("Probability calibration complete")
        return calibrated_probs
    
    def cross_validate_model(self, df, n_folds=5, **model_params):
        """
        Perform cross-validation for model evaluation
        
        Args:
            df: DataFrame with features and targets
            n_folds: Number of cross-validation folds
            model_params: Parameters for model training
            
        Returns:
            cv_results: Dictionary with cross-validation results
        """
        print(f"Performing {n_folds}-fold cross-validation...")
        
        # Prepare data
        X, y_time, y_event, feature_cols = self.prepare_survival_data(df)
        
        # Create stratified folds based on event status
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = []
        cv_components = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_event)):
            print(f"Fold {fold + 1}/{n_folds}")
            
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_time_train, y_time_val = y_time[train_idx], y_time[val_idx]
            y_event_train, y_event_val = y_event[train_idx], y_event[val_idx]
            
            # Train model
            if self.model_type == 'xgb_cox':
                self.train_xgb_cox(X_train, y_time_train, y_event_train, **model_params)
            elif self.model_type == 'xgb_aft':
                self.train_xgb_aft(X_train, y_time_train, y_event_train, **model_params)
            elif self.model_type == 'rsf':
                self.train_rsf(X_train, y_time_train, y_event_train, **model_params)
            
            # Evaluate on validation set
            hybrid_score, components = self.calculate_hybrid_score(
                X_val, y_time_val, y_event_val
            )
            
            cv_scores.append(hybrid_score)
            cv_components.append(components)
            
            print(f"  Fold {fold + 1} Hybrid Score: {hybrid_score:.4f}")
        
        # Calculate mean and std of scores
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        # Aggregate components
        aggregated_components = {}
        for key in cv_components[0].keys():
            if key == 'individual_brier_scores':
                # Handle nested dictionary
                time_points = cv_components[0][key].keys()
                aggregated_components[key] = {}
                for time_point in time_points:
                    values = [comp[key][time_point] for comp in cv_components]
                    aggregated_components[key][time_point] = {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
            else:
                values = [comp[key] for comp in cv_components]
                aggregated_components[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        cv_results = {
            'mean_hybrid_score': mean_score,
            'std_hybrid_score': std_score,
            'fold_scores': cv_scores,
            'components': aggregated_components,
            'n_folds': n_folds
        }
        
        print(f"Cross-validation complete: {mean_score:.4f} ± {std_score:.4f}")
        
        return cv_results
    
    def train_final_model(self, df, **model_params):
        """Train final model on all data"""
        print("Training final model on all data...")
        
        # Prepare data
        X, y_time, y_event, feature_cols = self.prepare_survival_data(df)
        
        # Train model
        if self.model_type == 'xgb_cox':
            self.train_xgb_cox(X, y_time, y_event, **model_params)
        elif self.model_type == 'xgb_aft':
            self.train_xgb_aft(X, y_time, y_event, **model_params)
        elif self.model_type == 'rsf':
            self.train_rsf(X, y_time, y_event, **model_params)
        
        # Calibrate probabilities
        self.calibrate_probabilities(X, y_time, y_event)
        
        # Evaluate final model
        hybrid_score, components = self.calculate_hybrid_score(X, y_time, y_event)
        
        print(f"Final model trained. Hybrid Score: {hybrid_score:.4f}")
        
        return hybrid_score, components
    
    def predict_for_submission(self, df, time_points=[12, 24, 48, 72]):
        """
        Generate predictions for submission
        
        Args:
            df: Test DataFrame
            time_points: Time points for probability prediction
            
        Returns:
            submission_df: DataFrame with submission format
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Prepare data
        X, _, _, feature_cols = self.prepare_survival_data(df)
        
        # Predict probabilities
        survival_probs = self.predict_survival_probabilities(X, time_points)
        
        # Apply calibration if available
        if self.calibrators:
            for i, time_point in enumerate(time_points):
                if time_point in self.calibrators:
                    survival_probs[:, i] = self.calibrators[time_point].transform(survival_probs[:, i])
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'event_id': df['event_id']
        })
        
        # Add probability columns
        for i, time_point in enumerate(time_points):
            submission_df[f'prob_{time_point}h'] = survival_probs[:, i]
        
        return submission_df

# Usage example
if __name__ == "__main__":
    # Test the survival models
    print("Testing survival models...")
    
    # Load and preprocess data
    from preprocessing import WildfireSurvivalPreprocessor
    
    df = pd.read_csv('data/raw/train.csv')
    preprocessor = WildfireSurvivalPreprocessor()
    processed_df = preprocessor.fit_transform(df)
    
    # Test XGBoost Cox model
    print("\nTesting XGBoost Cox model...")
    xgb_cox_model = WildfireSurvivalModels(model_type='xgb_cox')
    
    # Cross-validation
    cv_results = xgb_cox_model.cross_validate_model(processed_df, n_folds=3)
    print(f"XGBoost Cox CV Score: {cv_results['mean_hybrid_score']:.4f} ± {cv_results['std_hybrid_score']:.4f}")
    
    # Train final model
    final_score, components = xgb_cox_model.train_final_model(processed_df)
    print(f"XGBoost Cox Final Score: {final_score:.4f}")
    print(f"C-index: {components['cindex']:.4f}")
    print(f"Weighted Brier: {components['weighted_brier_score']:.4f}")
    
    print("\nModel testing complete!")
