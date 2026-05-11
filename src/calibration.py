"""
Wildfire Survival Analysis Probability Calibration
WiDS Global Datathon 2026

This module implements advanced probability calibration techniques
for survival model outputs:
1. Isotonic Regression calibration
2. Platt Scaling calibration
3. Time-specific calibration
4. Ensemble calibration methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings('ignore')

class WildfireSurvivalCalibrator:
    def __init__(self, calibration_method='isotonic', time_points=[12, 24, 48, 72]):
        """
        Initialize probability calibrator
        
        Args:
            calibration_method: Method for calibration ('isotonic', 'platt', 'ensemble')
            time_points: Time points for calibration
        """
        self.calibration_method = calibration_method
        self.time_points = time_points
        self.calibrators = {}
        self.calibration_scores = {}
        
    def prepare_calibration_data(self, y_time, y_event, predicted_probs, time_point):
        """
        Prepare data for calibration at a specific time point
        
        Args:
            y_time: True survival times
            y_event: Event indicators
            predicted_probs: Predicted survival probabilities
            time_point: Time point for calibration
            
        Returns:
            X_calib: Features for calibration (predicted probabilities)
            y_calib: True outcomes for calibration
            comparable_mask: Mask of comparable cases
        """
        # Create binary outcomes for this time point
        y_calib = ((y_time <= time_point) & y_event).astype(float)
        
        # Identify comparable cases
        # Cases with events before time_point: outcome = 1
        # Cases still at risk at time_point: outcome = 0
        # Cases censored before time_point: exclude
        comparable_mask = (y_time >= time_point) | ((y_time <= time_point) & y_event)
        
        X_calib = predicted_probs
        y_calib = y_calib
        
        return X_calib, y_calib, comparable_mask
    
    def fit_isotonic_calibration(self, X_calib, y_calib, sample_weight=None):
        """
        Fit isotonic regression calibration
        
        Args:
            X_calib: Predicted probabilities
            y_calib: True outcomes
            sample_weight: Sample weights
            
        Returns:
            calibrator: Fitted isotonic regression model
        """
        # Isotonic regression for monotonic calibration
        calibrator = IsotonicRegression(
            out_of_bounds='clip',
            increasing=True  # Survival probabilities should be monotonic
        )
        
        # Fit calibrator
        if sample_weight is not None and len(sample_weight) == len(X_calib):
            # Use sample weights if provided
            calibrator.fit(X_calib, y_calib, sample_weight=sample_weight)
        else:
            calibrator.fit(X_calib, y_calib)
        
        return calibrator
    
    def fit_platt_scaling(self, X_calib, y_calib, sample_weight=None):
        """
        Fit Platt scaling (logistic regression) calibration
        
        Args:
            X_calib: Predicted probabilities
            y_calib: True outcomes
            sample_weight: Sample weights
            
        Returns:
            calibrator: Fitted logistic regression model
        """
        # Platt scaling using logistic regression
        calibrator = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        
        # Reshape for sklearn
        X_calib_reshaped = X_calib.reshape(-1, 1)
        
        # Fit calibrator
        if sample_weight is not None and len(sample_weight) == len(X_calib):
            calibrator.fit(X_calib_reshaped, y_calib, sample_weight=sample_weight)
        else:
            calibrator.fit(X_calib_reshaped, y_calib)
        
        return calibrator
    
    def fit_ensemble_calibration(self, X_calib, y_calib, sample_weight=None):
        """
        Fit ensemble calibration (combination of isotonic and Platt)
        
        Args:
            X_calib: Predicted probabilities
            y_calib: True outcomes
            sample_weight: Sample weights
            
        Returns:
            calibrator: Dictionary with fitted calibrators
        """
        # Fit both calibrators
        isotonic_cal = self.fit_isotonic_calibration(X_calib, y_calib, sample_weight)
        platt_cal = self.fit_platt_scaling(X_calib, y_calib, sample_weight)
        
        calibrator = {
            'isotonic': isotonic_cal,
            'platt': platt_cal,
            'method': 'ensemble'
        }
        
        return calibrator
    
    def fit(self, y_time, y_event, predicted_probs):
        """
        Fit calibration models for all time points
        
        Args:
            y_time: True survival times
            y_event: Event indicators
            predicted_probs: Predicted survival probabilities (n_samples x n_time_points)
        """
        print(f"Fitting {self.calibration_method} calibration...")
        
        for i, time_point in enumerate(self.time_points):
            print(f"  Calibrating {time_point}h...")
            
            # Prepare calibration data
            X_calib, y_calib, comparable_mask = self.prepare_calibration_data(
                y_time, y_event, predicted_probs[:, i], time_point
            )
            
            # Ensure we have comparable cases
            if comparable_mask.sum() == 0:
                print(f"    Warning: No comparable cases for {time_point}h")
                continue
            
            X_calib_filtered = X_calib[comparable_mask]
            y_calib_filtered = y_calib[comparable_mask]
            
            if len(X_calib_filtered) < 10:
                print(f"    Warning: Few comparable cases ({len(X_calib_filtered)}) for {time_point}h")
                continue
            
            # Fit calibration based on method
            if self.calibration_method == 'isotonic':
                calibrator = self.fit_isotonic_calibration(X_calib_filtered, y_calib_filtered)
            elif self.calibration_method == 'platt':
                calibrator = self.fit_platt_scaling(X_calib_filtered, y_calib_filtered)
            elif self.calibration_method == 'ensemble':
                calibrator = self.fit_ensemble_calibration(X_calib_filtered, y_calib_filtered)
            else:
                raise ValueError(f"Unknown calibration method: {self.calibration_method}")
            
            # Store calibrator
            self.calibrators[time_point] = calibrator
            
            # Calculate calibration score
            calibrated_probs = self.transform_single_time_point(X_calib, time_point)
            calibration_score = self.calculate_calibration_score(y_calib_filtered, calibrated_probs[comparable_mask])
            self.calibration_scores[time_point] = calibration_score
            
            print(f"    Calibration score: {calibration_score:.4f}")
        
        print("Calibration fitting complete!")
    
    def transform_single_time_point(self, predicted_probs, time_point):
        """
        Apply calibration to predictions at a single time point
        
        Args:
            predicted_probs: Predicted probabilities for this time point
            time_point: Time point
            
        Returns:
            calibrated_probs: Calibrated probabilities
        """
        if time_point not in self.calibrators:
            return predicted_probs  # No calibration available
        
        calibrator = self.calibrators[time_point]
        
        if self.calibration_method == 'ensemble':
            # Use weighted average of both methods
            isotonic_probs = calibrator['isotonic'].transform(predicted_probs)
            platt_probs = calibrator['platt'].predict_proba(predicted_probs.reshape(-1, 1))[:, 1]
            calibrated_probs = 0.6 * isotonic_probs + 0.4 * platt_probs
        elif self.calibration_method == 'isotonic':
            calibrated_probs = calibrator.transform(predicted_probs)
        elif self.calibration_method == 'platt':
            calibrated_probs = calibrator.predict_proba(predicted_probs.reshape(-1, 1))[:, 1]
        else:
            calibrated_probs = predicted_probs
        
        # Ensure probabilities are in valid range
        calibrated_probs = np.clip(calibrated_probs, 0.001, 0.999)
        
        return calibrated_probs
    
    def transform(self, predicted_probs):
        """
        Apply calibration to all time points
        
        Args:
            predicted_probs: Predicted survival probabilities (n_samples x n_time_points)
            
        Returns:
            calibrated_probs: Calibrated probabilities
        """
        calibrated_probs = np.zeros_like(predicted_probs)
        
        for i, time_point in enumerate(self.time_points):
            calibrated_probs[:, i] = self.transform_single_time_point(predicted_probs[:, i], time_point)
        
        return calibrated_probs
    
    def calculate_calibration_score(self, y_true, y_pred, n_bins=10):
        """
        Calculate calibration score (Brier score for calibration)
        
        Args:
            y_true: True outcomes
            y_pred: Predicted probabilities
            n_bins: Number of bins for calibration curve
            
        Returns:
            calibration_score: Mean absolute calibration error
        """
        if len(y_true) < n_bins:
            return np.mean((y_true - y_pred) ** 2)  # Simple MSE if not enough data
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
        
        # Calculate mean absolute error
        calibration_error = np.mean(np.abs(prob_true - prob_pred))
        
        return calibration_error
    
    def plot_calibration_curves(self, y_time, y_event, predicted_probs, calibrated_probs):
        """
        Plot calibration curves before and after calibration
        
        Args:
            y_time: True survival times
            y_event: Event indicators
            predicted_probs: Original predicted probabilities
            calibrated_probs: Calibrated probabilities
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, time_point in enumerate(self.time_points):
            ax = axes[i]
            
            # Prepare data
            y_true = ((y_time <= time_point) & y_event).astype(int)
            comparable_mask = (y_time >= time_point) | ((y_time <= time_point) & y_event)
            
            if comparable_mask.sum() == 0:
                ax.text(0.5, 0.5, 'No comparable cases', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Calibration at {time_point}h (No data)')
                continue
            
            # Original calibration
            prob_true_orig, prob_pred_orig = calibration_curve(
                y_true[comparable_mask], 
                predicted_probs[comparable_mask, i], 
                n_bins=10
            )
            
            # Calibrated calibration
            prob_true_cal, prob_pred_cal = calibration_curve(
                y_true[comparable_mask], 
                calibrated_probs[comparable_mask, i], 
                n_bins=10
            )
            
            # Plot both curves
            ax.plot(prob_pred_orig, prob_true_orig, 'ro-', label='Original', alpha=0.7)
            ax.plot(prob_pred_cal, prob_true_cal, 'bs-', label='Calibrated', alpha=0.7)
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
            
            # Calculate errors
            error_orig = np.mean(np.abs(prob_true_orig - prob_pred_orig))
            error_cal = np.mean(np.abs(prob_true_cal - prob_pred_cal))
            
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Actual Event Rate')
            ax.set_title(f'Calibration at {time_point}h\n'
                        f'Original Error: {error_orig:.3f}, Calibrated Error: {error_cal:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Probability Calibration - {self.calibration_method.upper()} Method', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def evaluate_calibration_improvement(self, y_time, y_event, predicted_probs, calibrated_probs):
        """
        Evaluate improvement in calibration
        
        Args:
            y_time: True survival times
            y_event: Event indicators
            predicted_probs: Original predicted probabilities
            calibrated_probs: Calibrated probabilities
            
        Returns:
            improvement_metrics: Dictionary with improvement metrics
        """
        improvement_metrics = {}
        
        for i, time_point in enumerate(self.time_points):
            # Prepare data
            y_true = ((y_time <= time_point) & y_event).astype(int)
            comparable_mask = (y_time >= time_point) | ((y_time <= time_point) & y_event)
            
            if comparable_mask.sum() == 0:
                continue
            
            # Original metrics
            orig_brier = np.mean((predicted_probs[comparable_mask, i] - y_true[comparable_mask]) ** 2)
            orig_cal_error = self.calculate_calibration_score(
                y_true[comparable_mask], 
                predicted_probs[comparable_mask, i]
            )
            
            # Calibrated metrics
            cal_brier = np.mean((calibrated_probs[comparable_mask, i] - y_true[comparable_mask]) ** 2)
            cal_cal_error = self.calculate_calibration_score(
                y_true[comparable_mask], 
                calibrated_probs[comparable_mask, i]
            )
            
            # Calculate improvements
            brier_improvement = orig_brier - cal_brier
            cal_improvement = orig_cal_error - cal_cal_error
            
            improvement_metrics[time_point] = {
                'original_brier': orig_brier,
                'calibrated_brier': cal_brier,
                'brier_improvement': brier_improvement,
                'original_calibration_error': orig_cal_error,
                'calibrated_calibration_error': cal_cal_error,
                'calibration_improvement': cal_improvement
            }
        
        return improvement_metrics
    
    def generate_calibration_report(self, improvement_metrics):
        """
        Generate calibration improvement report
        
        Args:
            improvement_metrics: Dictionary with improvement metrics
        """
        print("\n" + "=" * 60)
        print("CALIBRATION IMPROVEMENT REPORT")
        print("=" * 60)
        
        for time_point, metrics in improvement_metrics.items():
            print(f"\nTime Point: {time_point}h")
            print(f"  Brier Score: {metrics['original_brier']:.4f} -> {metrics['calibrated_brier']:.4f}")
            print(f"  Brier Improvement: {metrics['brier_improvement']:.4f}")
            print(f"  Calibration Error: {metrics['original_calibration_error']:.4f} -> {metrics['calibrated_calibration_error']:.4f}")
            print(f"  Calibration Improvement: {metrics['calibration_improvement']:.4f}")
        
        # Overall improvement
        if improvement_metrics:
            avg_brier_improvement = np.mean([m['brier_improvement'] for m in improvement_metrics.values()])
            avg_cal_improvement = np.mean([m['calibration_improvement'] for m in improvement_metrics.values()])
            
            print(f"\nOverall Improvements:")
            print(f"  Average Brier Improvement: {avg_brier_improvement:.4f}")
            print(f"  Average Calibration Improvement: {avg_cal_improvement:.4f}")
        
        print("=" * 60)

# Usage example
if __name__ == "__main__":
    # Test the calibration module
    print("Testing calibration module...")
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic survival data
    y_time = np.random.exponential(scale=48, size=n_samples)
    y_time = np.clip(y_time, 0, 72)
    y_event = (np.random.random(n_samples) < np.exp(-y_time/48)).astype(int)
    
    # Generate predictions (with some mis-calibration)
    predicted_probs = np.random.beta(2, 2, (n_samples, 4))
    
    # Initialize and fit calibrator
    calibrator = WildfireSurvivalCalibrator(calibration_method='isotonic')
    calibrator.fit(y_time, y_event, predicted_probs)
    
    # Apply calibration
    calibrated_probs = calibrator.transform(predicted_probs)
    
    # Evaluate improvement
    improvement_metrics = calibrator.evaluate_calibration_improvement(
        y_time, y_event, predicted_probs, calibrated_probs
    )
    
    # Generate report
    calibrator.generate_calibration_report(improvement_metrics)
    
    print("\nCalibration module testing complete!")
