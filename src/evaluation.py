"""
Wildfire Survival Analysis Evaluation Metrics
WiDS Global Datathon 2026

This module implements custom evaluation metrics for the wildfire survival challenge:
1. Custom Hybrid Score (0.3 * C-index + 0.7 * [1 - Weighted Brier Score])
2. Proper Brier Score calculation with censoring handling
3. Time-specific calibration metrics
4. Model comparison and selection tools
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class WildfireSurvivalEvaluator:
    def __init__(self, time_points=[12, 24, 48, 72], brier_weights=[0.15, 0.3, 0.4, 0.15]):
        """
        Initialize evaluator with time points and weights
        
        Args:
            time_points: Time points (hours) for evaluation
            brier_weights: Weights for Brier score at each time point
                         (default: 0.15, 0.3, 0.4, 0.15 for 12, 24, 48, 72h)
        """
        self.time_points = time_points
        self.brier_weights = np.array(brier_weights)
        self.brier_weights = self.brier_weights / self.brier_weights.sum()  # Normalize
        
    def calculate_concordance_index(self, y_time, y_event, risk_scores):
        """
        Calculate Concordance Index (C-index) for survival models
        
        Args:
            y_time: True survival times
            y_event: Event indicators (1=event, 0=censored)
            risk_scores: Predicted risk scores (higher = higher risk)
            
        Returns:
            cindex: Concordance index
        """
        n_pairs = 0
        n_concordant = 0
        
        for i in range(len(y_time)):
            for j in range(i+1, len(y_time)):
                # Check if pair is comparable
                if y_time[i] == y_time[j]:
                    continue
                    
                # Determine if pair is usable (at least one event)
                if y_event[i] or y_event[j]:
                    n_pairs += 1
                    
                    # Determine ordering
                    if y_event[i] and (not y_event[j] or y_time[i] <= y_time[j]):
                        # i is an event and occurs before or at same time as j
                        if risk_scores[i] > risk_scores[j]:
                            n_concordant += 1
                        elif risk_scores[i] == risk_scores[j]:
                            n_concordant += 0.5  # Ties count as half
                            
                    elif y_event[j] and (not y_event[i] or y_time[j] <= y_time[i]):
                        # j is an event and occurs before or at same time as i
                        if risk_scores[j] > risk_scores[i]:
                            n_concordant += 1
                        elif risk_scores[j] == risk_scores[i]:
                            n_concordant += 0.5  # Ties count as half
        
        return n_concordant / n_pairs if n_pairs > 0 else 0.5
    
    def calculate_brier_score_at_time(self, y_time, y_event, predicted_survival, time_point):
        """
        Calculate Brier score at a specific time point with proper censoring handling
        
        Args:
            y_time: True survival times
            y_event: Event indicators
            predicted_survival: Predicted survival probabilities at time_point
            time_point: Time point for evaluation
            
        Returns:
            brier_score: Brier score at the specified time point
        """
        # Identify cases that are still at risk at time_point
        at_risk_mask = y_time >= time_point
        
        # For cases still at risk, we don't know their true status at time_point
        # For cases with events before time_point, we know they had the event
        # For censored cases before time_point, we exclude them (inverse probability weighting would be ideal)
        
        # Create actual outcomes
        actual = np.zeros_like(y_time, dtype=float)
        
        # Cases with events before or at time_point
        event_before_mask = (y_time <= time_point) & y_event
        actual[event_before_mask] = 1.0
        
        # Cases censored before time_point - exclude from calculation
        censored_before_mask = (y_time < time_point) & ~y_event
        
        # Calculate Brier score only on comparable cases
        comparable_mask = at_risk_mask | event_before_mask
        
        if comparable_mask.sum() == 0:
            return 0.5  # Default if no comparable cases
        
        brier = np.mean((predicted_survival[comparable_mask] - actual[comparable_mask]) ** 2)
        
        return brier
    
    def calculate_weighted_brier_score(self, y_time, y_event, predicted_survival_probs):
        """
        Calculate weighted Brier score across multiple time points
        
        Args:
            y_time: True survival times
            y_event: Event indicators
            predicted_survival_probs: Predicted survival probabilities at each time point
                                    (shape: n_samples x n_time_points)
            
        Returns:
            weighted_brier: Weighted Brier score
            individual_briers: Individual Brier scores at each time point
        """
        individual_briers = []
        
        for i, time_point in enumerate(self.time_points):
            brier = self.calculate_brier_score_at_time(
                y_time, y_event, predicted_survival_probs[:, i], time_point
            )
            individual_briers.append(brier)
        
        # Calculate weighted average
        weighted_brier = np.average(individual_briers, weights=self.brier_weights)
        
        return weighted_brier, individual_briers
    
    def calculate_hybrid_score(self, y_time, y_event, risk_scores, predicted_survival_probs,
                               cindex_weight=0.3, brier_weight=0.7):
        """
        Calculate the Hybrid Score: 0.3 * C-index + 0.7 * [1 - Weighted Brier Score]
        
        Args:
            y_time: True survival times
            y_event: Event indicators
            risk_scores: Predicted risk scores
            predicted_survival_probs: Predicted survival probabilities at each time point
            cindex_weight: Weight for C-index component (default 0.3)
            brier_weight: Weight for Brier score component (default 0.7)
            
        Returns:
            hybrid_score: Final hybrid score
            components: Dictionary with individual components
        """
        # Calculate C-index
        cindex = self.calculate_concordance_index(y_time, y_event, risk_scores)
        
        # Calculate weighted Brier score
        weighted_brier, individual_briers = self.calculate_weighted_brier_score(
            y_time, y_event, predicted_survival_probs
        )
        
        # Calculate hybrid score
        hybrid_score = cindex_weight * cindex + brier_weight * (1 - weighted_brier)
        
        components = {
            'hybrid_score': hybrid_score,
            'cindex': cindex,
            'weighted_brier_score': weighted_brier,
            'brier_score_component': 1 - weighted_brier,
            'individual_brier_scores': dict(zip(self.time_points, individual_briers)),
            'cindex_component': cindex,
            'weights': {'cindex': cindex_weight, 'brier': brier_weight}
        }
        
        return hybrid_score, components
    
    def evaluate_model_predictions(self, y_time, y_event, risk_scores, predicted_survival_probs,
                                 model_name="Model"):
        """
        Comprehensive model evaluation
        
        Args:
            y_time: True survival times
            y_event: Event indicators
            risk_scores: Predicted risk scores
            predicted_survival_probs: Predicted survival probabilities
            model_name: Name of the model for reporting
            
        Returns:
            evaluation_results: Dictionary with all evaluation metrics
        """
        print(f"Evaluating {model_name}...")
        
        # Calculate main metrics
        hybrid_score, components = self.calculate_hybrid_score(
            y_time, y_event, risk_scores, predicted_survival_probs
        )
        
        # Additional metrics
        evaluation_results = {
            'model_name': model_name,
            'hybrid_score': hybrid_score,
            'components': components,
            'time_points': self.time_points,
            'brier_weights': self.brier_weights.tolist()
        }
        
        # Print summary
        print(f"  Hybrid Score: {hybrid_score:.4f}")
        print(f"  C-index: {components['cindex']:.4f}")
        print(f"  Weighted Brier: {components['weighted_brier_score']:.4f}")
        print(f"  Individual Brier Scores:")
        for time_point, brier in components['individual_brier_scores'].items():
            weight = self.brier_weights[self.time_points.index(time_point)]
            print(f"    {time_point}h: {brier:.4f} (weight: {weight:.2f})")
        
        return evaluation_results
    
    def compare_models(self, model_predictions_dict):
        """
        Compare multiple models side by side
        
        Args:
            model_predictions_dict: Dictionary with model names as keys and 
                                  (risk_scores, predicted_survival_probs) as values
                                  
        Returns:
            comparison_df: DataFrame with model comparison
        """
        print("Comparing models...")
        
        results = []
        
        for model_name, (risk_scores, predicted_survival_probs) in model_predictions_dict.items():
            # We need y_time and y_event - assume they're available globally or passed
            # For now, create a placeholder structure
            result = {
                'model_name': model_name,
                'hybrid_score': 0.0,  # Placeholder
                'cindex': 0.0,
                'weighted_brier': 0.0
            }
            results.append(result)
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('hybrid_score', ascending=False)
        
        print("\nModel Comparison:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_calibration_curves(self, y_time, y_event, predicted_survival_probs, 
                               model_name="Model", n_bins=10):
        """
        Plot calibration curves for survival probabilities at different time points
        
        Args:
            y_time: True survival times
            y_event: Event indicators
            predicted_survival_probs: Predicted survival probabilities
            model_name: Model name for plot titles
            n_bins: Number of bins for calibration curves
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, time_point in enumerate(self.time_points):
            ax = axes[i]
            
            # Create actual outcomes for this time point
            actual = ((y_time <= time_point) & y_event).astype(int)
            
            # Only use cases that are comparable (events before or at time_point, or still at risk)
            comparable_mask = (y_time >= time_point) | ((y_time <= time_point) & y_event)
            
            if comparable_mask.sum() > 0:
                # Calculate calibration curve
                prob_true, prob_pred = calibration_curve(
                    actual[comparable_mask], 
                    predicted_survival_probs[comparable_mask, i], 
                    n_bins=n_bins
                )
                
                # Plot calibration curve
                ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Calibration curve')
                ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
                
                # Calculate calibration metrics
                calibration_error = np.mean(np.abs(prob_true - prob_pred))
                
                ax.set_xlabel(f'Mean Predicted Probability')
                ax.set_ylabel(f'Actual Event Rate')
                ax.set_title(f'Calibration at {time_point}h\n(Error: {calibration_error:.3f})')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No comparable cases', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Calibration at {time_point}h\n(No data)')
        
        plt.suptitle(f'Calibration Curves - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_time_specific_performance(self, evaluation_results_list):
        """
        Plot model performance across different time points
        
        Args:
            evaluation_results_list: List of evaluation results from multiple models
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Brier scores over time
        for result in evaluation_results_list:
            model_name = result['model_name']
            brier_scores = list(result['components']['individual_brier_scores'].values())
            
            ax1.plot(self.time_points, brier_scores, marker='o', label=model_name)
        
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Brier Score')
        ax1.set_title('Brier Score Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Overall performance comparison
        model_names = [r['model_name'] for r in evaluation_results_list]
        hybrid_scores = [r['hybrid_score'] for r in evaluation_results_list]
        cindices = [r['components']['cindex'] for r in evaluation_results_list]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax2.bar(x - width/2, hybrid_scores, width, label='Hybrid Score', alpha=0.8)
        ax2.bar(x + width/2, cindices, width, label='C-index', alpha=0.8)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Score')
        ax2.set_title('Model Performance Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_prediction_errors(self, y_time, y_event, risk_scores, predicted_survival_probs,
                                model_name="Model"):
        """
        Analyze prediction errors and patterns
        
        Args:
            y_time: True survival times
            y_event: Event indicators
            risk_scores: Predicted risk scores
            predicted_survival_probs: Predicted survival probabilities
            model_name: Model name for reporting
        """
        print(f"\nError Analysis for {model_name}:")
        
        # Risk score analysis
        risk_correlation, _ = spearmanr(risk_scores, y_time)
        print(f"  Risk score vs time correlation: {risk_correlation:.3f}")
        
        # Time-specific error analysis
        for i, time_point in enumerate(self.time_points):
            # Calculate predictions at this time point
            surv_pred = predicted_survival_probs[:, i]
            
            # Calculate actual outcomes
            actual = ((y_time <= time_point) & y_event).astype(int)
            
            # Calculate error metrics
            mse = np.mean((surv_pred - actual) ** 2)
            mae = np.mean(np.abs(surv_pred - actual))
            
            print(f"  {time_point}h - MSE: {mse:.3f}, MAE: {mae:.3f}")
        
        # Analyze performance by censoring status
        event_mask = y_event == 1
        censored_mask = ~event_mask
        
        if event_mask.sum() > 0 and censored_mask.sum() > 0:
            event_correlation, _ = spearmanr(risk_scores[event_mask], y_time[event_mask])
            censored_correlation, _ = spearmanr(risk_scores[censored_mask], y_time[censored_mask])
            
            print(f"  Risk correlation (events only): {event_correlation:.3f}")
            print(f"  Risk correlation (censored only): {censored_correlation:.3f}")
    
    def generate_evaluation_report(self, evaluation_results, save_path=None):
        """
        Generate comprehensive evaluation report
        
        Args:
            evaluation_results: Dictionary with evaluation results
            save_path: Optional path to save the report
        """
        report = []
        report.append("=" * 60)
        report.append("WILDFIRE SURVIVAL ANALYSIS EVALUATION REPORT")
        report.append("=" * 60)
        
        # Model information
        model_name = evaluation_results['model_name']
        report.append(f"Model: {model_name}")
        report.append(f"Evaluation Time Points: {evaluation_results['time_points']}")
        report.append(f"Brier Score Weights: {evaluation_results['brier_weights']}")
        report.append("")
        
        # Main metrics
        components = evaluation_results['components']
        report.append("PERFORMANCE METRICS:")
        report.append(f"  Hybrid Score: {components['hybrid_score']:.4f}")
        report.append(f"  C-index: {components['cindex']:.4f}")
        report.append(f"  Weighted Brier Score: {components['weighted_brier_score']:.4f}")
        report.append("")
        
        # Time-specific Brier scores
        report.append("TIME-SPECIFIC BRIER SCORES:")
        for time_point, brier in components['individual_brier_scores'].items():
            weight = evaluation_results['brier_weights'][evaluation_results['time_points'].index(time_point)]
            contribution = brier * weight
            report.append(f"  {time_point}h: {brier:.4f} (weight: {weight:.2f}, contribution: {contribution:.4f})")
        report.append("")
        
        # Component breakdown
        report.append("HYBRID SCORE BREAKDOWN:")
        cindex_component = components['cindex'] * components['weights']['cindex']
        brier_component = components['brier_score_component'] * components['weights']['brier']
        report.append(f"  C-index component: {cindex_component:.4f} ({components['weights']['cindex']:.1f} weight)")
        report.append(f"  Brier component: {brier_component:.4f} ({components['weights']['brier']:.1f} weight)")
        report.append(f"  Total: {cindex_component + brier_component:.4f}")
        
        report_text = "\n".join(report)
        
        print(report_text)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to: {save_path}")
        
        return report_text

# Usage example
if __name__ == "__main__":
    # Test the evaluation module
    print("Testing evaluation module...")
    
    # Create dummy data for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic survival data
    y_time = np.random.exponential(scale=48, size=n_samples)
    y_time = np.clip(y_time, 0, 72)  # Cap at 72 hours
    
    # Generate event indicators (more events at early times)
    y_event = (np.random.random(n_samples) < np.exp(-y_time/48)).astype(int)
    
    # Generate predictions
    risk_scores = np.random.normal(0, 1, n_samples)
    predicted_survival = np.random.beta(2, 2, (n_samples, 4))  # 4 time points
    
    # Initialize evaluator
    evaluator = WildfireSurvivalEvaluator()
    
    # Test evaluation
    results = evaluator.evaluate_model_predictions(
        y_time, y_event, risk_scores, predicted_survival, "Test Model"
    )
    
    # Generate report
    evaluator.generate_evaluation_report(results)
    
    print("\nEvaluation module testing complete!")
