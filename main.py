"""
Wildfire Survival Analysis Pipeline
WiDS Global Datathon 2026

Main pipeline for wildfire threat prediction using survival analysis.
This script orchestrates the complete workflow:
1. Data preprocessing and feature engineering
2. Survival model training (XGBoost Cox)
3. Model evaluation with custom Hybrid Score
4. Probability calibration
5. Submission generation
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.preprocessing import WildfireSurvivalPreprocessor
from src.survival_models import WildfireSurvivalModels
from src.evaluation import WildfireSurvivalEvaluator

def main():
    print("=" * 80)
    print("WILDFIRE SURVIVAL ANALYSIS - WiDS Global Datathon 2026")
    print("=" * 80)
    
    # Configuration
    config = {
        'data_path': 'data/raw/train.csv',
        'test_path': 'data/raw/test.csv',
        'submission_path': 'outputs/submission.csv',
        'model_type': 'xgb_cox',  # Best performing model
        'time_points': [12, 24, 48, 72],
        'brier_weights': [0.15, 0.3, 0.4, 0.15],  # Emphasize 48h as requested
        'random_state': 42,
        'cv_folds': 5
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Step 1: Load and preprocess data
    print("Step 1: Data Preprocessing")
    print("-" * 40)
    
    # Load training data
    train_df = pd.read_csv(config['data_path'])
    print(f"Training data loaded: {train_df.shape}")
    
    # Initialize preprocessor
    preprocessor = WildfireSurvivalPreprocessor(
        vif_threshold=10, 
        correlation_threshold=0.9
    )
    
    # Preprocess training data
    processed_train_df = preprocessor.fit_transform(train_df)
    print(f"Processed training data: {processed_train_df.shape}")
    
    # Step 2: Model training and evaluation
    print("\nStep 2: Model Training and Evaluation")
    print("-" * 40)
    
    # Initialize survival model
    model = WildfireSurvivalModels(
        model_type=config['model_type'],
        random_state=config['random_state']
    )
    
    # Perform cross-validation
    print(f"Performing {config['cv_folds']}-fold cross-validation...")
    cv_results = model.cross_validate_model(
        processed_train_df, 
        n_folds=config['cv_folds']
    )
    
    print(f"Cross-validation results:")
    print(f"  Mean Hybrid Score: {cv_results['mean_hybrid_score']:.4f} ± {cv_results['std_hybrid_score']:.4f}")
    print(f"  Component breakdown:")
    for component, values in cv_results['components'].items():
        if isinstance(values, dict) and 'mean' in values:
            print(f"    {component}: {values['mean']:.4f} ± {values['std']:.4f}")
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    final_score, final_components = model.train_final_model(processed_train_df)
    
    print(f"Final model performance:")
    print(f"  Hybrid Score: {final_score:.4f}")
    print(f"  C-index: {final_components['cindex']:.4f}")
    print(f"  Weighted Brier Score: {final_components['weighted_brier_score']:.4f}")
    
    # Step 3: Detailed evaluation
    print("\nStep 3: Detailed Model Evaluation")
    print("-" * 40)
    
    # Initialize evaluator
    evaluator = WildfireSurvivalEvaluator(
        time_points=config['time_points'],
        brier_weights=config['brier_weights']
    )
    
    # Get predictions for detailed evaluation
    X, y_time, y_event, feature_cols = model.prepare_survival_data(processed_train_df)
    risk_scores = model.predict_risk_scores(X)
    survival_probs = model.predict_survival_probabilities(X, config['time_points'])
    
    # Comprehensive evaluation
    evaluation_results = evaluator.evaluate_model_predictions(
        y_time, y_event, risk_scores, survival_probs, 
        model_name=f"Final {config['model_type'].upper()} Model"
    )
    
    # Error analysis
    evaluator.analyze_prediction_errors(
        y_time, y_event, risk_scores, survival_probs,
        model_name=f"Final {config['model_type'].upper()} Model"
    )
    
    # Generate evaluation report
    report_path = 'outputs/evaluation_report.txt'
    evaluator.generate_evaluation_report(evaluation_results, save_path=report_path)
    
    # Step 4: Generate submission (if test data available)
    print("\nStep 4: Submission Generation")
    print("-" * 40)
    
    try:
        # Load test data
        test_df = pd.read_csv(config['test_path'])
        print(f"Test data loaded: {test_df.shape}")
        
        # Preprocess test data
        processed_test_df = preprocessor.transform(test_df)
        print(f"Processed test data: {processed_test_df.shape}")
        
        # Generate predictions
        submission_df = model.predict_for_submission(processed_test_df, config['time_points'])
        
        # Save submission
        submission_df.to_csv(config['submission_path'], index=False)
        print(f"Submission saved to: {config['submission_path']}")
        print(f"Submission shape: {submission_df.shape}")
        print(f"Submission columns: {submission_df.columns.tolist()}")
        
        # Display sample predictions
        print("\nSample predictions:")
        print(submission_df.head(10).to_string(index=False))
        
    except FileNotFoundError:
        print("Test data not found. Skipping submission generation.")
    except Exception as e:
        print(f"Error during submission generation: {e}")
    
    # Step 5: Summary and recommendations
    print("\nStep 5: Analysis Summary")
    print("-" * 40)
    
    print("Key Findings:")
    print(f"1. Model Performance: Hybrid Score of {final_score:.4f}")
    print(f"2. Discrimination (C-index): {final_components['cindex']:.4f}")
    print(f"3. Calibration (Brier): {final_components['weighted_brier_score']:.4f}")
    
    print("\nTime-specific Performance:")
    for time_point, brier in final_components['individual_brier_scores'].items():
        weight = config['brier_weights'][config['time_points'].index(time_point)]
        print(f"  {time_point}h: Brier = {brier:.4f} (weight = {weight:.2f})")
    
    print("\nFeature Engineering Insights:")
    print(f"1. Selected {len(preprocessor.selected_features)} features after VIF reduction")
    print(f"2. Feature groups:")
    for group, features in preprocessor.feature_groups.items():
        print(f"   {group}: {len(features)} features")
    
    print("\nRecommendations for Competition:")
    print("1. Focus on 48-hour predictions (highest weight in evaluation)")
    print("2. Consider ensemble methods for improved performance")
    print("3. Feature engineering on rate-of-change metrics is crucial")
    print("4. Proper censoring handling is essential for accurate Brier scores")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
