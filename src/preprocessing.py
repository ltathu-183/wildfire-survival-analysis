"""
Wildfire Survival Analysis Preprocessing Pipeline
WiDS Global Datathon 2026

This module handles data preprocessing, feature engineering, and preparation
for survival modeling, with special attention to:
1. Multicollinearity reduction
2. Rate-of-change feature engineering
3. Missing value handling and outlier detection
4. Feature scaling and encoding
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

class WildfireSurvivalPreprocessor:
    def __init__(self, vif_threshold=10, correlation_threshold=0.9):
        """
        Initialize preprocessor with thresholds for feature selection
        
        Args:
            vif_threshold: Threshold for VIF-based feature removal
            correlation_threshold: Threshold for correlation-based feature removal
        """
        self.vif_threshold = vif_threshold
        self.correlation_threshold = correlation_threshold
        self.scaler = RobustScaler()
        self.selected_features = None
        self.feature_groups = {}
        
    def engineer_rate_of_change_features(self, df):
        """
        Engineer rate-of-change features from the 5-hour window data.
        These features capture acceleration and momentum patterns.
        """
        print("Engineering rate-of-change features...")
        
        df = df.copy()
        
        # 1. Growth acceleration features
        if 'area_growth_rate_ha_per_h' in df.columns:
            # Growth momentum (rate * initial size)
            df['growth_momentum'] = df['area_growth_rate_ha_per_h'] * df['area_first_ha']
            
            # Growth acceleration indicator (positive vs negative change)
            df['is_accelerating'] = (df['area_growth_rate_ha_per_h'] > 
                                    df['area_growth_rate_ha_per_h'].median()).astype(int)
        
        # 2. Distance dynamics features
        if 'closing_speed_m_per_h' in df.columns and 'dist_accel_m_per_h2' in df.columns:
            # Closing momentum (speed * acceleration)
            df['closing_momentum'] = df['closing_speed_m_per_h'] * df['dist_accel_m_per_h2']
            
            # Threat urgency score (combines speed and acceleration)
            df['threat_urgency'] = np.where(
                df['closing_speed_m_per_h'] > 0,
                df['closing_speed_m_per_h'] * (1 + df['dist_accel_m_per_h2']),
                0
            )
            
            # Time pressure indicator (high closing speed + high acceleration)
            df['high_time_pressure'] = ((df['closing_speed_m_per_h'] > df['closing_speed_m_per_h'].quantile(0.75)) &
                                       (df['dist_accel_m_per_h2'] > 0)).astype(int)
        
        # 3. Temporal resolution quality features
        if 'low_temporal_resolution_0_5h' in df.columns and 'num_perimeters_0_5h' in df.columns:
            # Data quality score (higher = better temporal coverage)
            df['temporal_quality_score'] = np.where(
                df['low_temporal_resolution_0_5h'] == 0,
                df['num_perimeters_0_5h'] / 5.0,  # Normalize by max possible perimeters
                0.1  # Low quality penalty
            )
            
            # Temporal coverage rate
            if 'dt_first_last_0_5h' in df.columns:
                df['temporal_coverage_rate'] = df['num_perimeters_0_5h'] / (df['dt_first_last_0_5h'] + 1)
        
        # 4. Combined threat indicators
        if all(col in df.columns for col in ['dist_min_ci_0_5h', 'closing_speed_m_per_h', 'area_growth_rate_ha_per_h']):
            # Proximity-adjusted growth threat
            df['proximity_growth_threat'] = df['area_growth_rate_ha_per_h'] / (df['dist_min_ci_0_5h'] + 1000)
            
            # Combined threat score (normalized)
            df['combined_threat_score'] = (
                (df['closing_speed_m_per_h'] > 0).astype(int) * 0.4 +
                (df['area_growth_rate_ha_per_h'] > df['area_growth_rate_ha_per_h'].median()).astype(int) * 0.3 +
                (df['dist_min_ci_0_5h'] < df['dist_min_ci_0_5h'].median()).astype(int) * 0.3
            )
        
        # 5. Directional consistency features
        if all(col in df.columns for col in ['alignment_abs', 'spread_bearing_deg']):
            # Directional stability (high alignment + consistent bearing)
            df['directional_stability'] = df['alignment_abs'] * np.cos(np.radians(df['spread_bearing_deg']))
            
            # Movement efficiency (alignment * speed)
            if 'centroid_speed_m_per_h' in df.columns:
                df['movement_efficiency'] = df['alignment_abs'] * df['centroid_speed_m_per_h']
        
        # 6. Time-based interaction features
        if all(col in df.columns for col in ['event_start_hour', 'area_growth_rate_ha_per_h']):
            # Daytime growth interaction
            df['is_daytime'] = ((df['event_start_hour'] >= 6) & (df['event_start_hour'] <= 18)).astype(int)
            df['daytime_growth_boost'] = df['is_daytime'] * df['area_growth_rate_ha_per_h']
        
        print(f"Added {len([col for col in df.columns if col not in df.columns])} new engineered features")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values with appropriate strategies for different feature types"""
        print("Handling missing values...")
        
        df = df.copy()
        missing_before = df.isnull().sum().sum()
        
        # Identify feature types
        numeric_features = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['event_id', 'time_to_hit_hours', 'event']
        numeric_features = [col for col in numeric_features if col not in exclude_cols]
        
        # 1. Handle missing values in temporal features (use median)
        temporal_features = ['num_perimeters_0_5h', 'dt_first_last_0_5h', 'low_temporal_resolution_0_5h']
        for col in temporal_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # 2. Handle missing values in growth features (use 0 for rates, median for sizes)
        growth_features = [col for col in df.columns if 'area' in col or 'growth' in col or 'radial' in col]
        for col in growth_features:
            if col in df.columns:
                if 'rate' in col or 'growth' in col:
                    df[col] = df[col].fillna(0)  # Rates default to 0 (no growth)
                else:
                    df[col] = df[col].fillna(df[col].median())  # Sizes use median
        
        # 3. Handle missing values in distance features (use median)
        distance_features = [col for col in df.columns if 'dist' in col or 'closing' in col]
        for col in distance_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # 4. Handle missing values in directional features (use 0)
        directional_features = [col for col in df.columns if 'alignment' in col or 'bearing' in col]
        for col in directional_features:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # 5. Handle remaining missing values with median imputation
        for col in numeric_features:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        missing_after = df.isnull().sum().sum()
        print(f"Missing values reduced from {missing_before} to {missing_after}")
        
        return df
    
    def detect_and_handle_outliers(self, df):
        """Detect and handle outliers using robust methods"""
        print("Detecting and handling outliers...")
        
        df = df.copy()
        numeric_features = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['event_id', 'time_to_hit_hours', 'event']
        numeric_features = [col for col in numeric_features if col not in exclude_cols]
        
        outlier_counts = {}
        
        for col in numeric_features:
            if col in df.columns:
                # Use IQR method for outlier detection
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
                outlier_counts[col] = outliers.sum()
                
                # Cap outliers at the bounds (winsorization)
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        
        # Print outlier summary
        total_outliers = sum(outlier_counts.values())
        print(f"Total outliers handled: {total_outliers}")
        
        # Top features with outliers
        top_outliers = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top features with outliers:")
        for col, count in top_outliers:
            if count > 0:
                print(f"  {col}: {count} outliers")
        
        return df
    
    def remove_highly_correlated_features(self, df):
        """Remove features with high correlation to reduce multicollinearity"""
        print("Removing highly correlated features...")
        
        df = df.copy()
        numeric_features = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['event_id', 'time_to_hit_hours', 'event']
        numeric_features = [col for col in numeric_features if col not in exclude_cols]
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_features].corr().abs()
        
        # Find highly correlated feature pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Identify features to remove
        to_remove = []
        for col in upper_triangle.columns:
            if any(upper_triangle[col] > self.correlation_threshold):
                to_remove.append(col)
        
        # Remove highly correlated features
        if to_remove:
            df = df.drop(columns=to_remove)
            print(f"Removed {len(to_remove)} highly correlated features: {to_remove[:5]}...")
        else:
            print("No highly correlated features found")
        
        return df, to_remove
    
    def calculate_vif_and_select_features(self, df):
        """Calculate VIF and select features to reduce multicollinearity"""
        print("Calculating VIF and selecting features...")
        
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant
        
        df = df.copy()
        numeric_features = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['event_id', 'time_to_hit_hours', 'event']
        numeric_features = [col for col in numeric_features if col not in exclude_cols]
        
        # Remove any constant features
        constant_features = []
        for col in numeric_features:
            if df[col].std() == 0:
                constant_features.append(col)
        
        if constant_features:
            df = df.drop(columns=constant_features)
            numeric_features = [f for f in numeric_features if f not in constant_features]
            print(f"Removed {len(constant_features)} constant features")
        
        # Iterative VIF calculation and feature removal
        features_to_keep = numeric_features.copy()
        iteration = 0
        max_iterations = 10
        
        while iteration < max_iterations:
            iteration += 1
            
            # Prepare data for VIF calculation
            X = df[features_to_keep].copy()
            X = X.fillna(X.median())
            
            # Add constant for VIF calculation
            X_const = add_constant(X)
            
            # Calculate VIF for each feature
            vif_data = []
            for i, col in enumerate(features_to_keep):
                try:
                    vif = variance_inflation_factor(X_const.values, i+1)  # +1 because of constant
                    vif_data.append({'feature': col, 'vif': vif})
                except:
                    vif_data.append({'feature': col, 'vif': np.inf})
            
            vif_df = pd.DataFrame(vif_data)
            
            # Check if all features have acceptable VIF
            max_vif = vif_df['vif'].max()
            if max_vif <= self.vif_threshold:
                print(f"All features have VIF <= {self.vif_threshold} after {iteration} iterations")
                break
            
            # Remove feature with highest VIF
            feature_to_remove = vif_df.loc[vif_df['vif'].idxmax(), 'feature']
            features_to_keep.remove(feature_to_remove)
            
            print(f"Iteration {iteration}: Removed {feature_to_remove} (VIF: {max_vif:.2f})")
        
        # Update dataframe with selected features
        columns_to_keep = features_to_keep + ['event_id', 'time_to_hit_hours', 'event']
        df = df[columns_to_keep]
        
        print(f"Final feature set: {len(features_to_keep)} features")
        self.selected_features = features_to_keep
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features for modeling"""
        print("Encoding categorical features...")
        
        df = df.copy()
        
        # Encode temporal categorical features
        if 'event_start_hour' in df.columns:
            # Create cyclical encoding for hour
            df['hour_sin'] = np.sin(2 * np.pi * df['event_start_hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['event_start_hour'] / 24)
            
            # Create time periods
            df['time_period'] = pd.cut(df['event_start_hour'], 
                                     bins=[0, 6, 12, 18, 24], 
                                     labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                     include_lowest=True)
            df = pd.get_dummies(df, columns=['time_period'], prefix='period')
        
        if 'event_start_dayofweek' in df.columns:
            # Create cyclical encoding for day of week
            df['dow_sin'] = np.sin(2 * np.pi * df['event_start_dayofweek'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['event_start_dayofweek'] / 7)
            
            # Create weekend indicator
            df['is_weekend'] = (df['event_start_dayofweek'] >= 5).astype(int)
        
        if 'event_start_month' in df.columns:
            # Create seasonal encoding
            df['season'] = pd.cut(df['event_start_month'], 
                                bins=[0, 3, 6, 9, 12], 
                                labels=['Winter', 'Spring', 'Summer', 'Fall'],
                                include_lowest=True)
            df = pd.get_dummies(df, columns=['season'], prefix='season')
            
            # Create cyclical encoding for month
            df['month_sin'] = np.sin(2 * np.pi * df['event_start_month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['event_start_month'] / 12)
        
        print("Categorical features encoded")
        return df
    
    def scale_features(self, df):
        """Scale numerical features using robust scaling"""
        print("Scaling features...")
        
        df = df.copy()
        numeric_features = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['event_id', 'time_to_hit_hours', 'event']
        numeric_features = [col for col in numeric_features if col not in exclude_cols]
        
        # Fit scaler on training data
        self.scaler.fit(df[numeric_features])
        
        # Transform features
        df[numeric_features] = self.scaler.transform(df[numeric_features])
        
        print(f"Scaled {len(numeric_features)} features")
        return df
    
    def create_feature_groups(self, df):
        """Create organized feature groups for analysis"""
        print("Creating feature groups...")
        
        df = df.copy()
        all_features = df.columns.tolist()
        
        # Define feature groups based on naming patterns
        self.feature_groups = {
            'temporal_coverage': [col for col in all_features if any(x in col for x in ['temporal', 'perimeter', 'dt'])],
            'growth_dynamics': [col for col in all_features if any(x in col for x in ['area', 'growth', 'radial', 'momentum'])],
            'distance_threat': [col for col in all_features if any(x in col for x in ['dist', 'closing', 'threat', 'proximity'])],
            'directional': [col for col in all_features if any(x in col for x in ['alignment', 'bearing', 'directional'])],
            'temporal_patterns': [col for col in all_features if any(x in col for x in ['hour', 'dow', 'month', 'season', 'period'])],
            'engineered': [col for col in all_features if any(x in col for x in ['combined', 'urgency', 'pressure', 'stability', 'efficiency'])]
        }
        
        # Print group sizes
        for group, features in self.feature_groups.items():
            print(f"  {group}: {len(features)} features")
        
        return df
    
    def fit_transform(self, df):
        """Complete preprocessing pipeline for training data"""
        print("Starting preprocessing pipeline...")
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 2: Engineer rate-of-change features
        df = self.engineer_rate_of_change_features(df)
        
        # Step 3: Handle outliers
        df = self.detect_and_handle_outliers(df)
        
        # Step 4: Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Step 5: Remove highly correlated features
        df, removed_corr = self.remove_highly_correlated_features(df)
        
        # Step 6: VIF-based feature selection
        df = self.calculate_vif_and_select_features(df)
        
        # Step 7: Scale features
        df = self.scale_features(df)
        
        # Step 8: Create feature groups
        df = self.create_feature_groups(df)
        
        print(f"Preprocessing complete. Final shape: {df.shape}")
        print(f"Selected features: {len(self.selected_features) if self.selected_features else 'N/A'}")
        
        return df
    
    def transform(self, df):
        """Preprocessing pipeline for test/new data"""
        print("Transforming new data...")
        
        # Apply same transformations as training
        df = self.handle_missing_values(df)
        df = self.engineer_rate_of_change_features(df)
        df = self.detect_and_handle_outliers(df)
        df = self.encode_categorical_features(df)
        
        # Ensure only selected features are kept
        if self.selected_features:
            keep_columns = self.selected_features + ['event_id', 'time_to_hit_hours', 'event']
            # Handle case where some features might be missing
            available_columns = [col for col in keep_columns if col in df.columns]
            df = df[available_columns]
        
        # Scale features
        if self.selected_features:
            numeric_features = [col for col in self.selected_features if col in df.columns and col not in ['event_id', 'time_to_hit_hours', 'event']]
            if numeric_features:
                df[numeric_features] = self.scaler.transform(df[numeric_features])
        
        print(f"Transformation complete. Final shape: {df.shape}")
        return df
    
    def get_preprocessing_summary(self):
        """Get summary of preprocessing steps"""
        summary = {
            'vif_threshold': self.vif_threshold,
            'correlation_threshold': self.correlation_threshold,
            'selected_features_count': len(self.selected_features) if self.selected_features else 0,
            'feature_groups': {k: len(v) for k, v in self.feature_groups.items()}
        }
        return summary

# Usage example and testing
if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing preprocessing pipeline...")
    
    # Load sample data
    df = pd.read_csv('data/raw/train.csv')
    
    # Initialize preprocessor
    preprocessor = WildfireSurvivalPreprocessor(vif_threshold=10, correlation_threshold=0.9)
    
    # Process data
    processed_df = preprocessor.fit_transform(df)
    
    # Print summary
    print("\nPreprocessing Summary:")
    summary = preprocessor.get_preprocessing_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print(f"\nProcessed data shape: {processed_df.shape}")
    print(f"Features: {processed_df.columns.tolist()[:10]}...")
