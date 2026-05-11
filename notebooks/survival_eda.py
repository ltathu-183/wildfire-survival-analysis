"""
Wildfire Survival Analysis EDA
WiDS Global Datathon 2026

This script performs specialized Exploratory Data Analysis for survival analysis
of wildfire evacuation zone threats, focusing on:
1. Target analysis (time_to_hit_hours, censoring)
2. Kaplan-Meier survival curves
3. Feature correlations and multicollinearity
4. Temporal dynamics analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Survival analysis libraries
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WildfireSurvivalEDA:
    def __init__(self, data_path):
        """Initialize EDA with data path"""
        self.data_path = data_path
        self.df = None
        self.numeric_features = None
        self.categorical_features = None
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("Loading wildfire survival data...")
        self.df = pd.read_csv(self.data_path)
        
        # Identify feature categories based on metadata
        temporal_features = [
            'num_perimeters_0_5h', 'dt_first_last_0_5h', 'low_temporal_resolution_0_5h'
        ]
        
        growth_features = [
            'area_first_ha', 'area_growth_abs_0_5h', 'area_growth_rel_0_5h',
            'area_growth_rate_ha_per_h', 'log1p_area_first', 'log1p_growth',
            'log_area_ratio_0_5h', 'relative_growth_0_5h', 'radial_growth_m',
            'radial_growth_rate_m_per_h'
        ]
        
        centroid_features = [
            'centroid_displacement_m', 'centroid_speed_m_per_h',
            'spread_bearing_deg', 'spread_bearing_sin', 'spread_bearing_cos'
        ]
        
        distance_features = [
            'dist_min_ci_0_5h', 'dist_std_ci_0_5h', 'dist_change_ci_0_5h',
            'dist_slope_ci_0_5h', 'closing_speed_m_per_h', 'closing_speed_abs_m_per_h',
            'projected_advance_m', 'dist_accel_m_per_h2', 'dist_fit_r2_0_5h'
        ]
        
        directionality_features = [
            'alignment_cos', 'alignment_abs', 'cross_track_component',
            'along_track_speed'
        ]
        
        temporal_metadata = [
            'event_start_hour', 'event_start_dayofweek', 'event_start_month'
        ]
        
        self.feature_groups = {
            'temporal': temporal_features,
            'growth': growth_features,
            'centroid': centroid_features,
            'distance': distance_features,
            'directionality': directionality_features,
            'temporal_metadata': temporal_metadata
        }
        
        # All numeric features (excluding targets and event_id)
        exclude_cols = ['event_id', 'time_to_hit_hours', 'event']
        self.numeric_features = [col for col in self.df.columns 
                                if col not in exclude_cols and col in self.df.select_dtypes(include=[np.number]).columns]
        
        # Categorical features (derived from temporal metadata)
        self.categorical_features = ['event_start_hour', 'event_start_dayofweek', 'event_start_month']
        
        print(f"Dataset loaded: {self.df.shape}")
        print(f"Events: {self.df['event'].sum()} ({self.df['event'].mean():.1%})")
        print(f"Censored: {(1-self.df['event']).sum()} ({(1-self.df['event']).mean():.1%})")
        
    def target_analysis(self):
        """Analyze survival targets and censoring patterns"""
        print("\n" + "="*50)
        print("TARGET ANALYSIS")
        print("="*50)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Distribution of Time to Hit (Hours)',
                'Event vs Censored Proportions',
                'Time to Hit by Event Status',
                'Censoring Distribution Across Time'
            ],
            specs=[[{"type": "histogram"}, {"type": "pie"}],
                   [{"type": "violin"}, {"type": "bar"}]]
        )
        
        # 1. Distribution of time_to_hit_hours
        fig.add_trace(
            go.Histogram(x=self.df['time_to_hit_hours'], 
                        name='Time to Hit',
                        nbinsx=30,
                        marker_color='lightblue'),
            row=1, col=1
        )
        
        # 2. Event vs Censored proportions
        event_counts = self.df['event'].value_counts()
        fig.add_trace(
            go.Pie(labels=['Censored', 'Event'],
                   values=[event_counts[0], event_counts[1]],
                   marker_colors=['lightcoral', 'lightgreen']),
            row=1, col=2
        )
        
        # 3. Time to hit by event status
        for status in [0, 1]:
            status_label = 'Censored' if status == 0 else 'Event'
            subset = self.df[self.df['event'] == status]
            fig.add_trace(
                go.Violin(y=subset['time_to_hit_hours'],
                         name=status_label,
                         box_visible=True,
                         meanline_visible=True),
                row=2, col=1
            )
        
        # 4. Censoring distribution across time windows
        time_windows = [(0, 12), (12, 24), (24, 48), (48, 72)]
        window_labels = ['0-12h', '12-24h', '24-48h', '48-72h']
        censoring_rates = []
        
        for start, end in time_windows:
            window_data = self.df[
                (self.df['time_to_hit_hours'] >= start) & 
                (self.df['time_to_hit_hours'] < end)
            ]
            if len(window_data) > 0:
                censoring_rate = (1 - window_data['event']).mean()
                censoring_rates.append(censoring_rate)
            else:
                censoring_rates.append(0)
        
        fig.add_trace(
            go.Bar(x=window_labels, y=censoring_rates,
                  name='Censoring Rate',
                  marker_color='orange'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="Wildfire Survival Target Analysis")
        fig.show()
        
        # Print summary statistics
        print("\nTarget Statistics:")
        print(f"Mean time to hit: {self.df['time_to_hit_hours'].mean():.2f} hours")
        print(f"Median time to hit: {self.df['time_to_hit_hours'].median():.2f} hours")
        print(f"Min time to hit: {self.df['time_to_hit_hours'].min():.2f} hours")
        print(f"Max time to hit: {self.df['time_to_hit_hours'].max():.2f} hours")
        
        # Censoring analysis by time windows
        print("\nCensoring Analysis by Time Windows:")
        for i, (start, end) in enumerate(time_windows):
            window_data = self.df[
                (self.df['time_to_hit_hours'] >= start) & 
                (self.df['time_to_hit_hours'] < end)
            ]
            if len(window_data) > 0:
                censoring_rate = (1 - window_data['event']).mean()
                print(f"{window_labels[i]}: {len(window_data)} events, {censoring_rate:.1%} censored")
    
    def kaplan_meier_analysis(self):
        """Generate Kaplan-Meier survival curves"""
        print("\n" + "="*50)
        print("KAPLAN-MEIER SURVIVAL ANALYSIS")
        print("="*50)
        
        kmf = KaplanMeierFitter()
        
        # Overall survival curve
        plt.figure(figsize=(15, 10))
        
        # 1. Overall survival curve
        plt.subplot(2, 3, 1)
        kmf.fit(self.df['time_to_hit_hours'], self.df['event'], label='Overall')
        kmf.plot_survival_function()
        plt.title('Overall Survival Curve')
        plt.xlabel('Time (hours)')
        plt.ylabel('Survival Probability')
        plt.grid(True, alpha=0.3)
        
        # 2. Survival by event start hour (day vs night)
        plt.subplot(2, 3, 2)
        self.df['is_daytime'] = (self.df['event_start_hour'] >= 6) & (self.df['event_start_hour'] <= 18)
        
        for is_day in [True, False]:
            label = 'Daytime (6-18h)' if is_day else 'Nighttime (19-5h)'
            mask = self.df['is_daytime'] == is_day
            kmf.fit(self.df.loc[mask, 'time_to_hit_hours'], 
                   self.df.loc[mask, 'event'], 
                   label=label)
            kmf.plot_survival_function()
        
        plt.title('Survival by Time of Day')
        plt.xlabel('Time (hours)')
        plt.ylabel('Survival Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Survival by initial fire size (quartiles)
        plt.subplot(2, 3, 3)
        self.df['size_quartile'] = pd.qcut(self.df['area_first_ha'], 
                                          q=4, 
                                          labels=['Small', 'Medium', 'Large', 'Very Large'])
        
        for quartile in ['Small', 'Medium', 'Large', 'Very Large']:
            mask = self.df['size_quartile'] == quartile
            if mask.sum() > 0:
                kmf.fit(self.df.loc[mask, 'time_to_hit_hours'], 
                       self.df.loc[mask, 'event'], 
                       label=quartile)
                kmf.plot_survival_function()
        
        plt.title('Survival by Initial Fire Size')
        plt.xlabel('Time (hours)')
        plt.ylabel('Survival Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Survival by closing speed (positive vs negative/zero)
        plt.subplot(2, 3, 4)
        self.df['is_closing'] = self.df['closing_speed_m_per_h'] > 0
        
        for is_closing in [True, False]:
            label = 'Closing Speed > 0' if is_closing else 'Closing Speed <= 0'
            mask = self.df['is_closing'] == is_closing
            if mask.sum() > 0:
                kmf.fit(self.df.loc[mask, 'time_to_hit_hours'], 
                       self.df.loc[mask, 'event'], 
                       label=label)
                kmf.plot_survival_function()
        
        plt.title('Survival by Closing Speed')
        plt.xlabel('Time (hours)')
        plt.ylabel('Survival Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Survival by temporal resolution quality
        plt.subplot(2, 3, 5)
        for low_res in [0, 1]:
            label = 'Good Temporal Resolution' if low_res == 0 else 'Low Temporal Resolution'
            mask = self.df['low_temporal_resolution_0_5h'] == low_res
            if mask.sum() > 0:
                kmf.fit(self.df.loc[mask, 'time_to_hit_hours'], 
                       self.df.loc[mask, 'event'], 
                       label=label)
                kmf.plot_survival_function()
        
        plt.title('Survival by Temporal Resolution')
        plt.xlabel('Time (hours)')
        plt.ylabel('Survival Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Survival by distance quartiles
        plt.subplot(2, 3, 6)
        self.df['dist_quartile'] = pd.qcut(self.df['dist_min_ci_0_5h'], 
                                          q=4, 
                                          labels=['Very Close', 'Close', 'Far', 'Very Far'])
        
        for quartile in ['Very Close', 'Close', 'Far', 'Very Far']:
            mask = self.df['dist_quartile'] == quartile
            if mask.sum() > 0:
                kmf.fit(self.df.loc[mask, 'time_to_hit_hours'], 
                       self.df.loc[mask, 'event'], 
                       label=quartile)
                kmf.plot_survival_function()
        
        plt.title('Survival by Initial Distance')
        plt.xlabel('Time (hours)')
        plt.ylabel('Survival Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Log-rank tests for statistical significance
        print("\nLog-rank Test Results:")
        
        # Test daytime vs nighttime
        results_daytime = logrank_test(
            self.df[self.df['is_daytime']]['time_to_hit_hours'],
            self.df[~self.df['is_daytime']]['time_to_hit_hours'],
            self.df[self.df['is_daytime']]['event'],
            self.df[~self.df['is_daytime']]['event']
        )
        print(f"Daytime vs Nighttime: p-value = {results_daytime.p_value:.4f}")
        
        # Test closing speed
        results_closing = logrank_test(
            self.df[self.df['is_closing']]['time_to_hit_hours'],
            self.df[~self.df['is_closing']]['time_to_hit_hours'],
            self.df[self.df['is_closing']]['event'],
            self.df[~self.df['is_closing']]['event']
        )
        print(f"Closing vs Non-closing: p-value = {results_closing.p_value:.4f}")
    
    def feature_correlation_analysis(self):
        """Analyze feature correlations and multicollinearity"""
        print("\n" + "="*50)
        print("FEATURE CORRELATION & MULTICOLLINEARITY")
        print("="*50)
        
        # Spearman correlation matrix
        correlation_data = self.df[self.numeric_features].copy()
        
        # Handle any remaining missing values
        correlation_data = correlation_data.fillna(correlation_data.median())
        
        # Calculate Spearman correlations
        spearman_corr = correlation_data.corr(method='spearman')
        
        # Create correlation heatmap
        plt.figure(figsize=(20, 16))
        
        # Mask for upper triangle
        mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(spearman_corr, 
                   mask=mask, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8},
                   annot=False)
        
        plt.title('Spearman Correlation Matrix of Features', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Find highly correlated features
        high_corr_threshold = 0.8
        high_corr_pairs = []
        
        for i in range(len(spearman_corr.columns)):
            for j in range(i+1, len(spearman_corr.columns)):
                if abs(spearman_corr.iloc[i, j]) > high_corr_threshold:
                    high_corr_pairs.append({
                        'feature1': spearman_corr.columns[i],
                        'feature2': spearman_corr.columns[j],
                        'correlation': spearman_corr.iloc[i, j]
                    })
        
        print(f"\nHighly correlated feature pairs (|r| > {high_corr_threshold}):")
        for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)[:10]:
            print(f"{pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
        
        # VIF Analysis
        print("\nVariance Inflation Factor (VIF) Analysis:")
        
        # Prepare data for VIF (drop any remaining NaN values)
        vif_data = correlation_data.copy()
        vif_data = vif_data.dropna()
        
        if len(vif_data) > 0:
            # Add constant for VIF calculation
            vif_data_const = pd.concat([pd.Series(1, index=vif_data.index, name='const'), vif_data], axis=1)
            
            # Calculate VIF for each feature
            vif_results = []
            for i, col in enumerate(vif_data.columns):
                try:
                    vif = variance_inflation_factor(vif_data_const.values, i+1)  # +1 because of constant
                    vif_results.append({'feature': col, 'vif': vif})
                except:
                    vif_results.append({'feature': col, 'vif': np.inf})
            
            vif_df = pd.DataFrame(vif_results)
            vif_df = vif_df.sort_values('vif', ascending=False)
            
            print("\nTop 20 features by VIF:")
            print(vif_df.head(20).to_string(index=False))
            
            # Identify features with high VIF
            high_vif_threshold = 10
            high_vif_features = vif_df[vif_df['vif'] > high_vif_threshold]['feature'].tolist()
            print(f"\nFeatures with VIF > {high_vif_threshold}: {len(high_vif_features)}")
    
    def temporal_dynamics_analysis(self):
        """Analyze temporal dynamics and rate-of-change features"""
        print("\n" + "="*50)
        print("TEMPORAL DYNAMICS ANALYSIS")
        print("="*50)
        
        # Create rate-of-change features from existing data
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Growth Rate vs Time to Hit',
                'Closing Speed vs Time to Hit',
                'Radial Growth Rate Distribution',
                'Acceleration Patterns'
            ],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # 1. Growth rate vs time to hit
        fig.add_trace(
            go.Scatter(
                x=self.df['area_growth_rate_ha_per_h'],
                y=self.df['time_to_hit_hours'],
                mode='markers',
                marker=dict(
                    color=self.df['event'],
                    colorscale='RdYlBu',
                    size=4,
                    opacity=0.6
                ),
                name='Growth Rate',
                text=self.df['event'],
                hovertemplate='Growth Rate: %{x:.2f} ha/h<br>Time to Hit: %{y:.2f}h<br>Event: %{text}'
            ),
            row=1, col=1
        )
        
        # 2. Closing speed vs time to hit
        fig.add_trace(
            go.Scatter(
                x=self.df['closing_speed_m_per_h'],
                y=self.df['time_to_hit_hours'],
                mode='markers',
                marker=dict(
                    color=self.df['event'],
                    colorscale='RdYlBu',
                    size=4,
                    opacity=0.6
                ),
                name='Closing Speed',
                text=self.df['event'],
                hovertemplate='Closing Speed: %{x:.2f} m/h<br>Time to Hit: %{y:.2f}h<br>Event: %{text}'
            ),
            row=1, col=2
        )
        
        # 3. Radial growth rate distribution by event status
        for event_status in [0, 1]:
            status_label = 'Censored' if event_status == 0 else 'Event'
            subset = self.df[self.df['event'] == event_status]
            fig.add_trace(
                go.Histogram(
                    x=subset['radial_growth_rate_m_per_h'],
                    name=status_label,
                    opacity=0.7,
                    nbinsx=30
                ),
                row=2, col=1
            )
        
        # 4. Acceleration patterns (distance acceleration vs time to hit)
        fig.add_trace(
            go.Scatter(
                x=self.df['dist_accel_m_per_h2'],
                y=self.df['time_to_hit_hours'],
                mode='markers',
                marker=dict(
                    color=self.df['event'],
                    colorscale='RdYlBu',
                    size=4,
                    opacity=0.6
                ),
                name='Acceleration',
                text=self.df['event'],
                hovertemplate='Acceleration: %{x:.2f} m/h²<br>Time to Hit: %{y:.2f}h<br>Event: %{text}'
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, title_text="Temporal Dynamics Analysis")
        fig.show()
        
        # Analyze key temporal patterns
        print("\nKey Temporal Patterns:")
        
        # Growth rate analysis
        growth_event = self.df[self.df['event'] == 1]['area_growth_rate_ha_per_h']
        growth_censored = self.df[self.df['event'] == 0]['area_growth_rate_ha_per_h']
        
        print(f"\nArea Growth Rate (ha/h):")
        print(f"  Events: Mean={growth_event.mean():.3f}, Median={growth_event.median():.3f}")
        print(f"  Censored: Mean={growth_censored.mean():.3f}, Median={growth_censored.median():.3f}")
        
        # Closing speed analysis
        closing_event = self.df[self.df['event'] == 1]['closing_speed_m_per_h']
        closing_censored = self.df[self.df['event'] == 0]['closing_speed_m_per_h']
        
        print(f"\nClosing Speed (m/h):")
        print(f"  Events: Mean={closing_event.mean():.3f}, Median={closing_event.median():.3f}")
        print(f"  Censored: Mean={closing_censored.mean():.3f}, Median={closing_censored.median():.3f}")
        
        # Distance acceleration analysis
        accel_event = self.df[self.df['event'] == 1]['dist_accel_m_per_h2']
        accel_censored = self.df[self.df['event'] == 0]['dist_accel_m_per_h2']
        
        print(f"\nDistance Acceleration (m/h²):")
        print(f"  Events: Mean={accel_event.mean():.3f}, Median={accel_event.median():.3f}")
        print(f"  Censored: Mean={accel_censored.mean():.3f}, Median={accel_censored.median():.3f}")
        
        # Create time-based feature analysis
        print("\nTime-based Feature Analysis:")
        
        # Analyze features by time windows
        time_windows = [(0, 12), (12, 24), (24, 48), (48, 72)]
        window_labels = ['0-12h', '12-24h', '24-48h', '48-72h']
        
        key_features = ['area_growth_rate_ha_per_h', 'closing_speed_m_per_h', 
                       'radial_growth_rate_m_per_h', 'dist_accel_m_per_h2']
        
        for feature in key_features:
            print(f"\n{feature} by time window:")
            for i, (start, end) in enumerate(time_windows):
                window_data = self.df[
                    (self.df['time_to_hit_hours'] >= start) & 
                    (self.df['time_to_hit_hours'] < end)
                ]
                if len(window_data) > 0:
                    mean_val = window_data[feature].mean()
                    std_val = window_data[feature].std()
                    event_rate = window_data['event'].mean()
                    print(f"  {window_labels[i]}: {mean_val:.3f} ± {std_val:.3f}, Event Rate: {event_rate:.1%}")
    
    def feature_importance_survival(self):
        """Quick Cox model to assess feature importance"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE (COX MODEL)")
        print("="*50)
        
        # Prepare data for Cox model
        cox_data = self.df[self.numeric_features + ['time_to_hit_hours', 'event']].copy()
        
        # Handle missing values
        cox_data = cox_data.fillna(cox_data.median())
        
        # Remove any constant features
        constant_features = []
        for col in self.numeric_features:
            if cox_data[col].std() == 0:
                constant_features.append(col)
        
        if constant_features:
            print(f"Removing constant features: {constant_features}")
            self.numeric_features = [f for f in self.numeric_features if f not in constant_features]
            cox_data = cox_data.drop(columns=constant_features)
        
        try:
            # Fit Cox model with a subset of features to avoid convergence issues
            feature_subset = self.numeric_features[:15]  # Limit features for stability
            cph = CoxPHFitter()
            cph.fit(cox_data[feature_subset + ['time_to_hit_hours', 'event']], 
                   duration_col='time_to_hit_hours', 
                   event_col='event')
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            cph.plot()
            plt.title('Cox Model Feature Importance (Hazard Ratios)')
            plt.tight_layout()
            plt.show()
            
            # Print top features
            hazard_ratios = cph.hazard_ratios_
            print("\nTop 10 Features by Hazard Ratio:")
            print(hazard_ratios.sort_values(ascending=False).head(10).to_string())
            
        except Exception as e:
            print(f"Cox model fitting failed: {e}")
            print("This is likely due to multicollinearity or insufficient events.")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*50)
        print("SURVIVAL ANALYSIS SUMMARY REPORT")
        print("="*50)
        
        summary_stats = {
            'Total Events': len(self.df),
            'Observed Events': self.df['event'].sum(),
            'Censored Events': (1 - self.df['event']).sum(),
            'Event Rate': f"{self.df['event'].mean():.1%}",
            'Mean Time to Hit': f"{self.df['time_to_hit_hours'].mean():.2f} hours",
            'Median Time to Hit': f"{self.df['time_to_hit_hours'].median():.2f} hours",
            'Time Range': f"{self.df['time_to_hit_hours'].min():.2f} - {self.df['time_to_hit_hours'].max():.2f} hours"
        }
        
        for key, value in summary_stats.items():
            print(f"{key}: {value}")
        
        # Key insights
        print("\nKey Insights:")
        
        # 1. Censoring pattern
        censoring_by_time = []
        time_windows = [(0, 12), (12, 24), (24, 48), (48, 72)]
        for start, end in time_windows:
            window_data = self.df[
                (self.df['time_to_hit_hours'] >= start) & 
                (self.df['time_to_hit_hours'] < end)
            ]
            if len(window_data) > 0:
                censoring_rate = (1 - window_data['event']).mean()
                censoring_by_time.append(censoring_rate)
        
        print(f"1. Censoring increases with time: {censoring_by_time[0]:.1%} (0-12h) -> {censoring_by_time[-1]:.1%} (48-72h)")
        
        # 2. Feature correlations
        # Find top correlated features with time_to_hit_hours
        correlations = []
        for feature in self.numeric_features:
            corr, p_val = spearmanr(self.df[feature], self.df['time_to_hit_hours'])
            if not np.isnan(corr):
                correlations.append((feature, abs(corr)))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        print(f"2. Most correlated features with time to hit: {correlations[0][0]}, {correlations[1][0]}")
        
        # 3. Temporal patterns
        positive_closing = (self.df['closing_speed_m_per_h'] > 0).mean()
        print(f"3. {positive_closing:.1%} of fires show positive closing speed (moving toward evacuation zones)")
        
        # 4. Growth patterns
        high_growth = (self.df['area_growth_rate_ha_per_h'] > self.df['area_growth_rate_ha_per_h'].median()).mean()
        print(f"4. {high_growth:.1%} of fires have above-median growth rates")
        
        print("\nRecommendations for Modeling:")
        print("1. Consider feature selection to reduce multicollinearity")
        print("2. Focus on rate-of-change features (growth rates, closing speeds)")
        print("3. Account for temporal resolution quality in models")
        print("4. Consider stratification by initial fire size or distance")
    
    def run_full_eda(self):
        """Execute complete EDA pipeline"""
        print("Starting Wildfire Survival Analysis EDA...")
        
        self.load_data()
        self.target_analysis()
        self.kaplan_meier_analysis()
        self.feature_correlation_analysis()
        self.temporal_dynamics_analysis()
        self.feature_importance_survival()
        self.generate_summary_report()
        
        print("\nEDA Complete! Check generated plots and summary above.")

# Execute the analysis
if __name__ == "__main__":
    # Initialize and run EDA
    eda = WildfireSurvivalEDA('data/raw/train.csv')
    eda.run_full_eda()
