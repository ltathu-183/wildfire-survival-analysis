# Wildfire Survival Analysis - WiDS Global Datathon 2026

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This repository contains a comprehensive survival analysis pipeline for wildfire threat prediction, developed for the WiDS Global Datathon 2026. The project leverages machine learning techniques to predict the time until a wildfire threatens an evacuation zone, based on the first 5 hours of perimeter data. The primary objective is to forecast survival probabilities at 12-hour, 24-hour, 48-hour, and 72-hour horizons.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Data](#data)
- [Model](#model)
- [Evaluation](#evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Survival Analysis**: Utilizes Random Survival Forests for time-to-event prediction.
- **Feature Engineering**: Implements Oracle-inspired feature transformations for enhanced predictive power.
- **Cross-Validation**: 5-fold CV with custom hybrid scoring metric.
- **Modular Codebase**: Clean, reusable Python modules for data processing, modeling, and evaluation.
- **Jupyter Notebooks**: Interactive notebooks for EDA, feature engineering, and modeling.
- **Automated Pipeline**: End-to-end pipeline via `main.py` for streamlined execution.

## Project Structure

```
├── data/
│   ├── raw/                 # Raw dataset files
│   └── processed/           # Processed data outputs
├── notebooks/               # Jupyter notebooks for analysis
│   ├── 1_eda.ipynb          # Exploratory Data Analysis
│   ├── 2_feature_engineering.ipynb  # Feature Engineering
│   └── 3_modeling.ipynb     # Model Training and Submission
├── outputs/                 # Generated reports and submissions
├── src/                     # Core Python modules
│   ├── data.py              # Data loading utilities
│   ├── feature_engineering.py  # Feature engineering and scaling
│   ├── modeling.py          # Modeling pipeline
│   ├── evaluation.py        # Custom evaluation metrics
│   └── preprocessing.py     # Data preprocessing
├── main.py                  # Main pipeline entrypoint
├── pyproject.toml           # Project configuration
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Data

The dataset comprises wildfire perimeter data collected during the initial 5 hours of fire progression. Key features include:

- Area metrics (e.g., area_first_ha, area_growth)
- Distance and speed measurements (e.g., dist_min_ci_0_5h, closing_speed)
- Spatial attributes (e.g., alignment, centroid displacement)
- Temporal indicators (e.g., event_start_hour, event_start_dayofweek)

**Target Variable**: Time to hit evacuation zone (time_to_hit_hours) with event indicator.

- **Training Data**: `data/raw/train.csv`
- **Test Data**: `data/raw/test.csv`
- **Sample Submission**: `data/raw/sample_submission.csv`

## Model

The core model is a **Random Survival Forest (RSF)** implemented using `scikit-survival`. Key components:

- **Feature Engineering**: Oracle-inspired transformations including log-scaling, volatility measures, and momentum calculations.
- **Preprocessing**: Robust scaling for numerical stability.
- **Hyperparameter Tuning**: Optimized via cross-validation.

## Evaluation

Model performance is assessed using a custom hybrid metric:

**Hybrid Score = 0.3 × Concordance Index + 0.7 × (1 - Weighted Brier Score)**

- **Concordance Index (C-Index)**: Measures ranking quality of predictions.
- **Weighted Brier Score**: Evaluates calibration at time horizons (12h, 24h, 48h, 72h) with weights [0.15, 0.3, 0.4, 0.15].

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd wildfire-survival-analysis
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Complete Pipeline
Execute the end-to-end pipeline:
```bash
python main.py
```

### Run Individual Notebooks
Launch Jupyter and execute notebooks in sequence:
1. `notebooks/1_eda.ipynb` - Data exploration and visualization
2. `notebooks/2_feature_engineering.ipynb` - Feature creation and preprocessing
3. `notebooks/3_modeling.ipynb` - Model training and submission generation

### Generate Submission
The final submission file (`submission.csv`) is saved in the `outputs/` directory with survival probabilities for each time horizon.

## Results

The trained model achieves a cross-validation hybrid score of approximately 0.94. Submission predictions are calibrated survival probabilities at 12h, 24h, 48h, and 72h intervals.

Sample output:
```
event_id,prob_12h,prob_24h,prob_48h,prob_72h
10662602,0.85,0.72,0.58,0.45
...
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- WiDS Global Datathon 2026 organizers for the challenge and dataset
- `scikit-survival` library for survival analysis tools
- Open-source community for invaluable resources
- `notebooks/1_eda.ipynb`
- `notebooks/2_feature_engineering.ipynb`
- `notebooks/3_modeling.ipynb`

## Recommended Workflow

1. Explore and validate the raw data in `notebooks/1_eda.ipynb`
2. Build Oracle features and scale them in `notebooks/2_feature_engineering.ipynb`
3. Train and evaluate the final model in `notebooks/3_modeling.ipynb`

## Key Components

### Feature Engineering
- Oracle features derived from fire gravity, volatility, and temporal signals
- Robust scaling to protect against outliers
- VIF-based multicollinearity diagnostic

### Modeling
- Random Survival Forest for final production model
- Cox PH baseline for econometric insights
- Hybrid score combining C-index and weighted Brier score

### Evaluation
- Time-aware Brier score across [12, 24, 48, 72] hours
- Survival prediction calibration and risk scoring

## Notes

- Keep `src/` as the main reusable codebase for modeling and preprocessing.
- The new notebooks are the recommended analysis path.
- Old duplicate notebooks have been cleaned up for better maintainability.
