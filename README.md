# Vidyut Sanchay: Power Grid Demand Forecasting

## Project Overview
Vidyut Sanchay is a machine learning project focused on forecasting materials demand for power grid infrastructure projects. The system helps in supply chain planning, procurement, and inventory optimization for POWERGRID projects across India.

## Features
- **Data Preprocessing**: Handles missing values, encodes categorical variables, and scales numerical features
- **Multiple Models**: Implements and compares various regression models
- **Model Evaluation**: Provides comprehensive metrics for model performance
- **MLflow Integration**: Tracks experiments and manages model versions
- **Modular Design**: Easy to extend and maintain

## Project Structure
```
vidyut_sanchay/
├── data/
│   ├── raw/                   # Raw data files
│   └── processed/             # Processed data files
├── models/                    # Trained models
├── notebooks/                 # Jupyter notebooks for exploration
├── src/                       # Source code
│   ├── data/                  # Data loading and preprocessing
│   ├── features/              # Feature engineering
│   ├── models/                # Model definitions and training
│   └── visualization/         # Visualization utilities
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd vidyut_sanchay
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation
1. Place your raw data file (`power_grid_data_expanded(1).csv`) in the `data/raw/` directory.

2. Run the data preparation script:
   ```bash
   python -m src.main --prepare-data
   ```

### Model Training
To train the models:
```bash
python -m src.main --train
```

This will:
1. Load and preprocess the data
2. Train multiple regression models
3. Evaluate each model's performance
4. Save the best model to the `models/` directory
5. Log all experiments using MLflow

### Viewing Results
To view the MLflow dashboard:
```bash
mlflow ui
```
Then open `http://localhost:5000` in your browser.

## Model Evaluation
Models are evaluated using the following metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score (Coefficient of Determination)

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- POWERGRID Corporation of India Limited
- Open-source machine learning community