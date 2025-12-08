import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    make_scorer
)
import mlflow
import mlflow.sklearn
from datetime import datetime

from ..config import MODEL_DIR, RANDOM_SEED

# Set up MLflow
tracking_uri = "file:///" + str(Path(__file__).parents[2] / "mlruns")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("power_grid_demand_forecasting")

def load_processed_data():
    """Load the processed data."""
    from ..config import PROCESSED_DATA_DIR, PROCESSED_DATA_FILE
    data = joblib.load(PROCESSED_DATA_DIR / PROCESSED_DATA_FILE)
    return (
        data['X_train'], data['X_test'],
        data['y_train'], data['y_test'],
        data['feature_names'], data['preprocessor']
    )

def get_models():
    """Get a list of models to evaluate."""
    models = {
        'RandomForest': MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=100,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
        ),
        'ExtraTrees': MultiOutputRegressor(
            ExtraTreesRegressor(
                n_estimators=100,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
        ),
        'XGBoost': MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                enable_categorical=False
            )
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=100,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1
        ),
        'CatBoost': CatBoostRegressor(
            iterations=100,
            random_seed=RANDOM_SEED,
            verbose=0,
            allow_writing_files=False
        )
    }
    return models

def evaluate_model(model, X_test, y_test, name):
    """Evaluate a model and return metrics."""
    y_pred = model.predict(X_test)
    
    # Calculate metrics for each target
    metrics = {}
    for i, col in enumerate(y_test.columns):
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        
        metrics[f'{col}_mae'] = mae
        metrics[f'{col}_rmse'] = rmse
        metrics[f'{col}_r2'] = r2
    
    # Calculate average metrics
    metrics['avg_mae'] = np.mean([metrics[f'{col}_mae'] for col in y_test.columns])
    metrics['avg_rmse'] = np.mean([metrics[f'{col}_rmse'] for col in y_test.columns])
    metrics['avg_r2'] = np.mean([metrics[f'{col}_r2'] for col in y_test.columns])
    
    print(f"\n{name} Model Evaluation:")
    print(f"Average MAE: {metrics['avg_mae']:.4f}")
    print(f"Average RMSE: {metrics['avg_rmse']:.4f}")
    print(f"Average R²: {metrics['avg_r2']:.4f}")
    
    return metrics

def train_and_evaluate_models():
    """Train and evaluate multiple models."""
    # Load processed data
    X_train, X_test, y_train, y_test, feature_names, preprocessor = load_processed_data()
    
    # Get models
    models = get_models()
    
    # Dictionary to store results
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        with mlflow.start_run(run_name=f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            metrics = evaluate_model(model, X_test, y_test, name)
            
            # Log parameters and metrics
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            
            # Log the model
            mlflow.sklearn.log_model(model, f"{name.lower()}_model")
            
            # Save results
            results[name] = {
                'model': model,
                'metrics': metrics
            }
            
            # Save the model
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model_path = MODEL_DIR / f"{name.lower()}_model.joblib"
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")
    
    return results

def select_best_model(results):
    """Select the best model based on average R² score."""
    best_score = -np.inf
    best_model = None
    best_name = ""
    
    for name, result in results.items():
        if result['metrics']['avg_r2'] > best_score:
            best_score = result['metrics']['avg_r2']
            best_model = result['model']
            best_name = name
    
    print(f"\nBest model: {best_name} with average R²: {best_score:.4f}")
    return best_model, best_name, best_score

if __name__ == "__main__":
    # Train and evaluate models
    results = train_and_evaluate_models()
    
    # Select and save the best model
    best_model, best_name, best_score = select_best_model(results)
    
    # Save the best model
    best_model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_model, best_model_path)
    print(f"Best model saved to {best_model_path}")
