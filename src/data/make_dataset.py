import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ..config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_FILE, PROCESSED_DATA_FILE,
    TARGET_COLUMNS, DROP_COLUMNS, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS,
    RANDOM_SEED, TEST_SIZE
)

def load_raw_data():
    """Load the raw dataset."""
    file_path = RAW_DATA_DIR / RAW_DATA_FILE
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the raw data."""
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Drop unnecessary columns
    df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns])
    
    # Handle missing values if any
    df = handle_missing_values(df)
    
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    # For numerical columns, fill with median
    num_cols = [col for col in NUMERICAL_COLUMNS if col in df.columns]
    for col in num_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    # For categorical columns, fill with mode
    cat_cols = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
    
    return df

def create_preprocessor():
    """Create a preprocessing pipeline."""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_COLUMNS),
            ('cat', categorical_transformer, CATEGORICAL_COLUMNS)
        ])
    
    return preprocessor

def get_feature_names(preprocessor, X):
    """Get feature names after preprocessing."""
    # Get feature names from the preprocessor
    feature_names = []
    
    # Add numerical features
    feature_names.extend(NUMERICAL_COLUMNS)
    
    # Add one-hot encoded categorical features
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_features = cat_encoder.get_feature_names_out(CATEGORICAL_COLUMNS)
    feature_names.extend(cat_features)
    
    return feature_names

def prepare_data():
    """Main function to prepare the data for modeling."""
    # Create directories if they don't exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_raw_data()
    df = preprocess_data(df)
    
    # Separate features and target
    X = df.drop(columns=TARGET_COLUMNS)
    y = df[TARGET_COLUMNS]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    
    # Create and fit preprocessor
    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names
    feature_names = get_feature_names(preprocessor, X_train)
    
    # Save processed data and preprocessor
    joblib.dump({
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'preprocessor': preprocessor
    }, PROCESSED_DATA_DIR / PROCESSED_DATA_FILE)
    
    print(f"Processed data saved to {PROCESSED_DATA_DIR / PROCESSED_DATA_FILE}")
    
    return {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'preprocessor': preprocessor
    }

if __name__ == "__main__":
    data = prepare_data()
