from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_DIR = BASE_DIR / 'models'

# Data file names
RAW_DATA_FILE = 'power_grid_data_expanded(1).csv'
PROCESSED_DATA_FILE = 'processed_data.pkl'

# Random seed for reproducibility
RANDOM_SEED = 42

# Test size for train-test split
TEST_SIZE = 0.2

# Target columns
TARGET_COLUMNS = [
    'steel_tonnes',
    'conductor_km',
    'insulators_unit',
    'concrete_cubic_meter',
    'Bus_reactor',
    'Transformers',
    'circuit_breaker'
]

# Columns to drop
DROP_COLUMNS = [
    'state',
    'project_name',
    'start_date',
    'planned_duration_months',
    'inflation_rate',
    'steel_price_index',
    'conductor_price_index',
    'fuel_price_index'
]

# Categorical columns for one-hot encoding
CATEGORICAL_COLUMNS = [
    'project_type',
    'region',
    'soil_type',
    'terrain_type'
]

# Numerical columns
NUMERICAL_COLUMNS = [
    'voltage_kv',
    'Length_km',
    'num_towers'
]
