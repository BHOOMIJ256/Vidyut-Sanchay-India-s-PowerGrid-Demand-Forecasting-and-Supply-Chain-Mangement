import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Power Grid Demand Forecasting')
    parser.add_argument(
        '--prepare-data',
        action='store_true',
        help='Prepare the data for modeling'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the models'
    )
    return parser.parse_args()

def main():
    """Main function to run the pipeline."""
    args = parse_args()
    
    if args.prepare_data:
        logger.info("Preparing data...")
        from data.make_dataset import prepare_data
        prepare_data()
        logger.info("Data preparation complete.")
    
    if args.train:
        logger.info("Training models...")
        from models.train_model import train_and_evaluate_models, select_best_model
        results = train_and_evaluate_models()
        best_model, best_name, best_score = select_best_model(results)
        logger.info(f"Training complete. Best model: {best_name} with R²: {best_score:.4f}")

if __name__ == "__main__":
    main()
