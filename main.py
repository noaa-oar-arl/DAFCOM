# Import Library
import argparse
from src.forecast.trainer import ModelTrainer

# Define Class and Functions
def main(config_path):
    trainer = ModelTrainer(config_path)
    trainer.run()

if __name__ == '__main__"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)




















