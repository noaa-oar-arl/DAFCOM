# Import Library
#import argparse
#from src.forecast.trainer import ModelTrainer

# Define Class and Functions
#def main(config_path):
#    trainer = ModelTrainer(config_path)
#    trainer.run()

#if __name__ == '__main__"
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--config", required=True)
#    args = parser.parse_args()
#    main(args.config)


#============================
#
#============================

from forecast import DataLoader 
#use this cause i ahve __init__ in src/forecast and already set DataLoader


type_obs = 'csv'
path_obs = 'folder_obs/example.csv'

def main():
    df = DataLoader.load(path_obs)
    print(df.head())





if __name__= '__main__':
    main()

















