from src.forecast.trainer import ModelTrainer
from src.forecast.config import load_config
from forecast import DataLoader 

#########
#section 0. load config
#########
def main():
    config = load_config('config.yaml')

    #use this cause i ahve __init__ in src/forecast and already set DataLoader
    df = DataLoader.load(path_obs)



if __name__= '__main__':
    main()

















