#======================
#import library

#======================
import pandas as pd
import xarray as xr

#======================
#main code
#
#======================

class DataLoader:
    def __init__(self, config):
        self.config = config

    def load(self, file_path):
        if self.config["data_type"] == "netcdf":
            return self.read_netcdf(file_path)
        elif self.config["data_type"] == "csv":
            return self.read_csv(file_path)
        else:
            return ValueError(f"Unsupported data type")

    def read_netcdf(self, file_path: str) -> xr.Dataset:
        return xr.open_dataset(file_path)

    def read_csv(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)


















