#===================
# Import Library
#===================

import pandas as pd
#from pandera import DataFrameSchema, Column, Check
#import dask.dataframe as dd



#===========================
# Define Class and Functions
#===========================

#class DataValidator:
#    def __init__(self, schema: DataFrameSchema):
#        self.schema = schema
#    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
#        return self.schema.validate(df)


class Preprocessor:
    def __init__(self, config):
        self.config = config
    def transform(self, data):
        #implement preprocessing logic
        pass
    def drop():
        pass
    def duplicate():
        pass


class Input_Preprocessor:
    def __init__(self, config):
        self.config = config

    def obs_input_prep:
        pass

    def model_input_prep():
        pass
























