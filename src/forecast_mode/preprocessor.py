# Import Library
import pandas as pd
from pandera import DataFrameSchema, Column, Check


# Define Class and Functions
class DataValidator:
    def __init__(self, schema: DataFrameSchema):
        self.schema = schema
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.schema.validate(df)

