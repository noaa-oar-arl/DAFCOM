# Import Library
import matplotlib.pyplot as plt

# Define Class and Functions
class Visualization:
    def __init__(self, config):
        self.config = config

    def plot_timeseries(self, df, x_col, y_col, title=None):
        plt.figure(figsize=(10,5))
        plt.plot(df[x_col],df[y_col])
        plt.title(title or f"{y_col} over {x_col})
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()



