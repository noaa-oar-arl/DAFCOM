# Import Library
import matplotlib.pyplot as plt

# Define Class and Functions
class Visualization:
   def __init__(self, config):
      self.config = config

   def plot_timeseries(self, df, x_col, y_col, title=None):
      plt.figure(figsize=(10,5))



