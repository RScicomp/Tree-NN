import pandas as pd
import numpy as np


def get_float_array(signal_arr):
    signal_arr = signal_arr.replace("[","").replace("]","").split()
    return [float(i) for i in signal_arr]

# df = pd.read_csv("dataset.csv")
# signals = df.sensor_signals.apply(get_float_array)
# truth = df.action.apply(float)


