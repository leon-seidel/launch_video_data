# Plot existing data from the csv files in /mission_data

import pandas as pd
from tkinter import Tk
import tkinter.filedialog
from matplotlib import pyplot as plt

Tk().withdraw()
filename = tkinter.filedialog.askopenfilename()
df = pd.read_csv(filename)
col_number = df.shape[1]
plot_type = "scatter"
data_types = ["velo", "alti", "acc"]

for data_type in data_types:
    plt.figure()
    print()
    if plot_type == "plot" and col_number == 7:
        if data_type == "velo":
            plt.plot(df.t, df.v1)
            plt.plot(df.t, df.v2)
        elif data_type == "alti":
            plt.plot(df.t, df.h1)
            plt.plot(df.t, df.h2)
        elif data_type == "acc":
            plt.plot(df.t, df.a1)
            plt.plot(df.t, df.a2)

    elif plot_type == "scatter" and col_number == 7:
        if data_type == "velo":
            plt.scatter(df.t, df.v1)
            plt.scatter(df.t, df.v2)
        elif data_type == "alti":
            plt.scatter(df.t, df.h1)
            plt.scatter(df.t, df.h2)
        elif data_type == "acc":
            plt.scatter(df.t, df.a1)
            plt.scatter(df.t, df.a2)

    elif plot_type == "plot" and col_number == 4:
        if data_type == "velo":
            plt.plot(df.t, df.v)
        elif data_type == "alti":
            plt.plot(df.t, df.h)
        elif data_type == "acc":
            plt.plot(df.t, df.a)

    elif plot_type == "scatter" and col_number == 4:
        if data_type == "velo":
            plt.scatter(df.t, df.v)
        elif data_type == "alti":
            plt.scatter(df.t, df.h)
        elif data_type == "acc":
            plt.scatter(df.t, df.a)

    if data_type == "velo":
        plt.title("Time vs. velocity")
        plt.xlabel("Time in s")
        plt.ylabel("Velocity in kph")
    elif data_type == "alti":
        plt.title("Time vs. altitude")
        plt.xlabel("Time in s")
        plt.ylabel("Altitude in km")
    elif data_type == "acc":
        plt.title("Time vs. acceleration")
        plt.xlabel("Time in s")
        plt.ylabel("Acceleration in m/s^2")

    if col_number == 7:
        plt.legend(["Stage 1", "Stage 2"])

    plt.grid()

plt.show()
