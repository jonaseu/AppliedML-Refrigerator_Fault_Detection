import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random 


def RefriPlot_PlotPreprocessedLog(log,title = ""):
    """
    This function plots the preprocessed log data.
    The plot consists of 3 subplots:
        1. Main temperature chart that displays the temperature trends of all columns (except first two that should be time)
        2. Heatmap of the correlation matrix of the temperature data
        3. Box plot of the temperature data
    
    Parameters:
    log (pandas DataFrame): Preprocessed log data
    title (str, optional): Title of the plot. Default is "".
    
    Returns:
    None
    """
    
    fig = plt.figure(figsize=(12, 8))
    if (title != ""): fig.suptitle(title)
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # Plot Main Temperature chart
    temp_cols = log.columns[2:]
    ax1.plot(log['test time (m)'],log[temp_cols],label=temp_cols)
    ax1.legend(loc='best')
    ax1.set_ylabel('Temperature [°C]')
    ax1.set_xlabel('Time [min]')

    # Plot correlation matrix 
    temperature_data = log.iloc[:, 2:]
    corr = temperature_data.corr()
    sns.heatmap(corr,ax=ax2, annot=True)
    ax2.set_title("Correlation Matrix")
    
    # Plot box plot of the temperature data
    sns.boxplot(data=temperature_data, ax=ax3)
    ax3.set_title("Temperatures Distribution")
    ax3.set_ylabel('Temperature [°C]')
    
    plt.tight_layout()
    plt.show()