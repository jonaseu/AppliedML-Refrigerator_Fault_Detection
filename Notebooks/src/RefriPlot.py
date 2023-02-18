import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random 

def RefriPlot_PlotTemperature(log,title = ""):
    """
    This function plots the preprocessed temperature data.
    
    Parameters:
    log (pandas DataFrame): Preprocessed log data
    title (str, optional): Title of the plot. Default is "".
    
    Returns:
    None
    """
    fig = plt.figure(figsize=(12, 4))
    if (title != ""): fig.suptitle(title)

    # Plot Main Temperature chart
    temp_cols = log.columns[2:]
    plt.plot(log['test time (m)'],log[temp_cols],label=temp_cols)
    plt.legend(loc='best')
    plt.ylabel('Temperature [°C]')
    plt.xlabel('Time [min]')
    
    plt.show()


def RefriPlot_PlotCorrelationAndBoxPlot(log,title = ""):
    """
    This function plots the preprocessed log data.
    The plot consists of 2 subplots:
        1. Heatmap of the correlation matrix of the temperature data
        2. Box plot of the temperature data
    
    Parameters:
    log (pandas DataFrame): Preprocessed log data
    title (str, optional): Title of the plot. Default is "".
    
    Returns:
    None
    """
    
    fig = plt.figure(figsize=(12, 4))
    if (title != ""): fig.suptitle(title)
    gs = fig.add_gridspec(1,2)
    ax2 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[0, 1])

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
