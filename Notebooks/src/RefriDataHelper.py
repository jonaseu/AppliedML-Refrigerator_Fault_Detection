import os
import pandas as pd
import pathlib


def RefriDataHelper_ReadParsedLogs():
    """
    This function reads all the .csv files in Data/Pre_Processed folder, converts them into pandas data frames, and returns a dictionary 
    with filenames as keys and data frames as values.

    Returns:
    dict: Dictionary with filenames as keys and data frames as values.
    """

    data_frames = {}
    
    folder_path = str(pathlib.Path(__file__).parent.resolve()) + "\\..\\..\\Data\\Pre_Processed\\Temperature_Logs\\"

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.replace(' temp', '')            #Just simplify the column names TODO: Improve on the next phases of data pre processing
            df.columns = df.columns.str.replace('w ','')             
            data_frames[filename] = df
    
    return data_frames
