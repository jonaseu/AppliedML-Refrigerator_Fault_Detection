from matplotlib.pyplot import plot, Line2D,Rectangle,Text
import pandas as pd
import glob
import matplotlib.pylab as plt
from statsmodels.graphics.tsaplots import plot_acf,acf
import scipy.signal as ss
from collections import Counter
from scipy.fft import fft, fftfreq
import numpy as np
import math
import seaborn as sns
import mplcursors
import statistics
import sys
import os
from scipy import signal
import yaml

#Define if should plot all available plots or not
ENABLE_DEBUG_VISUALIZATIONS = True
#Choose if the temperatures shall be converted to °C or not
TRANSFORM_FROM_F_TO_C = True

PARAMETERS = yaml.safe_load(open('Parameters.yaml','r'))
#===============================================================================================================
#DEFINES
#===============================================================================================================

TEST_TIME_COLUMN_NAME = 'test time (s)'

SECONDS_IN_HOUR = 3600

number_of_active_cursors = 0
active_cursors = []
previous_cursor_time = 0

fft_active_cursors = {}
#===============================================================================================================
#CODE
#===============================================================================================================
def Get_Seconds_FromStringHMS(time_str):
    """Get Seconds from time."""
    
    #Some logs come with format h:mm:ss.ms instead of h:mm:ss, so remove .ms part
    time_str = time_str.split(".", 1)[0]

    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def Visualize_BoxAndHist(data,xlabel,title=''):
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    sns.boxplot(data, ax=ax_box)
    sns.histplot(data=data, ax=ax_hist,bins=30)
    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')
    ax_hist.set(xlabel=xlabel)
    ax_hist.set(ylabel='Ocurrences')
    f.suptitle(title)
    f.canvas.set_window_title(title)
    mplcursors.cursor(hover=True)
    plt.plot()
    plt.savefig(PARAMETERS['PATHS']['OUTPUT_FIGURES_PATH']+title)

def Visualize_SimpleTemperatureCharts(log,title=''):
    
    def onClickShowXDelta(event):
        global number_of_active_cursors
        global active_cursors
        global previous_cursor_time
        number_of_active_cursors = number_of_active_cursors + 1
        if(number_of_active_cursors <= 2):
            color = 'g'
            if(number_of_active_cursors == 2):
                color = 'r'
                delta = event.xdata - previous_cursor_time
                active_cursors.append(plt.text(previous_cursor_time, event.ydata, ' T = {:.0f}m \n f = {:.2e}Hz'.format(delta,1/(delta*60)), fontsize = 16,bbox = dict(facecolor = 'red', alpha = 0.5)))
            active_cursors.append(plt.axvline(event.xdata,linestyle='--',linewidth=2,color=color))
            active_cursors.append(plt.text(event.xdata, event.inaxes.axes.get_ylim()[0], ' {:.0f}m'.format(event.xdata), fontsize = 6,color=color))
            previous_cursor_time = event.xdata
        else:
            for cursor in active_cursors:
                cursor.remove()
            active_cursors = []
            number_of_active_cursors = 0
        plt.show()

    fig = plt.figure()
    for col in EXPECTED_COLUMNS:
        plt.plot(log['test time (m)'],log[col],label=col, picker=True)
    
    plt.title(title)
    plt.legend(EXPECTED_COLUMNS)
    plt.ylabel('Temperature °C')
    plt.ylim([-40,40])
    plt.xlabel('Test Time (m)')
    fig.canvas.callbacks.connect('button_release_event', onClickShowXDelta)
    plt.savefig(PARAMETERS['PATHS']['OUTPUT_FIGURES_PATH']+title)


def Visualize_CCFOfDesiredColumns(log,title=''):
    
    def ccf(x, y, lag_max = 100):

        result = ss.correlate(y - np.mean(y), x - np.mean(x), method='direct') / (np.std(y) * np.std(x) * len(y))
        length = (len(result) - 1) // 2
        lo = length - lag_max
        hi = length + (lag_max + 1)

        return result[lo:hi]

    number_of_rows = int(math.ceil((len(EXPECTED_COLUMNS) -1)/2 ))
    for col in EXPECTED_COLUMNS:
        fig = plt.figure()
        subplot_counter = 1
        for col_to_cor in EXPECTED_COLUMNS:
            if(col_to_cor !=  col):
                sub_plot = fig.add_subplot(number_of_rows,2,subplot_counter)
                ccf_output = ccf(log[col],log[col_to_cor])
                sub_plot.plot(ccf_output)
                sub_plot.title.set_text('CCF - {}'.format(col_to_cor))
            subplot_counter += 1

        fig.suptitle(title + ' - {}'.format(col))
        fig.tight_layout()

def Visualize_ACFOfDesiredColumns(log,title='',lags=PARAMETERS['FEATURE_EXTRACTION']['ACF_NUMBER_OF_LAGS']):
    
    fig = plt.figure()

    number_of_rows = int((math.ceil(len(EXPECTED_COLUMNS)))/2)
    subplot_counter = 1
    for col in EXPECTED_COLUMNS:
        sub_plot = fig.add_subplot(number_of_rows,2,subplot_counter)
        plot_acf(log[col],lags=lags,ax=sub_plot)
        sub_plot.title.set_text('ACF - {}'.format(col))
        subplot_counter += 1
    
    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(PARAMETERS['PATHS']['OUTPUT_FIGURES_PATH']+title)

def Visualize_SpectogramOfDesiredColumns(log,title=''):

    fig = plt.figure()
    #fig2 = plt.figure()
    number_of_rows = int(math.ceil(len(EXPECTED_COLUMNS)/2))
    subplot_counter = 1
    for col in EXPECTED_COLUMNS:
        sub_plot = fig.add_subplot(number_of_rows,2 if len(EXPECTED_COLUMNS) > 1 else 1,subplot_counter)
        #sub_plot2 = fig2.add_subplot(number_of_rows,2 if len(EXPECTED_COLUMNS) > 1 else 1,subplot_counter)
        f, t, Sxx = signal.spectrogram(log[col], 1/50,nperseg=256,noverlap=256 - 1 )

        #Filter desired frequencies
        filtered_f = []
        filtered_Sxx = []
        for id,value in enumerate(f):
            if(f[id] >= PARAMETERS['FEATURE_EXTRACTION']['MININUM_FREQUENCY'] and f[id] <= PARAMETERS['FEATURE_EXTRACTION']['MAXIMUM_FREQUENCY']):
                filtered_f.append(f[id])
                filtered_Sxx.append(Sxx[id])
        f = filtered_f
        Sxx = filtered_Sxx

        t = t/60 #Time to minutes so we can match other charts
        sub_plot.pcolormesh(t, np.array(f), np.array(Sxx))
        #sub_plot2.specgram(log[col], Fs=1/50)
        sub_plot.title.set_text('Spectogram - {}'.format(col))
        subplot_counter += 1
    
    mplcursors.cursor(hover=True)
    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(PARAMETERS['PATHS']['OUTPUT_FIGURES_PATH']+title)



def Visualize_FFTOfDesiredColumns(log,title=''):
    
    fig = plt.figure()
    number_of_rows = int(math.ceil(len(EXPECTED_COLUMNS)/2))
    subplot_counter = 1
    for col in EXPECTED_COLUMNS:
        sub_plot = fig.add_subplot(number_of_rows,2,subplot_counter)
        fft_y = fft(log[col].values)
        fft_f = fftfreq(len(log[col]), log[TEST_TIME_COLUMN_NAME][1] - log[TEST_TIME_COLUMN_NAME][0] )
        
        #Filter desired frequencies
        filtered_fft_y = []
        filtered_fft_f = []
        for id,value in enumerate(fft_y):
            if(fft_f[id] >= PARAMETERS['FEATURE_EXTRACTION']['MININUM_FREQUENCY'] and fft_f[id] <= PARAMETERS['FEATURE_EXTRACTION']['MAXIMUM_FREQUENCY']):
                filtered_fft_y.append(fft_y[id])
                filtered_fft_f.append(fft_f[id])
        fft_y = filtered_fft_y
        fft_f = filtered_fft_f

        sub_plot.plot( np.array(fft_f), np.abs(fft_y))

        sub_plot.title.set_text('FFT - {}'.format(col))
        subplot_counter += 1
    
    mplcursors.cursor(hover=True)
    fig.suptitle(title)
    def onClickShowFrequency(event):
        global fft_active_cursors
        chart_pressed = event.inaxes
        if chart_pressed in fft_active_cursors:
            fft_active_cursors[chart_pressed][0].remove()
            fft_active_cursors[chart_pressed][1].remove()
            del fft_active_cursors[chart_pressed]
        else:
            fft_active_cursors[chart_pressed] = ['','']
            fft_active_cursors[chart_pressed][0] = event.inaxes.text(0.5, 0.1,' T = {:.1f}m'.format(1/event.xdata/60),bbox = dict(facecolor = 'red', alpha = 0.5),
                                                        horizontalalignment='center', verticalalignment='center', transform=event.inaxes.transAxes)
            fft_active_cursors[chart_pressed][1] = event.inaxes.plot(event.xdata,event.ydata,'ro')[0]
        plt.show()


    fig.canvas.callbacks.connect('button_release_event', onClickShowFrequency)
    fig.tight_layout()
    plt.savefig(PARAMETERS['PATHS']['OUTPUT_FIGURES_PATH']+title)


def Visualize_CompletePreProcessedData(df_collection):
        
    IGNORE_PERIOD = 0

    complete_df = pd.DataFrame(columns=df_collection[list(df_collection.keys())[0]].columns)
    columns_acfs = {key:[] for key in EXPECTED_COLUMNS}
    columns_acf_average = {key:[] for key in EXPECTED_COLUMNS}

    #Creates a complete df with data from all logs and also get acf for each log
    summarized_dataset = {}
    print("Starting Summary of the complete Pre Processed Data")
    count = 1
    for df_key in df_collection:
        df = df_collection[df_key]
        df = df.loc[df['test time (s)'] >= IGNORE_PERIOD]
        complete_df = complete_df.append(df, ignore_index=True)

        summarized_dataset[df_key]  = []

        for col in EXPECTED_COLUMNS: 
            columns_acfs[col].append( acf(df[col],nlags=PARAMETERS['FEATURE_EXTRACTION']['ACF_NUMBER_OF_LAGS']) )
            
            column_statistics = df[col].describe()
            for value in column_statistics:
                summarized_dataset[df_key].append(value)
        
        print('Percentage {:.2%}'.format(count/len(df_collection.keys())),end='\r' )
        count += 1

    summarized_column_names = []
    summarized_dataset['TOTAL'] = []
    for col in EXPECTED_COLUMNS: 
        column_statistics = complete_df[col].describe()
        for key,value in enumerate(column_statistics):
            summarized_column_names.append(col + '_' + column_statistics.index[key])
            summarized_dataset['TOTAL'].append(value)

    pd.DataFrame.from_dict(summarized_dataset,orient='index',columns=summarized_column_names).to_csv(PARAMETERS['PATHS']['OUTPUT_PATH']+'_Pre Processed - Summarized Dataset.csv')

    #Plot the ACF avereaged for each column
    number_of_rows = int((math.ceil(len(EXPECTED_COLUMNS)))/2)
    fig = plt.figure()
    subplot_counter = 1
    for col in EXPECTED_COLUMNS: 
        df_col_acfs = pd.DataFrame(columns_acfs[col])
        columns_acf_average[col] = df_col_acfs.mean()
        
        sub_plot = fig.add_subplot(number_of_rows,2,subplot_counter)
        sub_plot.stem(columns_acf_average[col])
        sub_plot.title.set_text('ACF AVERAGED - {}'.format(col))
        subplot_counter += 1
    
    fig_title = 'FFT - Amplitude x Freq (Hz) - Complete Dataset'
    print('Ploting '+fig_title+'...') 
    Visualize_FFTOfDesiredColumns(complete_df,fig_title)

    fig_title = 'ACF Averaged of Temperatures'  
    print('Ploting '+fig_title+'...') 
    mplcursors.cursor(hover=True)
    fig.suptitle(fig_title)
    fig.tight_layout()
    plt.savefig(PARAMETERS['PATHS']['OUTPUT_FIGURES_PATH']+fig_title)

    #Plot Correlation Matrix
    fig_title = 'Correlation Matrix of Temperatures - Complete Dataset'
    plt.figure()
    print('Ploting '+fig_title+'...')
    sns.heatmap(complete_df.corr(), annot=True)
    plt.title(fig_title)
    plt.savefig(PARAMETERS['PATHS']['OUTPUT_FIGURES_PATH']+fig_title)
    
    #Plot violin plot of temperatures
    fig_title = 'Violin Plot of Temperatures - Complete Dataset'
    plt.figure()
    print('Ploting '+fig_title+'...')
    sns.violinplot(data=complete_df[EXPECTED_COLUMNS])
    plt.ylabel('Temperature (°C)')
    plt.xlabel('Sensor')
    plt.title(fig_title)
    plt.savefig(PARAMETERS['PATHS']['OUTPUT_FIGURES_PATH']+fig_title)
    
    plt.show()


def Visualize_Logs_Duration_And_Sample_Rate(df_collection,title=''):
    
    logs_duration = []
    logs_sample_rate = []
    for df_key in df_collection:
        try:
            #Get what is the last sample in hours and also calculate average sample rate
            log_duration = df_collection[df_key][TEST_TIME_COLUMN_NAME].iloc[-1] / SECONDS_IN_HOUR
            
            interval_between_samples = df_collection[df_key][TEST_TIME_COLUMN_NAME].diff()[1:]
            average_sample_rate = statistics.mean(interval_between_samples)

            logs_duration.append(log_duration)
            logs_sample_rate.append(average_sample_rate)
        except:
            pass

    Visualize_BoxAndHist(logs_duration,'Logs Duration (h)',title + '- Logs Duration')
    Visualize_BoxAndHist(logs_sample_rate,'Average Sample Interval (s)',title + '- Logs Sample Rate')

def Visualize__AllLogCharts(log,log_title):
    Visualize_SimpleTemperatureCharts(log,'Temperature Chart - '+ log_title)
    Visualize_ACFOfDesiredColumns(log,'ACF - ' + log_title)
    #Visualize_CCFOfDesiredColumns(log,'CCF - ' + log_title) #CCF is not showing a clear correlation, removing it for now
    Visualize_FFTOfDesiredColumns(log,'FFT - Amplitude x Freq (Hz) - ' + log_title)
    Visualize_SpectogramOfDesiredColumns(log,'Spectogram - Freq (Hz) x Log Time (m) - ' + log_title)

def Filter_Logs_Not_According_To_Duration(df_collection):

    valid_df_collection = df_collection.copy()
    SECONDS_IN_HOUR = 3600

    print("\nRemoving logs that are not according to the expected duration and number of samples")
    logs_to_remove = []
    for df_key in df_collection:
        try:
            #Get what is the last sample in hours
            log_duration = Get_Seconds_FromStringHMS(df_collection[df_key]['test time'].iloc[-1]) / SECONDS_IN_HOUR

            #Check the average sample rate  
            interval_between_samples = df_collection[df_key][TEST_TIME_COLUMN_NAME].diff()[1:]
            average_sample_rate = statistics.mean(interval_between_samples)

            #Remove logs that do not respect desired duration
            if( (log_duration < PARAMETERS['LOG_FILTERING']['MINIMAL_LOG_DURATION'] or log_duration > PARAMETERS['LOG_FILTERING']['MAXIMUM_LOG_DURATION']) or
                (average_sample_rate < PARAMETERS['LOG_FILTERING']['MINIMAL_LOG_SAMPLE_INTERVAL'] or average_sample_rate > PARAMETERS['LOG_FILTERING']['MAXIMAL_LOG_SAMPLE_INTERVAL']) ):
                logs_to_remove.append(df_key)
                print('\t\t{} removed. Duration={:.2f},Average Sample Interval={:.2f}'.format(df_key,log_duration,average_sample_rate))

        except:
            #Print more details about logs that failed to filter due to some unkonw reasons
            print('\t\t{} removed as it had some unexpected format:'.format(df_key))
            logs_to_remove.append(df_key)
            
    for log_to_remove in logs_to_remove:
        del valid_df_collection[log_to_remove]

    return(valid_df_collection)

def Filter_Logs_Not_According_To_Expected_Columns(df_collection):
    valid_df_collection = df_collection.copy()
    
    #Dictionary that will hold the values for all datasets for a given column as key
    valid_column_values = {}
    #How many logs used the given column with valid values
    valid_column_occurences = {}
    
    logs_to_remove = []
    print("\nRemoving logs that are not according to the expected columns")
    for df_key in df_collection:
        
        df = df_collection[df_key]
        df_column_names = df_collection[df_key].columns
        df_column_names = df_column_names[2:] #skip

        #Search the log for all expected columns, if missing any then remove log
        for expected_col in EXPECTED_COLUMNS:
            if(expected_col.lower() not in df_column_names):
                logs_to_remove.append(df_key)
                print('\t\t{} removed as it was missing at least expected col \'{}\''.format(df_key,expected_col))
                break

        #Loop trough each column to check if any of the expected is equal to invalid
        for col_name in df_column_names:
                        
            col_values = df[col_name].to_numpy()
            ocurrences_on_col = df[col_name].value_counts()
            if(PARAMETERS['RTS_PARAMETERS']['INVALID_READING'] in ocurrences_on_col):
                number_invalid_readings = ocurrences_on_col[PARAMETERS['RTS_PARAMETERS']['INVALID_READING']] 
                percentage_valid_readings = (len(df.index) - number_invalid_readings) / (len(df.index))

                if (col_name in EXPECTED_COLUMNS and
                    percentage_valid_readings < PARAMETERS['LOG_FILTERING']['MINIMAL_VALID_READING_PERCENTAGE']):
                    #If most column values are invalid and is in the expected columns, remove the log
                    logs_to_remove.append(df_key)
                    print('\t\t{} removed as it had invalid values for col \'{}\''.format(df_key,col_name))
            else:
                #If the values are not invalid, add it to the valid dictionary
                if(col_name in valid_column_values):
                    valid_column_values[col_name].extend(col_values.tolist())
                    valid_column_occurences[col_name] += 1
                else:
                    valid_column_occurences[col_name] = 1
                    valid_column_values[col_name] = col_values.tolist()

    #Remove the defined logs
    for log_to_remove in logs_to_remove:
        if(log_to_remove in valid_df_collection):
            del valid_df_collection[log_to_remove] 

    #Visualize most common valid columns. Not easy to visualize but can help out on checking most common columns
    if(ENABLE_DEBUG_VISUALIZATIONS):
        plt.figure()
        valid_column_occurences = pd.DataFrame( list(Counter(valid_column_occurences).items()) , columns = ['Column Name','Ocurrences'])
        valid_column_occurences.sort_values(by='Ocurrences', ascending=False,inplace=True)
        chart_valid_columns = valid_column_occurences.plot.barh(x='Column Name',y='Ocurrences')
        chart_valid_columns.set(xlabel='Number of Valid Logs [{:.2%} different then {}](#)'.format(
                                                                                                    PARAMETERS['LOG_FILTERING']['MINIMAL_VALID_READING_PERCENTAGE'],
                                                                                                    PARAMETERS['RTS_PARAMETERS']['INVALID_READING']))
        chart_valid_columns.set(ylabel='Column Labels')
        fig_title = 'Number of valid columns'
        plt.title(fig_title)
        mplcursors.cursor(hover=True)
        plt.savefig(PARAMETERS['PATHS']['OUTPUT_FIGURES_PATH']+fig_title)

    return(valid_df_collection)

def Filter_Valid_Logs(df_collection):
    
    valid_df_collection = Filter_Logs_Not_According_To_Expected_Columns(df_collection)
    valid_df_collection = Filter_Logs_Not_According_To_Duration(valid_df_collection)

    return(valid_df_collection)

def Filter_Desired_Columns(df_collection):
    for df_key in df_collection:
        df = df_collection[df_key]
        df_collection[df_key] = df[df.columns.intersection([TEST_TIME_COLUMN_NAME] + ['test time (m)'] + EXPECTED_COLUMNS)]

    return(df_collection)

def Filter_Valid_Rows(df_collection):
    for df_key in df_collection:
        df = df_collection[df_key]
        #Remove rows that have invalid reading, as they very likely mean that RTS lost communication to the product under test
        df_collection[df_key] = df[df[EXPECTED_COLUMNS[0]] != PARAMETERS['RTS_PARAMETERS']['INVALID_READING']]

    return df_collection

def Transform_Logs_From_Degree_F_To_C(df_collection):
    
    #Visualize_Analysis_Of_C_vs_F_Temperature_Unit(df_collection)
    
    #Transforms F to C for all columns
    if(TRANSFORM_FROM_F_TO_C):
        for df_key in df_collection:
            df = df_collection[df_key]
            df[EXPECTED_COLUMNS] = df[EXPECTED_COLUMNS].apply(lambda x: (x - 32)*(5/9))
            df_collection[df_key] = df

    return(df_collection)


def Visualize_Analysis_Of_C_vs_F_Temperature_Unit(df_collection):
    """
    As there was the open point if all the logs were in °C or °F, this function
    will support on that Analysis. The main assumption done here is that the 
    product will never publish temperature reading greater than MAXIMUM_POSSIBLE_READING_IN_DEGREE_C
    according to software requirements, so if it's greater than that for sure it's on °F unit
    If more logs are added and in the future, this analysis need to be redone
    """
    MAXIMUM_POSSIBLE_READING_IN_DEGREE_C = 71.6
    try:
        #For each log, get some statistic data from it and export data to a file
        degree_c_or_f = []
        for df_key in df_collection:
            df = df_collection[df_key]
            df_max = max(df[EXPECTED_COLUMNS].max())
            df_min = min(df[EXPECTED_COLUMNS].min())
            df_mean = df[EXPECTED_COLUMNS].mean()
            c_or_f = 'Unknow'
            if(df_max > MAXIMUM_POSSIBLE_READING_IN_DEGREE_C):
                c_or_f = '°F'

            degree_c_or_f.append( [df_key,c_or_f,df_min,df_max] + df_mean.values.tolist())
        
        #Export the data to a CSV file to facilitate anaylisis
        expected_columns_mean_name = ['MEAN '+ col_name for col_name in EXPECTED_COLUMNS ]
        df_degree_c_or_f = pd.DataFrame(degree_c_or_f, columns=['Log Name','Temperature Unit','MIN VALUE','MAX VALUE',]+expected_columns_mean_name)
        df_degree_c_or_f.to_csv(PARAMETERS['PATHS']['OUTPUT_PATH'] + 'Analysis of Degrees in C or F.csv')
        
        if(ENABLE_DEBUG_VISUALIZATIONS):
            df_degree_c_or_f['Temperature Unit'].value_counts().sort_index().plot(kind='bar', rot=0, ylabel='count')
            df_degree_c_or_f.hist(bins=30)
            mplcursors.cursor(hover=True)
            plt.show()

        #Most of the data is know to be in °F, so plot unknows or °C to try to confirm that 
        logs_not_in_degree_f = df_degree_c_or_f.loc[df_degree_c_or_f['Temperature Unit'] != '°F']['Log Name']
        for log_name in logs_not_in_degree_f:
            Visualize_SimpleTemperatureCharts(df_collection[log_name],log_name)
            plt.show()
        
        return(logs_not_in_degree_f)

    except:
        print('Some file log was not according to the expected columns')


def PreProcessInputLogs(inputLogsPathFolder):
    log_collection ={}
    
    input_logs = glob.glob(inputLogsPathFolder + "\*.csv")
    if(len(input_logs) > 0 ):
        
        #Import files from desired folder 
        for file_path in input_logs:
            file_name = file_path.replace(PARAMETERS['PATHS']['INPUT_LOG_FILES_PATH']+'\\','')
            try:
                log_collection[file_name] = pd.read_csv (file_path,header=1)
                #Columns to lower to facilitate further analysis on checking integrity of columns
                log_collection[file_name].columns = [col.lower() for col in log_collection[file_name].columns]
                #Insert a new column of time in seconds
                log_time_in_s = [Get_Seconds_FromStringHMS(reading) for reading in log_collection[file_name]['test time'] ]
                log_time_in_m = [round(time/60) for time in log_time_in_s]
                log_collection[file_name].insert(1,TEST_TIME_COLUMN_NAME,log_time_in_s,True)
                log_collection[file_name].insert(1,'test time (m)',log_time_in_m,True)

            except:
                if(file_name in log_collection):
                    del log_collection[file_name]
                print('\t\t{} not considered as it had some unexpected format'.format(file_name))
    
        #Clean the Input logs
        cleaned_log_collection = Filter_Valid_Logs(log_collection)
        cleaned_log_collection = Filter_Desired_Columns(cleaned_log_collection)
        cleaned_log_collection = Filter_Valid_Rows(cleaned_log_collection)
        cleaned_log_collection = Transform_Logs_From_Degree_F_To_C(cleaned_log_collection)

        #Visualize data before and after data cleaning
        INPUT_DATA_LABEL = 'Input Data '
        PREPROCESSED_DATA_LABEL = 'Preprocessed Data '

        Visualize_Logs_Duration_And_Sample_Rate(log_collection,INPUT_DATA_LABEL)
        Visualize_Logs_Duration_And_Sample_Rate(cleaned_log_collection,PREPROCESSED_DATA_LABEL)

        #Plot bar chart for comparison
        log_collection_size = len(log_collection.keys())
        cleaned_log_collection_size  = len(cleaned_log_collection.keys())
        fig_title = 'Data Preprocessing Evaluation {}/{} = {:.1%} of the input dataset'.format( cleaned_log_collection_size, 
                                                log_collection_size, 
                                                cleaned_log_collection_size/log_collection_size )
        print(fig_title)
        labels = [INPUT_DATA_LABEL,PREPROCESSED_DATA_LABEL]
        values = [log_collection_size,cleaned_log_collection_size]
        plt.figure()
        plt.bar(labels,values)
        plt.ylabel('Number of Logs [#]')
        plt.title(fig_title)
        for index, value in enumerate(values):
            plt.text(labels[index], value, str(value),va='bottom')
        plt.plot()
        plt.savefig(PARAMETERS['PATHS']['OUTPUT_FIGURES_PATH']+'Data Processing Evaluation')
        plt.show()


        #Export data to output folder
        for log_name in cleaned_log_collection:
            cleaned_log_collection[log_name].to_csv(PARAMETERS['PATHS']['OUTPUT_LOG_FILES_PATH'] + '/Pre Processed - ' + log_name,index=False)
    else:
        print("No .csv logs on input folder, please check \'{}\'".format(inputLogsPathFolder))

    return(cleaned_log_collection)


def Remove_Files_From_Path(path):
    logs_on_output_folder = glob.glob(path + "\*")
    if(len(logs_on_output_folder) > 0):
        for log_path in logs_on_output_folder:
            os.remove(log_path)
        print("Cleaned output folder" + path)


if __name__ == '__main__':
    EXPECTED_COLUMNS = PARAMETERS['LOG_FILTERING']['EXPECTED_COLUMNS']
    EXPECTED_COLUMNS = [col.lower() for col in EXPECTED_COLUMNS]
    
    #Check if Clean parameter received, if yes, then clean Pre Processed folder
    Remove_Files_From_Path(PARAMETERS['PATHS']['OUTPUT_FIGURES_PATH'])
    if( len(sys.argv) == 2 and sys.argv[1].lower() == 'true'):
        Remove_Files_From_Path(PARAMETERS['PATHS']['OUTPUT_LOG_FILES_PATH'])

    #Get Pre Processed logs or recreate them
    log_collection = {}
    logs_on_output_folder = glob.glob(PARAMETERS['PATHS']['OUTPUT_LOG_FILES_PATH'] + "\*.csv")
    if(len(logs_on_output_folder) == 0):
        print("No Pre Processed logs, recreating them")
        log_collection = PreProcessInputLogs(PARAMETERS['PATHS']['INPUT_LOG_FILES_PATH'])
    else:
        print("Pre Processed logs recovered")
        for file_path in logs_on_output_folder:
            file_name = file_path.replace(PARAMETERS['PATHS']['OUTPUT_LOG_FILES_PATH']+'\\','') #Remove path from file name
            log_collection[file_name] = pd.read_csv(file_path)

    # Visualize_CompletePreProcessedData(log_collection)

    #Plot some temperature charts for clarity according to what is defined on parameters yaml
    desired_logs = PARAMETERS['PLOTS']['LOGS_TO_PLOT']
    logs_to_plot = []
    if(type(desired_logs) is str):
        if(desired_logs == 'ALL'):
            desired_logs = log_collection.keys()
        else:
            desired_logs = [desired_logs]

    for log_key in log_collection:   
        for desired_log in desired_logs:
            if(desired_log in log_key):
                logs_to_plot.append(log_key)

    for log_key in logs_to_plot:
        log_title = log_key.replace(".csv",'')
        Visualize__AllLogCharts(log_collection[log_key],log_title)
        if(PARAMETERS['PLOTS']['PLOT_LOGS_INDIVIDUALLY'] == True):
            plt.show()

    plt.show()