

import os
from tkinter.messagebox import NO
import matplotlib.pyplot as plt
import pandas as pd


def check_and_reset_datatype(time_series, ts_name=None):
    if time_series is not None:
        if isinstance(time_series, list):
            time_series = pd.Series(time_series)

        if isinstance(time_series, pd.Series):
                df = time_series.to_frame()
                df.columns = ts_name
        
        elif isinstance(time_series, pd.DataFrame):
            df = time_series.copy()
            # set the columns' name
            df.columns = ts_name

        else:
            raise TypeError("Argument 'time_series' must be list, Series or DataFrame.")

        # set the index
        ts_len = len(df)
        df.index = range(ts_len)
        
    else:
        raise ValueError("Argument 'time_series' is None")  
    return df


def plot_curve(df, axes):
    columns_name_list = df.columns
    lines_num = len(columns_name_list)
    if lines_num == 1:
        axes.plot(df.index, df[columns_name_list[0]], label=columns_name_list[0])
    else:
        for i in range(lines_num):
            axes[i].plot(df.index, df[columns_name_list[i]], label=columns_name_list[0])


def plot_anomaly_curve(anomaly, df, axes, anomaly_columns):
    '''
    Paramters:
    anomaly: data type -> Dataframe, cols -> [index, anomaly_cols_name]
    df: original time series data, data type -> Dataframe, cols -> [index, cols_name]
    axes: plot variance
    
    '''
    anomaly = anomaly.values
    anomaly_curve = df.loc[anomaly == 1]

    anomaly_curve.columns = anomaly_columns

    lines_num = len(anomaly_columns)

    if lines_num == 1:
        axes.plot(anomaly_curve.index, anomaly_curve[anomaly_columns[0]], 'ro', label=anomaly_columns[0])
    else:
        for i in range(lines_num):
            axes[i].plot(anomaly_curve.index, anomaly_curve[anomaly_columns[i]], 'ro', label=anomaly_columns[i])




def plot_anomaly_time_series_line(time_series, anomaly_ts, ts_name=None, legend=True, ground_truth=None, save_file_path=None):
    '''Plot time series and /or anomalies

    Parameters
    -----------

    time_series: list, pandas Series or DataFrame, optional; Time Series to plot
    anomaly_ts: list, pandas Series or DataFrame, optional; Anomaly time series to plot
    ts_name: list, optional; The name list of the anomaly series
    legend: bool, optioinal; Whether display the legend, default is true
    ground_truth: list, optional; The ground truth list of the anomaly

    
    '''
    # setup axes
    _, axes =plt.subplots(nrows=len(ts_name), sharex=True)


    # plot time series first
    df = check_and_reset_datatype(time_series, ts_name)
    plot_curve(df, axes)
    
    # plot anomaly point
    anomaly_cols_name = [f'Anomaly-{i}' for i in df.columns]
    anomaly_ts = check_and_reset_datatype(anomaly_ts, ts_name=anomaly_cols_name)

    plot_anomaly_curve(anomaly_ts, df, axes, anomaly_cols_name)

    # display legend
    if legend and (time_series is not None or anomaly_ts is not None):
        axes.legend()
    
    # display the ground truth
    if ground_truth is not None:
        for gt in ground_truth:
            axes.axvline(x=gt, c='r', ls='--', lw=0.5)

    if save_file_path is None:
        plt.savefig('./Pic.png')
    else:
        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path)
        plt.savefig(save_file_path + ts_name[0] + '.png')

    return axes









