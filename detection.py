from collections import Counter
import os
from adtk.detector import PersistAD
import pandas as pd
import numpy as np
import torch
import constant_str
from evaluation import transfer_cos_similarity, transfer_using_min_max
import re


class Detector:

    def __init__(self, window, c, side, detector_type='PersistAD'):
        self.window = window
        self.c = c
        self.side = side

        self.detector_type = detector_type
        self.detector = None

        self.init_detector()
        
        pass

    def init_detector(self):
        if self.detector_type == 'PersistAD':
            self.detector = PersistAD(window=self.window, c=self.c, side=self.side)
        else:
            raise TypeError("Please set the argument 'detector_type' from {PersistAD ......}")

    def using_slope_detection(self, time_series, top_k_slope, detection_range):
        # time series analysis
        len_columns = len(time_series.columns)

        (start_, end_) = detection_range
        detection_range = range(start_, end_)

        detection_results = []
        detection_series = []

        for k in range(len_columns):
            ones_ts = time_series.iloc[:, k]
            # get the slope list
            slope_list = []
            for i in range(len(ones_ts)):
                if (i - 1) < 0:
                    slope_list.append(0)
                else:
                    slope_list.append(ones_ts[i] - ones_ts[i-1])
            
            # using the slope
            sorted_idx = np.argsort(slope_list)
        
            detection_result = []

            for i in sorted_idx:
                if i in detection_range:
                    detection_result.append(i)
        
            detection_result = detection_result[:top_k_slope]

            detection_serie = []
            for i in range(len(ones_ts)):
                if i in detection_result:
                    detection_serie.append(True)
                else:
                    detection_serie.append(False)

            detection_series.append(pd.DataFrame(detection_serie))
            detection_results.append(detection_result)
        
        return detection_results, detection_series

    def detect_anomaly_series(self, time_series, detection_range):

        # check the time series format
        if not isinstance(time_series.index, pd.DatetimeIndex):
            datetime_index = pd.date_range('20200202', periods=len(time_series))

            time_series.index = datetime_index

        # input the time series into the detector

        # the number of the columns
        len_columns = len(time_series.columns)

        detection_anomaly_points = []
        detection_anomaly_series = []

        (start_, end_) = detection_range

        for i in range(len_columns):
            one_ts = time_series.iloc[:, i]
            one_detection_results = self.detector.fit_detect(one_ts)
            
            detection_anomaly_series.append(one_detection_results)

            temp = np.where(one_detection_results)[0]
            temp = range_detection_filter(temp, start_= start_, end_= end_)

            detection_anomaly_points.append(list(temp))

        return detection_anomaly_points, detection_anomaly_series

    
def compute_precision_and_recall_f1(detection_anomaly_points, ground_truth):
    hits = 0
    # the length of detection results and ground truth
    len_dr, len_gt = len(detection_anomaly_points), len(ground_truth)
    for temp in detection_anomaly_points:
        if temp in ground_truth:
            hits += 1


    # precision
    if len_dr != 0:
        precision = hits / (1.0 * len_dr)
    else:
        precision = 0
    
    if len_gt != 0:
        recall = hits / (1.0 * len_gt)
    else:
        recall = 0

    f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1

def convert_array_to_list(input_list):
    temp_list = []
    for i in input_list:
        if isinstance(i, int):
            temp_list.append(i)

        else:
            if isinstance(i, np.ndarray):
                i = i.tolist()
            
            if len(temp_list) == 0:
                temp_list = i
            else:
                temp_list += i
        
    return temp_list

def range_detection_filter(input_list, start_, end_):
    
    detection_range = range(start_, end_)

    filter_result = []
    for i in input_list:
        if i in detection_range:
            filter_result.append(i)
    
    return filter_result


def vote_topk_index(input_list, top_k):
    # print(input_list)

    temp_list = np.array(input_list).reshape(1, -1)[0]
    # print(temp_list)

    count_list = Counter(temp_list)
    # print(count_list)

    dic = sorted(count_list.items(), key = lambda x: x[1], reverse=True)[:top_k]

    key_index = [x[0] for x in dic]

    return key_index

def vote_threshold_index(input_list, threshold):

    input_list = convert_array_to_list(input_list)

    temp_list = np.array(convert_array_to_list(input_list)).reshape(1, -1)[0]
    # print(temp_list)

    count_list = Counter(temp_list)
    print(count_list)

    dic = sorted(count_list.items(), key = lambda x: x[1], reverse=True)

    key_index = [x[0] for x in dic if x[1] >= threshold]

    # print(key_index)  
    return key_index


def anomaly_detection(detection_file_path, ground_truth):

    if os.path.exists(f'{detection_file_path}/{constant_str.DETECTION_METRICS_FILE}'):
        input_time_series = pd.read_csv(f'{detection_file_path}/{constant_str.DETECTION_METRICS_FILE}')

        input_time_series = input_time_series.iloc[:, 1:]

        cols_name_list = input_time_series.columns
        values = input_time_series.values

        for i in range(len(cols_name_list)):
            temp_list = values[:, i]

            if i == 0:
                input_time_series.iloc[:, i] = transfer_cos_similarity(temp_list)
            else:
                input_time_series.iloc[:, i] = transfer_using_min_max(temp_list)

        
        dec = Detector(window=2, c=1.0, side='negative', detector_type='PersistAD')

        detection_range = (20, 80)

        detector_anomaly_points, detector_anomaly_series = dec.detect_anomaly_series(input_time_series, detection_range)
        slope_anomaly_points, slope_anomaly_series = dec.using_slope_detection(input_time_series, 5, detection_range)

        vote_list = vote_threshold_index([detector_anomaly_points, slope_anomaly_points], threshold=(len(detector_anomaly_points) + len(slope_anomaly_points))/2)

        precision, recall, f1 = compute_precision_and_recall_f1(vote_list, ground_truth=ground_truth)

        return precision, recall, f1, vote_list
    
    else:
        return None

def read_client_id_from_exp_record(epoch_folder_file_path):

    epoch_folder_file_list = os.listdir(epoch_folder_file_path)

    if constant_str.PARTICIPANTS_FILE_NAME in epoch_folder_file_list:
        client_id_list = torch.load(f'{epoch_folder_file_path}/{constant_str.PARTICIPANTS_FILE_NAME}')
    else:
        client_id_list = []
        for i in epoch_folder_file_list:
            temp = re.findall(r"\d+", i)
            # use the second number: the client index
            if len(temp) > 1:
                client_id_list.append(int(temp[1]))
        # print(client_id_list)
    return client_id_list


def find_adversaries(anomaly_epochs, detection_file_path, method):

    if not os.path.exists(detection_file_path):
        raise FileExistsError('%s is not exist' % detection_file_path)

    client_id_list = []
 
    for epoch in anomaly_epochs:
        if not os.path.exists(f'{detection_file_path}/{constant_str.DETECTION_MODEL_FOLDER}/{epoch}/'):
            raise FileExistsError(f'the epoch %s is not existed in the experiment records' % epoch)
        
        epoch_client_id_list = read_client_id_from_exp_record(f'{detection_file_path}/{constant_str.DETECTION_MODEL_FOLDER}/{epoch}/')

        client_id_list.append(epoch_client_id_list)

    if method == 'intersection':

        adversaries_list = set(client_id_list[0]).intersection(*client_id_list[1:])
    elif method == 'count':
        count_alpha = 0.5
        adversaries_list = vote_threshold_index(client_id_list, len(client_id_list) * count_alpha)

    print(adversaries_list)
    return adversaries_list










