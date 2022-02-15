from copy import copy
import os
import pandas as pd
import numpy as np

import yaml

from evaluation import transfer_cos_similarity, transfer_using_min_max
import tsod.hampel
from tsod import CombinedDetector, RangeDetector, ConstantValueDetector, ConstantGradientDetector, RollingStandardDeviationDetector, DiffDetector

import statsmodels as sm
import statsmodels.tsa.arima.model as smm

from adtk.detector import PersistAD
from adtk.visualization import plot

from visualization import plot_anomaly_time_series_line
from detection import Detector, anomaly_detection, convert_array_to_list, find_adversaries, vote_topk_index


from collections import Counter

folder_path = 'exp_records/'

# file_path = 'exp_records/model_mnist_Jan.18_16.11.39/'
# file_path = 'exp_records/model_mnist_Jan.24_16.40.02/'
# file_path = 'exp_records/model_mnist_Jan.24_16.52.45/'
# file_path = 'exp_records/model_mnist_Jan.24_16.03.59/'
# file_path = 'exp_records/model_mnist_Jan.30_13.09.11/'
# file_path = 'exp_records/model_mnist_Jan.30_13.32.40/'
# file_path = 'exp_records/model_mnist_Jan.30_14.17.41/'
# file_path = 'exp_records/model_mnist_Jan.30_15.10.24/'
# file_path = 'exp_records/model_mnist_Jan.30_15.22.51/'
# file_path = 'exp_records/model_mnist_Jan.30_15.35.06/'

file_name = 'output_eval_result.csv'
param_file_name = 'params.yaml'

file_path_list = os.listdir(folder_path)

for i in file_path_list:

    with open(f'{folder_path}/{i}/{param_file_name}', 'r') as f:
        params_loaded = yaml.load(f, Loader=yaml.FullLoader)
    
    adversary_list = params_loaded['adversary_list']
    attack_time_set = set()

    for j in range(len(adversary_list)):
        attack_time_set = attack_time_set | set(params_loaded[str(j) + '_poison_epochs'])


    res = anomaly_detection(f'{folder_path}/{i}', ground_truth=list(attack_time_set))
    if res is not None:
        print(f'Precision: {res[0]}, Recall: {res[1]}, F1: {res[2]}')
        suspected_adverisaries = find_adversaries(res[3], f'{folder_path}/{i}', method='intersection')
    else:
        print(res)



# time_series = pd.read_csv(f'{file_path}/output_eval_result.csv')

# time_series = time_series.iloc[:, 1:]

# cols_name_list = time_series.columns

# # normalize the time series
# values = time_series.values

# for i in range(len(cols_name_list)):
#     temp_list = values[:, i]

#     # cosine simialrity
#     if i == 0:
#         time_series.iloc[:, i] = transfer_cos_similarity(temp_list)
#     else:
#         time_series.iloc[:, i] = transfer_using_min_max(temp_list)



# dec = Detector(window=2, c=1.0, side='negative', detector_type='PersistAD')

# detection_range = (20, 80)
# anomaly_points, anomaly_series = dec.detect_anomaly_series(time_series=time_series, detection_range=detection_range)
# anomaly_points_slope, anomaly_series_slope = dec.using_slope_detection(time_series, 5, (20, 80))

# slope_cols_name = ['Slope' + i for i in cols_name_list]


# for i in range(len(cols_name_list)):

#     print('-'*20 + 'PersistAD' + str(cols_name_list[i]) + '-'*20)
#     precision, recall, f1 = dec.compute_precision_and_recall_f1(anomaly_points[i], list(attack_time_set))
#     print(f'Precision: {precision}\nRecall: {recall}\nF1-Score: {f1}')

#     print('-'*20 + 'slope' + str(cols_name_list[i]) + '-'*20)
#     precision, recall, f1 = dec.compute_precision_and_recall_f1(anomaly_points_slope[i], list(attack_time_set))
#     print(f'Precision: {precision}\nRecall: {recall}\nF1-Score: {f1}')

#     plot_anomaly_time_series_line(time_series = time_series.iloc[:, i],anomaly_ts=anomaly_series[i], ts_name=[cols_name_list[i]], legend=True, ground_truth=list(attack_time_set), save_file_path=f'{file_path}/detection_result/')

#     plot_anomaly_time_series_line(time_series = time_series.iloc[:, i], anomaly_ts=anomaly_series_slope[i], ts_name=[slope_cols_name[i]], legend=True, ground_truth=list(attack_time_set), save_file_path=f'{file_path}/detection_result/')

#     # logger.info('plot and save the anomaly lines......')

# vote_list = []

# for i in range(len(anomaly_points)):
#     temp = anomaly_points[i] + anomaly_points_slope[i]
#     vote_list += vote_topk_index(temp, 2)
    
# print(vote_list)

# anomaly_points = convert_array_to_list(anomaly_points)

# anomaly_points_slope = convert_array_to_list(anomaly_points_slope)



# temp = anomaly_points.copy()
# temp += anomaly_points_slope

# vote_list = vote_topk_index(temp, 4)

# print(vote_list)
# print(list(attack_time_set))
# prec, rec, f1, = dec.compute_precision_and_recall_f1(vote_list, list(attack_time_set))
# print('-'*20 + 'vote' + '-'*20)
# print(f'Precision: {prec}\nRecall: {rec}\nF1-Score: {f1}')



# print(attack_time_set)




# columns_name = csv_file.columns.values

# datetime_index = pd.date_range('20200202', periods=len(csv_file))
# temp_index = csv_file.index
# csv_file.index = datetime_index


# cosine_similarity_list = csv_file.iloc[:, 1]
# cross_entropy_list = csv_file.iloc[:, 2]
# mse_list = csv_file.iloc[:, 3]


# temp_df = csv_file.iloc[:, 1:4]
# print(temp_df)

# cosine_similarity_list = cosine_similarity_list.to_frame()
# cross_entropy_list = cross_entropy_list.to_frame()
# mse_list = mse_list.to_frame()

# temp_list = cosine_similarity_list.values
# temp_list = temp_list.flatten()
# normalized_cosine_similarity = transfer_cos_similarity(temp_list)

# temp_list = cross_entropy_list.values
# temp_list = temp_list.flatten()
# normalized_cross_entropy = transfer_using_min_max(temp_list)

# temp_list = mse_list.values
# temp_list = temp_list.flatten()
# normalized_mse_list = transfer_using_min_max(temp_list)

# normalized_cosine_similarity_series = pd.Series(normalized_cosine_similarity)
# normalized_cross_entropy_series = pd.Series(normalized_cross_entropy)
# normalized_mse_list_series = pd.Series(normalized_mse_list)

# temp_df.iloc[:, 0] = normalized_cosine_similarity
# temp_df.iloc[:, 1] = normalized_cross_entropy
# temp_df.iloc[:, 2] = normalized_mse_list

# print(temp_df.iloc[:,0])
# print(temp_df)


# mode = smm.ARIMA(cosine_similarity_list, order=(1,1,1))
# res = mode.fit()
# print(res.summary())

# normalized_cosine_similarity_series = normalized_cosine_similarity_series.to_frame()
# normalized_cosine_similarity_series.index = datetime_index

# normalized_cross_entropy_series = normalized_cross_entropy_series.to_frame()
# normalized_cross_entropy_series.index = datetime_index

# normalized_mse_list_series = normalized_mse_list_series.to_frame()
# normalized_mse_list_series.index = datetime_index

# persist_ad = PersistAD(window=2, c=1.0, side='negative')
# ano = persist_ad.fit_detect(normalized_cosine_similarity_series)
# ano2 = persist_ad.fit_detect(normalized_cross_entropy_series)
# ano3 = persist_ad.fit_detect(normalized_mse_list_series)


# axxx = persist_ad.fit_detect(temp_df)

# print(ano, ano2, ano3)

# cosine_similarity_list.index = temp_index
# ano.index = temp_index

# plot_anomaly_time_series_line(csv_file, ano, ts_name=['index','Cosine Similarity', 'Cross Entropy', 'MSE loss'])
# plot_anomaly_time_series_line(normalized_cosine_similarity_series, ano, ts_name=['Cosine Similarity'], ground_truth=list(attack_time_set), save_file_path=f'{file_path}/detection_result/')
# plot_anomaly_time_series_line(normalized_cross_entropy_series, ano2, ts_name=['Cross Entropy'], ground_truth=list(attack_time_set), save_file_path=f'{file_path}/detection_result/')
# plot_anomaly_time_series_line(normalized_mse_list_series, ano3, ts_name=['MSE loss'], ground_truth=list(attack_time_set), save_file_path=f'{file_path}/detection_result/')


# plot(normalized_cosine_similarity_series, anomaly=ano, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker")
# plot(normalized_cross_entropy_series, anomaly=ano2, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker")
# plot(normalized_mse_list_series, anomaly=ano3, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker")

# import matplotlib.pyplot as plt
# plt.show()

# win_size = range(1,10,1)
# threshold_size = np.arange(start=0.1, stop=3.0, step=0.1)

# print(np.where(ano)[0])
# print(np.where(ano2)[0])
# print(np.where(ano3)[0])



# print(np.where(axxx.values[0])[0])
# print(np.where(axxx.values[1])[0])
# print(np.where(axxx.values[2])[0])
# for i in win_size:
#     for j in threshold_size:
#         print('-'*20 + 'win_size_' + str(i) + 'threshold_' + str(j) + '-'*20)
#         # cd = RollingStandardDeviationDetector(window_size=i)
#         cd = tsod.hampel.HampelDetector(window_size=i, threshold=j)


#         xx = cd.detect(normalized_cosine_similarity_series)
#         xxx = cd.detect(normalized_cross_similarity_series)
#         xxxx = cd.detect(normalized_mse_list_series)

#         # print(xx.values)
#         # print(xxx.values)
#         # print(xxxx.values)



#         print(np.where(xx)[0])
#         print(np.where(xxx)[0])
#         print(np.where(xxxx)[0])
#         print('-'*20)
        


# cd = CombinedDetector([RollingStandardDeviationDetector(window_size=3, center=True), DiffDetector()])
# cd = tsod.hampel.HampelDetector(window_size=3)
# cd = RollingStandardDeviationDetector(window_size=3, center=True)
# cd = CombinedDetector([tsod.hampel.HampelDetector(window_size=3), RangeDetector(min_value=0)])


# xx = cd.detect(normalized_cosine_similarity_series)
# xxx = cd.detect(normalized_cross_similarity_series)
# xxxx = cd.detect(normalized_mse_list_series)

# print(np.where(xx)[0])
# print(np.where(xxx)[0])
# print(np.where(xxxx)[0])

# print(xx.values)
# print(xxx.values)
# print(xxxx.values)
