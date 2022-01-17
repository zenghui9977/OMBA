import csv
import copy
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch

logger = logging.getLogger('logger')

global_results_header = ["epoch", "loss", "accuracy", "correct", "datasize"]
global_poison_results_header = ["epoch", "loss", "accuracy", "correct", "datasize"]
global_output_res_list_header = ["epoch", "cosine similarity", "cross entropy", "MSE"]


poison_test_results = []
test_results = []

output_res_list = []

def save_output_res_csv(folder_path):
    output_res_csvFile = open(f'{folder_path}/output_eval_result.csv', "w", newline='')
    output_res_writer = csv.writer(output_res_csvFile)

    output_res_writer.writerow(global_output_res_list_header)
    output_res_writer.writerows(output_res_list)
    output_res_csvFile.close()

def save_result_csv(folder_path):
    poison_test_csvFile = open(f'{folder_path}/poison_test_result.csv', "w", newline='')
    poison_test_writer = csv.writer(poison_test_csvFile)
    poison_test_writer.writerow(global_poison_results_header)
    poison_test_writer.writerows(poison_test_results)
    poison_test_csvFile.close()

    test_csvFile = open(f'{folder_path}/test_result.csv', "w", newline='')
    test_writer = csv.writer(test_csvFile)
    test_writer.writerow(global_results_header)
    test_writer.writerows(test_results)
    test_csvFile.close()

def save_training_models_and_updates(model_saved_folder, current_epoch, global_model, local_updates_dict):
    current_model_saved_folder = f'{model_saved_folder}/{current_epoch}'
    try:
        os.makedirs(current_model_saved_folder)
    except FileExistsError:
        logger.info(f'{current_model_saved_folder}. Folder already exist.')

    # save the global model
    torch.save(global_model, f'{current_model_saved_folder}/epoch_{current_epoch}_global_model')
    # save the local updates
    for key in local_updates_dict.keys():
        local_updates_name = f'{current_model_saved_folder}/epoch_{current_epoch}_client_{key}_local_update'
        torch.save(local_updates_dict[key], local_updates_name)
    logger.info('save the global model and local updates finished!')

def read_csv_from_file(file_folder, file_name, column_name):
    csv_list = pd.read_csv(file_folder + file_name)
    res_list = list(csv_list[column_name])
    
    return res_list

def get_the_interval_of_attacks(attack_time_list):
    temp = list(attack_time_list)
    temp.sort()
    interval_list = []
    temp_interval_list = []
    for i in temp:
        if len(temp_interval_list) == 0:
            temp_interval_list.append(i)
        elif len(temp_interval_list) == 1:
            temp_interval_list.append(i)
        elif len(temp_interval_list) == 2:
            if i == temp_interval_list[-1] + 1:
                temp_interval_list[-1] = i
            else:
                interval_list.append(temp_interval_list)
                temp_interval_list = []

    return interval_list


def pic_line(input_list, title, y_label, saved_file_name, attack_time_list, aux_line=None):


    x = range(len(input_list))
    plt.title(title)
    plt.xlabel('Commnuication Rounds')
    plt.ylabel(y_label)
    plt.plot(x, input_list, marker="o", markersize=2)
    if aux_line is not None:
        plt.axhline(y=aux_line, c='r', ls='-', lw=0.5)

    for attack_time in attack_time_list:
        plt.axvline(x=attack_time, c='r', ls='--', lw=0.5)

    plt.savefig(saved_file_name)
    plt.cla()

    


