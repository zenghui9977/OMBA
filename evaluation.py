import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import constant_str
from sklearn import preprocessing

logger = logging.getLogger('logger')

def test_on_data(helper, epoch, model, is_test_on_poison):
    model.eval()
    total_loss = 0
    correct = 0
    dataset_size = 0
    

    if helper.params['type'] == constant_str.MNIST_TYPE or helper.params['type'] == constant_str.CIFAR_TYPE:
        
        if is_test_on_poison:
            data_iterator = helper.poisoned_test_dataloader
        else:
            data_iterator = helper.test_dataloader

        for batch_id, batch in enumerate(data_iterator):
            if is_test_on_poison:
                data, targets, poison_num = helper.get_poison_batch(batch, evaluation=True)
                dataset_size += poison_num
   
            else:
                data, targets = helper.get_batch(batch)
                 
                dataset_size += len(data)

            output = model(data)

            total_loss += nn.functional.cross_entropy(output, targets, reduction='sum').item()
            pred = output.data.max(1)[1]
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size)) if dataset_size != 0 else 0
        total_l = total_loss / dataset_size if dataset_size!=0 else 0

        logger.info('Test {} poisoned {} epoch {} Average loss {:.4f}\n'
            'Accuracy {}/{} ({:.4f}%)'
            .format(helper.name, is_test_on_poison, epoch, 
            total_l, correct, dataset_size, acc
            )
        )

    model.train()
        
    return total_l, acc, correct, dataset_size



def compute_the_output_of_each_sample(helper, is_test_on_poison):


    helper.filter_success_fail_samples()

    test_folder = helper.params['test_folder']
    
    if test_folder == 'None': 
        analystic_results_folder = helper.model_saved_folder 
    else:
        analystic_results_folder = f'{test_folder}/models_and_localupdates/'

    logger.info('-'*40)
    logger.info(f'Analyzing the global models in each round.')
    logger.info(f'load the model paramters from{analystic_results_folder}')
    logger.info('-'*40)

    model_output = []

    for e in range(helper.params['epochs']):

        the_eval_model_path = f'{analystic_results_folder}/{e}/epoch_{e}_global_model'
        eval_model = copy.deepcopy(helper.local_model)
        eval_model_dict = torch.load(the_eval_model_path)
        eval_model.load_state_dict(eval_model_dict)
        eval_model.eval()

        data_iterator = helper.backdoor_success_samples_dataloader

        for _, batch in enumerate(data_iterator):
            if is_test_on_poison:
                data, targets, poison_num = helper.get_poison_batch(batch, evaluation=True)
            else:
                data, targets = helper.get_batch(batch)
            
            output = eval_model(data)

            model_output.append(output)
    
    return model_output
            
def compute_cosine_similarity(model_output):
    # model output, type: list, lev_index: epohc, lev2_index: output tensor
    the_last_epoch_output = model_output[-1]
    cosine_similarity_list = []

    for e in range(len(model_output) - 1):
        temp_output = model_output[e]
        cos_res = torch.cosine_similarity(the_last_epoch_output, temp_output, dim=1)
        cos_res_mean = torch.mean(cos_res)
        cosine_similarity_list.append(cos_res_mean.item())

    return cosine_similarity_list

def cross_entropy_between_2_tensor(input_tensor, target_tensor, m):

    input_tensor = F.softmax(input_tensor, dim=1)
    target_tensor = F.softmax(target_tensor, dim=1)
    # print(target_tensor)

    log_input_tensor = torch.log(input_tensor)

    res = - torch.sum(log_input_tensor * target_tensor) / m
    # print(res)
    return res

def compute_cross_entropy(model_output):

    the_last_epoch_output = model_output[-1]
    cross_entropy_list = []
    m = the_last_epoch_output.shape[0]
    
    for e in range(len(model_output) - 1):

        temp_output = model_output[e]
        cros_res = cross_entropy_between_2_tensor(the_last_epoch_output, temp_output, m)
        cross_entropy_list.append(cros_res.item())
    
    return cross_entropy_list

def compute_multilabel_soft_margin_loss(model_output):
    the_last_epoch_output = model_output[-1]
    multilabel_soft_margin_loss_list = []
    m = the_last_epoch_output.shape[0]
    
    for e in range(len(model_output) - 1):

        temp_output = model_output[e]
        mulloss_res = nn.MultiLabelSoftMarginLoss(temp_output, the_last_epoch_output, m)
        multilabel_soft_margin_loss_list.append(mulloss_res.item())
    
    return multilabel_soft_margin_loss_list    

def compute_2_tensor_mse(model_output):
    the_last_epoch_output = model_output[-1]

    mse_loss_list = []

    for e in range(len(model_output) - 1):

        temp_output = model_output[e]
        mes_res = F.mse_loss(temp_output, the_last_epoch_output)
        mse_loss_list.append(mes_res.item())
    
    return mse_loss_list

def JS_div(p_output, q_output, get_softmax=True):
    '''
    Function that measures JS divergence between target and output logits:
    '''
    KLDiv_loss = nn.KLDivLoss(reduction='batchmean')

    if get_softmax:
        p_output = F.softmax(p_output, dim=-1)
        q_output = F.softmax(q_output, dim=-1)

    log_mean_output = ((p_output + q_output)/2).log()

    return (KLDiv_loss(log_mean_output, p_output) + KLDiv_loss(log_mean_output, p_output))/2


def compute_KL_loss(model_output):
    the_last_epoch_output = model_output[-1]
    logits_the_last_epoch_output = torch.log_softmax(the_last_epoch_output, dim=-1)

    kl_loss_list = []

    for e in range(len(model_output) - 1):

        temp_output = model_output[e]
        logits_temp_output = torch.softmax(temp_output, dim=-1)
        kl_loss_res = F.kl_div(logits_the_last_epoch_output, logits_temp_output, reduction='batchmean')
        kl_loss_list.append(kl_loss_res.item())
    
    return kl_loss_list


def compute_JS_loss(model_output):
    the_last_epoch_output = model_output[-1]

    JS_loss_list = []
    for e in range(len(model_output) - 1):
        temp_output = model_output[e]
        js_res = JS_div(the_last_epoch_output, temp_output, get_softmax=True)
        JS_loss_list.append(js_res.item())
    
    return JS_loss_list

def compute_slope_in_each_round(input_list):
    slope_list = []
    for i in range(len(input_list)):
        if i - 1 < 0:
            slope_list.append(0)
        else:
            slope_list.append(input_list[i] - input_list[i - 1])
    
    return slope_list

def transfer_cos_similarity(input_list):
    transfer_list = [1-temp if 1-temp<=1 else 1 for temp in input_list]
    return transfer_list

def transfer_using_min_max(input_list):
    min_max_scalar = preprocessing.MinMaxScaler()
    data_trans_minmax = min_max_scalar.fit_transform(np.array(input_list).reshape(-1, 1))
    data_trans_minmax = list(np.array(data_trans_minmax).flatten())
    return data_trans_minmax

def sort_cut_list(input_list):
    sort_idx = np.argsort(input_list)
    slice_idx = 0
    for i in sort_idx:
        if input_list[i] >= 0:
            slice_idx += 1
    
    sort_idx = sort_idx[:slice_idx]
    return sort_idx
    
    




