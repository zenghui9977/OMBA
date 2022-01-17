
import os
import time
import logging
import numpy as np
import argparse



import torch
import yaml
import datetime
import visdom
import copy
from args_setting import Args_parse
import constant_str
import save_records
from helper import Helper
from local_train import Image_Trainer
from evaluation import compute_2_tensor_mse, compute_JS_loss, compute_KL_loss, compute_cosine_similarity, compute_cross_entropy, compute_slope_in_each_round, compute_the_output_of_each_sample, sort_cut_list, test_on_data, transfer_cos_similarity, transfer_using_min_max

logger = logging.getLogger('logger')


vis = visdom.Visdom(port=8098)

logger.info('start training.......')

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)


time_start = time.time()

# parser = argparse.ArgumentParser(description='DK')
# parser.add_argument('--params', dest='params')
# parser.add_argument('--exp_code', dest='exp_code')
args = Args_parse()

with open(f'./{args.params}', 'r') as f:
    params_loaded = yaml.load(f, Loader=yaml.FullLoader)

current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')

helper = Helper(params=params_loaded, name=params_loaded['type'], current_time=current_time, args=args)

helper.load_data()
logger.info('data loading finished......')
helper.create_model()
global_model = copy.deepcopy(helper.target_model)
logger.info('model creation finished......')
weight_accumulator = helper.init_weight_accumulator()
logger.info('init weight accumulator finished......')

if helper.params['is_train']:
    # training process
    acc_list = []
    poison_acc_list = []
    for epoch in range(0, helper.params['epochs']):
        start_time = time.time()
        participants_list_current, adversarial_list_current = helper.choose_clients_every_epoch(epoch)
        logger.info(f'Server Epoch:{epoch} choose agents : {participants_list_current}. Adversaries: {adversarial_list_current}')

        submit_update_dict, num_samples_dict = Image_Trainer(helper, current_epoch=epoch, global_model=global_model, participants=participants_list_current, adversarial_participants=adversarial_list_current)

        weight_accumulator = helper.accumulate_weight(weight_accumulator, submit_update_dict, participants_list_current)

        global_model = helper.average_shrink_model(weight_accumulator, global_model)

        weight_accumulator = helper.init_weight_accumulator()


        if helper.params['pretrain']:
            if epoch >= helper.params['pretrain_terminal_round']:
                torch.save(global_model.state_dict(), f"./saved_models/pretrain_{helper.params['type']}")
                logger.info('pretrainig finished......')
                break
        else:

            # Test and evaluation
            if helper.params['is_poison']:
                epoch_p_loss, epoch_p_acc, epoch_p_correct, epoch_p_datasize =  test_on_data(helper, epoch, global_model, is_test_on_poison=True)
                save_records.poison_test_results.append([epoch, epoch_p_loss, epoch_p_acc, epoch_p_correct, epoch_p_datasize])
                # show in visdom
                vis.line(
                    X=np.array([epoch]), Y=np.array([epoch_p_acc]), win=f'global_poison_acc_{current_time}', update='append' if vis.win_exists(f'global_poison_acc_{current_time}') else None,
                    opts=dict(showlegend=False, title='global attack success rate',width=400, height=400, xlabel='Communication rounds', ylabel='Attack Success Rate')
                )

                poison_acc_list.append(epoch_p_acc)

            epoch_loss, epoch_acc, epoch_correct, epoch_datasize = test_on_data(helper, epoch, global_model, is_test_on_poison=False)
            save_records.test_results.append([epoch, epoch_loss, epoch_acc, epoch_correct, epoch_datasize])

            vis.line(
                X=np.array([epoch]), Y=np.array([epoch_acc]), win=f'global_acc_{current_time}', update='append' if vis.win_exists(f'global_poison_acc_{current_time}') else None,
                opts=dict(showlegend=False, title='global accuracy',width=400, height=400, xlabel='Communication rounds', ylabel='Accuracy')
            )

            acc_list.append(epoch_acc)

            save_records.save_result_csv(helper.folder_path)

            # save the global model and submitted local updates       
            save_records.save_training_models_and_updates(helper.model_saved_folder, epoch, global_model.state_dict(), submit_update_dict)
            
    # save the list into pictures
    if helper.params['is_poison']:
        save_records.pic_line(poison_acc_list, title='global attack success rate', y_label='Attack Success Rate', saved_file_name=f'{helper.pic_saved_folder}/global_attack_success_rate.png', attack_time_list=list(helper.attack_time_set))
    save_records.pic_line(acc_list, title='global accuracy', y_label='Accuracy', saved_file_name=f'{helper.pic_saved_folder}/global_accuracy.png', attack_time_list=list(helper.attack_time_set))



# attacker sourcing process
if helper.params['attacker_trace']:
    logger.info('Start to trace the attackers during the training process')

    logger.info(f'Fetch the model outputs of the failure samples......')

    model_output = compute_the_output_of_each_sample(helper, is_test_on_poison=True)
    logger.info(f'Compute the cosine similarity')
    cs_list = compute_cosine_similarity(model_output)
    cs_slope_list = compute_slope_in_each_round(cs_list)

    # cos similarity transfer
    transfer_cs_list = transfer_cos_similarity(cs_list)
    transfer_cs_slope_list = compute_slope_in_each_round(transfer_cs_list)

    logger.info(f'Compute the MSE loss')
    mse_loss_list = compute_2_tensor_mse(model_output)
    mse_loss_slope_list = compute_slope_in_each_round(mse_loss_list)

    # MSE transfer
    transfer_mse_loss = transfer_using_min_max(mse_loss_list)
    transfer_mse_loss_slope_list = compute_slope_in_each_round(transfer_mse_loss)    


    logger.info(f'Compute the Cross Entropy')
    cross_entropy_list = compute_cross_entropy(model_output)
    cross_entropy_slope_list = compute_slope_in_each_round(cross_entropy_list)

    # Cross Entropy transferred
    transfer_cross_entropy = transfer_using_min_max(cross_entropy_list)
    transfer_cross_entropy_slope_list = compute_slope_in_each_round(transfer_cross_entropy)


    # save the records of the output
    for i in range(len(cs_list)):
        save_records.output_res_list.append([i, cs_list[i], cross_entropy_list[i], mse_loss_list[i]])
    save_records.save_output_res_csv(helper.folder_path)





    # draw the pic
    vis.line(
        X = np.array(range(len(cs_list))), Y=np.array(cs_list), win=f'cosine similarity_{current_time}',
        opts=dict(showlegend=False, title='Cosine Similarity',width=400, height=400, xlabel='Communication rounds', ylabel='Cosine Similarity')
    )
    vis.line(
        X = np.array(range(len(cs_slope_list))), Y=np.array(cs_slope_list), win=f'cosine similarity_slope_{current_time}',
        opts=dict(showlegend=False, title='Cosine Similarity Slope',width=400, height=400, xlabel='Communication rounds', ylabel='Cosine Similarity Slope')
    )
    vis.line(
        X = np.array(range(len(transfer_cs_list))), Y=np.array(transfer_cs_list), win=f'transfer cosine similarity_{current_time}',
        opts=dict(showlegend=False, title='1 - Cosine Similarity',width=400, height=400, xlabel='Communication rounds', ylabel='Cosine Similarity(normalized)')
    )
    vis.line(
        X = np.array(range(len(transfer_cs_slope_list))), Y=np.array(transfer_cs_slope_list), win=f'transfer cosine similarity_slope_{current_time}',
        opts=dict(showlegend=False, title='(1 - Cosine Similarity) Slope',width=400, height=400, xlabel='Communication rounds', ylabel='(1 - Cosine Similarity) Slope')
    )    

    save_records.pic_line(cs_list, title='Cosine Similarity', y_label='Cosine Similarity', saved_file_name=f'{helper.pic_saved_folder}/cosine_similarity.png', attack_time_list=list(helper.attack_time_set))
    save_records.pic_line(cs_slope_list, title='Cosine Similarity Slope', y_label='Cosine Similarity Slope', saved_file_name=f'{helper.pic_saved_folder}/cosine_similarity_slope.png', attack_time_list=list(helper.attack_time_set), aux_line=0)
    save_records.pic_line(transfer_cs_list, title='1 - Cosine Similarity', y_label='1 - Cosine Similarity', saved_file_name=f'{helper.pic_saved_folder}/normalized_cosine_similarity.png', attack_time_list=list(helper.attack_time_set))
    save_records.pic_line(transfer_cs_slope_list, title='(1 - Cosine Similarity) Slope', y_label='(1 - Cosine Similarity) Slope', saved_file_name=f'{helper.pic_saved_folder}/normalized_cosine_similarity_slope.png', attack_time_list=list(helper.attack_time_set), aux_line=0)

    vis.line(
        X = np.array(range(len(mse_loss_list))), Y=np.array(mse_loss_list), win=f'mse_{current_time}',
        opts=dict(showlegend=False, title='MSE loss',width=400, height=400, xlabel='Communication rounds', ylabel='MSE loss')
    )
    vis.line(
        X = np.array(range(len(mse_loss_slope_list))), Y=np.array(mse_loss_slope_list), win=f'mse_slope_{current_time}',
        opts=dict(showlegend=False, title='MSE loss Slope',width=400, height=400, xlabel='Communication rounds', ylabel='MSE loss Slope')
    ) 
    vis.line(
        X = np.array(range(len(transfer_mse_loss))), Y=np.array(transfer_mse_loss), win=f'transfer mse_loss_{current_time}',
        opts=dict(showlegend=False, title='MSE loss(normalized)',width=400, height=400, xlabel='Communication rounds', ylabel='MSE loss(normalized)')
    ) 
    vis.line(
        X = np.array(range(len(transfer_mse_loss_slope_list))), Y=np.array(transfer_mse_loss_slope_list), win=f'transfer mse_slope_{current_time}',
        opts=dict(showlegend=False, title='MSE loss Slope(normalized)',width=400, height=400, xlabel='Communication rounds', ylabel='MSE loss Slope(normalized)')
    )

    save_records.pic_line(mse_loss_list, title='MSE loss', y_label='MSE loss', saved_file_name=f'{helper.pic_saved_folder}/mse.png', attack_time_list=list(helper.attack_time_set))
    save_records.pic_line(mse_loss_slope_list, title='MSE loss Slope', y_label='MSE loss Slope', saved_file_name=f'{helper.pic_saved_folder}/mse_slope.png', attack_time_list=list(helper.attack_time_set), aux_line=0)
    save_records.pic_line(transfer_mse_loss, title='MSE loss(normalized)', y_label='MSE loss(normalized)', saved_file_name=f'{helper.pic_saved_folder}/normalized_mse.png', attack_time_list=list(helper.attack_time_set))
    save_records.pic_line(transfer_mse_loss_slope_list, title='MSE loss Slope(normalized)', y_label='MSE loss Slope(normalized)', saved_file_name=f'{helper.pic_saved_folder}/normalized_mse_slope.png', attack_time_list=list(helper.attack_time_set), aux_line=0)


    vis.line(
        X = np.array(range(len(cross_entropy_list))), Y=np.array(cross_entropy_list), win=f'cross entropy_{current_time}',
        opts=dict(showlegend=False, title='Cross Entropy',width=400, height=400, xlabel='Communication rounds', ylabel='Cross Entropy')
    )
    vis.line(
        X = np.array(range(len(cross_entropy_slope_list))), Y=np.array(cross_entropy_slope_list), win=f'cross entropy_slope_{current_time}',
        opts=dict(showlegend=False, title='Cross Entropy Slope',width=400, height=400, xlabel='Communication rounds', ylabel='Cross Entropy Slope')
    )      
    vis.line(
        X = np.array(range(len(transfer_cross_entropy))), Y=np.array(transfer_cross_entropy), win=f'transfer cross entropy_{current_time}',
        opts=dict(showlegend=False, title='Cross Entropy(normalized)',width=400, height=400, xlabel='Communication rounds', ylabel='Cross Entropy(normalized)')
    )    
    vis.line(
        X = np.array(range(len(transfer_cross_entropy_slope_list))), Y=np.array(transfer_cross_entropy_slope_list), win=f'transfer cross entropy_slope_{current_time}',
        opts=dict(showlegend=False, title='Cross Entropy Slope(normalized)',width=400, height=400, xlabel='Communication rounds', ylabel='Cross Entropy Slope(normalized)')
    ) 

    save_records.pic_line(cross_entropy_list, title='Cross Entropy', y_label='Cross Entropy', saved_file_name=f'{helper.pic_saved_folder}/cross_entropy.png', attack_time_list=list(helper.attack_time_set))
    save_records.pic_line(cross_entropy_slope_list, title='Cross Entropy Slope', y_label='Cross Entropy Slope', saved_file_name=f'{helper.pic_saved_folder}/cross_entropy_slope.png', attack_time_list=list(helper.attack_time_set), aux_line=0)
    save_records.pic_line(transfer_cross_entropy, title='Cross Entropy(normalized)', y_label='Cross Entropy(normalized)', saved_file_name=f'{helper.pic_saved_folder}/normalized_cross_entropy.png', attack_time_list=list(helper.attack_time_set))
    save_records.pic_line(transfer_cross_entropy_slope_list, title='Cross Entropy Slope(normalized)', y_label='Cross Entropy Slope(normalized)', saved_file_name=f'{helper.pic_saved_folder}/normalized_cross_entropy_slope.png', attack_time_list=list(helper.attack_time_set), aux_line=0)

  

    composite_list = np.array(transfer_cs_slope_list)+np.array(transfer_mse_loss_slope_list)+np.array(transfer_cross_entropy_slope_list)
    
    vis.line(
        X = np.array(range(len(composite_list))), Y=np.array(composite_list), win=f'composite_list_{current_time}',
        opts=dict(showlegend=False, title='Composite value',width=400, height=400, xlabel='Communication rounds', ylabel='Composite Slope')
    ) 

    attack_set = list(helper.get_attack_set())
    attack_set.sort(reverse=False)
    print(attack_set)

    print(sort_cut_list(transfer_cs_slope_list))
    print(sort_cut_list(transfer_mse_loss_slope_list))
    print(sort_cut_list(transfer_cross_entropy_slope_list))

    print(sort_cut_list(composite_list))
    # x = np.argsort(transfer_cs_slope_list)
    # print(transfer_cs_slope_list)
    # transfer_cs_slope_list.sort(reverse=False)
    # print(transfer_cs_slope_list)
    # print(x)

    # x2 = np.argsort(transfer_mse_loss_slope_list)
    # print(transfer_mse_loss_slope_list)
    # transfer_mse_loss_slope_list.sort(reverse=False)
    # print(transfer_mse_loss_slope_list)
    # print(x2)


    # x3 = np.argsort(transfer_cross_entropy_slope_list)
    # print(transfer_cross_entropy_slope_list)
    # transfer_cross_entropy_slope_list.sort(reverse=False)
    # print(transfer_cross_entropy_slope_list)
    # print(x3)

logger.info('saving the lines......')

