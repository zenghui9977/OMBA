import copy
import torch
import logging
import torch.nn as nn

from torch import random
logger = logging.getLogger('logger')

def Image_Trainer(helper, current_epoch, global_model, participants, adversarial_participants):
    
    submit_update_dict = dict()
    num_samples_dict = dict()

    # update the model weight
    for participant in participants:
        # the benign participant trains the global model based on the local data
        # downloaded the global model

        local_model = copy.deepcopy(global_model)
        
        local_model.train()
        
        # adversaries
        if helper.params['is_poison'] and adversarial_participants is not None and participant in adversarial_participants:
            
            logger.info('The adversary performs local backdoor training now')


            local_epochs = helper.params['poison_local_epochs']
            poison_lr = helper.params['poison_lr']
            poison_optimizer = torch.optim.SGD(local_model.parameters(), lr=poison_lr, momentum=helper.params['momentum'], weight_decay=helper.params['decay'])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer, milestones=[0.2 * local_epochs, 0.8 * local_epochs], gamma=0.1)

            _, data_iterator = helper.train_dataloader[participant]
            for local_epoch in range(local_epochs):
                poison_num_counter = 0
                total_loss = 0.
                correct = 0
                dataset_size = 0

                last_local_model = copy.deepcopy(local_model)
                last_local_model_param_var = dict()
                for name, _ in last_local_model.named_parameters():
                    last_local_model_param_var[name] = last_local_model.state_dict()[name].clone().detach().requires_grad_(False)

                for batch_id, batch in enumerate(data_iterator):
                    data, targets, poison_num = helper.get_poison_batch(batch, evaluation=False)
                    poison_optimizer.zero_grad()
                    
                    dataset_size += len(data)
                    poison_num_counter += poison_num

                    output = local_model(data)

                    class_loss = nn.functional.cross_entropy(output, targets)
                    distance_loss = helper.model_dist_norm_var(local_model, last_local_model_param_var)

                    loss = helper.params['alpha_loss'] * class_loss + (1 - helper.params['alpha_loss']) * distance_loss

                    loss.backward()
                    poison_optimizer.step()

                    total_loss += loss
                    pred = output.data.max(1)[1]
                    correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                if helper.params['poison_step_lr']:
                    scheduler.step()
                    logger.info(f'Current poison lr: {scheduler.get_last_lr()}')
                
                acc = 100.0 * (float(correct) / float(dataset_size))
                total_l = total_loss / dataset_size
                num_samples_dict[participant] = dataset_size

                logger.info(
                    'Poison Local Train {}, epoch {:3d}, local epoch {:3d}, local model {},\nAverage training loss: {:.4f}'
                    ' Training Accuracy: {}/{} ({:.4f}%), poison data number: {} '.format(
                        helper.name, current_epoch, local_epoch, participant,
                        total_l, correct, dataset_size, acc, poison_num_counter
                    )
                )

                if helper.params['scale_at_start_epoch']:
                    if participant in helper.start_attack_epoch.keys():
                        if current_epoch in helper.start_attack_epoch[participant]:
                            scale_rate = helper.params['scale_weight_at_start_epoch']
                            logger.info(f"Scale at the start epoch. Scaling by {scale_rate}")
                            for key, value in local_model.state_dict().items():
                                target_value = global_model.state_dict()[key]
                                new_value = target_value + (value - target_value) * scale_rate
                            
                                local_model.state_dict()[key].copy_(new_value)                            

                if helper.params['one_shot_attack']:
                    scale_rate = helper.params['scale_weight']
                    logger.info(f"One-shot Attack. Scaling by {scale_rate}")
                    for key, value in local_model.state_dict().items():
                        target_value = global_model.state_dict()[key]
                        new_value = target_value + (value - target_value) * scale_rate
                    
                        local_model.state_dict()[key].copy_(new_value)

        else:

            logger.info('Client {} participate the local training'.format(participant))

            _, data_iterator = helper.train_dataloader[participant]


            local_epochs = helper.params['local_epochs']
            
            optimizer = torch.optim.SGD(local_model.parameters(), lr=helper.params['lr'], momentum=helper.params['momentum'], weight_decay=helper.params['decay'])

            for local_epoch in range(helper.params['local_epochs']):
                total_loss = 0.
                correct = 0
                dataset_size = 0

                for batch_id, batch in enumerate(data_iterator):

                    optimizer.zero_grad()
                    data, targets =  helper.get_batch(batch, evaluation=False)

                    dataset_size += len(data)

                    output = local_model(data)

                    loss = nn.functional.cross_entropy(output, targets)
                    loss.backward()

                    optimizer.step()
                    total_loss += loss.data
                    pred = output.data.max(1)[1]
                    correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                acc = 100.0 * (float(correct) / float(dataset_size))
                total_l = total_loss / dataset_size
                num_samples_dict[participant] = dataset_size
                
                logger.info(
                    'Local Train {}, epoch {:3d}, local epoch {:3d}, local model {},\nAverage training loss: {:.4f}'
                    ' Training Accuracy: {}/{} ({:.4f}%)'.format(
                        helper.name, current_epoch, local_epoch, participant,
                        total_l, correct, dataset_size, acc
                    )
                )
                
        local_model_update_dict = dict()
        for name, data in local_model.state_dict().items():
            local_model_update_dict[name] = torch.zeros_like(data)
            local_model_update_dict[name] = (data - global_model.state_dict()[name])

        submit_update_dict[participant] = local_model_update_dict

        # save the submitted updates
        


    return submit_update_dict, num_samples_dict



             





                 
             

        



        