from collections import defaultdict
import numpy as np
import copy
import logging
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import os
import random
import yaml
import constant_str
from torch.utils.data import DataLoader, Dataset, sampler

from models.MNIST_model import LeNet5
from models.CIFAR_model import ResNet18

logger = logging.getLogger("logger")




class Helper:
    def __init__(self, params, name, current_time, args):
        
        self.params = params
        self.name = name
        self.args = vars(args)
        self.train_dataset = None
        self.test_dataset = None
        self.local_model = None
        self.target_model = None
        self.class_dict = None
        self.poisoned_test_dataloader = None
        self.targetlabel_test_dataloader = None
        self.indices_per_participant = None
        self.participants_namelist = list(range(self.params['number_of_total_participants']))
        self.advasarial_namelist = self.params['adversary_list']
        self.benign_namelist = list(set(self.participants_namelist) - set(self.advasarial_namelist))
        self.poison_test_data_idx = None
        self.backdoor_success_samples_dataloader = None
        self.start_attack_epoch = {}
        


        if self.params['attack_mode'] == 'on_and_off_attack':
            logger.info('Processing the on and off attack epochs......')
            self.process_params()

        self.attack_time_set = self.get_attack_set()

        # config the log information
        if self.params['is_test']:
            self.folder_path = f'exp_records/test'
        else:
            self.folder_path = f'exp_records/model_{self.name}_{current_time}'
        try:
            os.makedirs(self.folder_path)
        except FileExistsError:
            logger.info(f'{self.folder_path}. Folder already exists')   
        logger.addHandler(logging.FileHandler(filename=f'{self.folder_path}/log.txt'))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
        logger.info(f'current path: {self.folder_path}')

        with open(f'{self.folder_path}/params.yaml', 'w') as f:
            yaml.dump(self.params, f)
        logger.info('save the parameters finished......')

        # create the folder to save the models and local updates during the training process
        
        self.model_saved_folder = self.folder_path + '/models_and_localupdates/'
        try:
            os.makedirs(self.model_saved_folder)
        except FileExistsError:
            logger.info(f'{self.model_saved_folder}. Folder already exists') 
        
        # create the folder to save the pcitures
        self.pic_saved_folder = self.folder_path + '/pictures/'
        try:
            os.makedirs(self.pic_saved_folder)
        except FileExistsError:
            logger.info(f'{self.pic_saved_folder}. Folder already exists') 



    @staticmethod
    def model_dist_norm_var(model, target_params_variables, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.FloatTensor(size).fill_(0)
        sum_var= sum_var.to(constant_str.device)
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
                    layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)

    @staticmethod
    def dp_noise(param, sigma):

        noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)

        return noised_layer   


    def process_params(self):

        # trans local_poison_epochs to param
        if self.args['poison_local_epochs'] >= 1:
            self.params['poison_local_epochs'] = self.args['poison_local_epochs']

        self.params['scale_at_start_epoch'] = self.args['scale_at_start_epoch']
        self.params['scale_weight_at_start_epoch'] = self.args['scale_weight_at_start_epoch']
        self.params['scale_rounds_at_start_epoch'] = self.args['scale_rounds_at_start_epoch']


        # trans args to params
        for i in range(len(self.params['adversary_list'])):
            args_name = f'{i}_poison_epochs'
            self.params[args_name] = self.args[args_name]

        # processing the attackers participates list
        for idx in range(len(self.params['adversary_list'])):
            temp_setting = self.params[str(idx) + '_poison_epochs']
            if len(temp_setting) == 4:
                temp_list = generate_attack_time(temp_setting[0], temp_setting[1], temp_setting[2], temp_setting[3])
                self.params[str(idx) + '_poison_epochs'] = temp_list
                self.start_attack_epoch[self.params['adversary_list'][idx]] = temp_list[:self.args['scale_rounds_at_start_epoch']]


    def cos_simlarity(self, output_vector, failure_vector):
        cs = F.cosine_similarity(output_vector, failure_vector)
        return cs


    def build_classes_dict(self):
        classes_dict = {}
        for ind, x in enumerate(self.train_dataset):  # for cifar: 50000; for tinyimagenet: 100000
            _, label = x
            if label in classes_dict:
                classes_dict[label].append(ind)
            else:
                classes_dict[label] = [ind]
        return classes_dict

    def sample_dirichlet_train_data(self, num_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """
        class_dict = self.class_dict
        each_class_size = len(class_dict[0])
        per_participant_list = defaultdict(list)

        num_class = len(class_dict.keys())

        img_nums = []
        for n in range(num_class):
            img_num = []
            random.shuffle(class_dict[n])
            sampled_probabilities = each_class_size * np.random.dirichlet(np.array(num_participants * [alpha]))

            for user in range(num_participants):
                num_imgs_each_user = int(round(sampled_probabilities[user]))
                sampled_list = class_dict[n][:min(len(class_dict[n]), num_imgs_each_user)]
                img_num.append(len(sampled_list))

                per_participant_list[user].extend(sampled_list)
                class_dict[n] = class_dict[n][min(len(class_dict[n]), num_imgs_each_user):]
            
            img_nums.append(img_num)

        return per_participant_list

    def get_train(self, indices):
        train_loader = DataLoader(self.train_dataset, batch_size=self.params['batch_size'], sampler=sampler.SubsetRandomSampler(indices))
        return train_loader

    def get_test(self):
        test_loader = DataLoader(self.test_dataset, batch_size=self.params['batch_size'], shuffle=True)
        return test_loader

    def poison_test_data(self):
        test_classes = {}
        for idx, x in enumerate(self.test_dataset):
            _, label = x
            if label in test_classes:
                test_classes[label].append(idx)
            else:
                test_classes[label] = [idx]
        
        idx_list = list(range(0, len(self.test_dataset)))
        
        # the data list except the poisoned label
        for image_idx in test_classes[self.params['poison_label_swap']]:
            if image_idx in idx_list:
                idx_list.remove(image_idx)
        # the data list that all the label is the poisoned label
        poisoned_label_idx = test_classes[self.params['poison_label_swap']]

        poisoned_test_dataloader = DataLoader(self.test_dataset, batch_size=self.params['batch_size'], sampler=sampler.SubsetRandomSampler(idx_list))
        targetlabel_test_dataloader = DataLoader(self.test_dataset, batch_size=self.params['batch_size'], sampler=sampler.SubsetRandomSampler(poisoned_label_idx))

        self.poison_test_data_idx = idx_list

        return poisoned_test_dataloader, targetlabel_test_dataloader
    
    def get_batch(self, bptt, evaluation=False):
        data, target = bptt
        data = data.to(constant_str.device)
        target = target.to(constant_str.device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target

    def get_poison_batch(self, bptt, evaluation=False):

        images, targets = bptt

        poison_count = 0
        new_images, new_targets = images, targets

        for index in range(0, len(images)):
            if evaluation:
                new_targets[index] = self.params['poison_label_swap']
                new_images[index] = self.add_pixel_pattern(images[index])
                poison_count += 1
            else:
                if index < self.params['poisoning_per_batch']:
                    new_targets[index] = self.params['poison_label_swap']
                    new_images[index] = self.add_pixel_pattern(images[index])
                    poison_count += 1
                else:
                    new_images[index] = images[index]
                    new_targets[index] = targets[index]

        new_images, new_targets = new_images.to(constant_str.device), new_targets.to(constant_str.device)
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)

        return new_images, new_targets, poison_count

    def add_pixel_pattern(self, ori_image):
        image = copy.deepcopy(ori_image)
        poison_patterns = self.params['poison_pattern']

        if self.params['type'] == constant_str.CIFAR_TYPE:
            for poison_pat_pos in poison_patterns:
                image[0][poison_pat_pos[0]][poison_pat_pos[1]] = 1
                image[1][poison_pat_pos[0]][poison_pat_pos[1]] = 1
                image[2][poison_pat_pos[0]][poison_pat_pos[1]] = 1
        elif self.params['type'] == constant_str.MNIST_TYPE:
            for poison_pat_pos in poison_patterns:
                image[0][poison_pat_pos[0]][poison_pat_pos[1]] = 1
        
        return image

        
    def create_model(self):
        local_model = None
        target_model = None
        try:
            os.makedirs("./saved_models/")
        except FileExistsError:
            logger.info('Folder already exists') 

        if self.params['type'] == constant_str.MNIST_TYPE:
            local_model = LeNet5()
            target_model = LeNet5()

        elif self.params['type'] == constant_str.CIFAR_TYPE:
            local_model = ResNet18()
            target_model = ResNet18()

        local_model.to(constant_str.device)
        target_model.to(constant_str.device)
        if self.params['use_pretrain_model']:
            load_model_parameters = torch.load(f"saved_models/pretrain_{self.params['type']}")
            logger.info(f"load the pretrain model from: saved_models/pretrain_{self.params['type']}")
            target_model.load_state_dict(load_model_parameters)
            self.target_model = target_model
        else:
            self.local_model = local_model
            self.target_model = target_model


    def init_weight_accumulator(self):
        weight_accumulator = dict()
        for name, data in self.target_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)
        return weight_accumulator

    def accumulate_weight(self, weight_accumulator, submit_update_dict, participants):
          
        for participant in participants:
            local_update = submit_update_dict[participant]
            for name, _ in local_update.items():
                weight_accumulator[name].add_(local_update[name])
        return weight_accumulator

    def average_shrink_model(self, weight_accumulator, target_model):

        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                continue   
            updates_per_layer = weight_accumulator[name] * (self.params['eta']/self.params['number_of_each_round_participants'])

            if self.params['diff_privacy']:
                updates_per_layer.add_(self.dp_noise(data, self.params['sigma']))
            
            data.add_(updates_per_layer)
        return target_model

    
    def load_data(self):
        
        dataPath = './data/'

        if self.params['type'] == constant_str.CIFAR_TYPE:
            simple_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            self.train_dataset = datasets.CIFAR10(dataPath, train=True, download=False, transform=simple_transform)
            self.test_dataset = datasets.CIFAR10(dataPath, train=False, transform=simple_transform)

        elif self.params['type'] == constant_str.MNIST_TYPE:
            
            simple_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            self.train_dataset = datasets.MNIST(dataPath, train=True, transform=simple_transform)
            self.test_dataset = datasets.MNIST(dataPath, train=False, transform=simple_transform)

        self.class_dict = self.build_classes_dict()

        if self.params['sampling_dirichlet']:
            indices_per_participant = self.sample_dirichlet_train_data(self.params['number_of_total_participants'], alpha=self.params['dirichlet_alpha'])
            
            train_loader = [(pos, self.get_train(indices)) for pos, indices in indices_per_participant.items()]
            test_loader = self.get_test()

        self.train_dataloader = train_loader
        self.test_dataloader = test_loader

        self.poisoned_test_dataloader, self.targetlabel_test_dataloader = self.poison_test_data()

        self.indices_per_participant = indices_per_participant


    def choose_clients_every_epoch(self, current_epoch):
        participants_list_current = []
        adversarial_list_current = []
        participatant_num = self.params['number_of_each_round_participants']
        if self.params['is_poison']:
            if self.params['is_random_adversary']:
                participants_list_current = random.sample(self.participants_namelist, participatant_num)    
                for i in participants_list_current:
                    if i in self.params['adversary_list']:
                        adversarial_list_current.append(i)
                
            else:
                for idx_adv in range(len(self.params['adversary_list'])):
                    if current_epoch in self.params[str(idx_adv) + '_poison_epochs']:
                        participants_list_current.append(self.params['adversary_list'][idx_adv])
                        adversarial_list_current.append(self.params['adversary_list'][idx_adv])

                non_attacker_list = []
                for adv in self.params['adversary_list']:
                    if adv not in adversarial_list_current:
                        non_attacker_list.append(adv)
                
                benign_num = self.params['number_of_each_round_participants'] - len(adversarial_list_current)
                random_client_list = random.sample(self.benign_namelist + non_attacker_list, benign_num)
                participants_list_current += random_client_list


        else:
            participants_list_current = random.sample(self.benign_namelist, participatant_num)

        return participants_list_current, adversarial_list_current

    def process_backdoor_samples(self, input_tensor):
        
        input_tensor = self.add_pixel_pattern(input_tensor)
        input_tensor = torch.unsqueeze(input_tensor, dim=0)
        
        input_tensor = input_tensor.to(constant_str.device)

        label_tensor = self.params['poison_label_swap']
        return input_tensor, label_tensor

    def get_backdoor_success_samples(self, indices):
        success_backdoor_samples_dataloader = DataLoader(self.test_dataset, batch_size=self.params['failure_samples_num'], sampler=sampler.SubsetRandomSampler(indices))
        return success_backdoor_samples_dataloader

    def get_attack_set(self):
        attackers_num = len(self.params['adversary_list'])
        attack_time_set = set()
        for i in range(attackers_num):
            attack_time_set = attack_time_set | set(self.params[str(i) + '_poison_epochs'])
        
        return attack_time_set



    def filter_success_fail_samples(self):

        the_final_epoch = self.params['epochs'] - 1
        if self.params['test_folder'] == 'None': 
            test_folder = self.model_saved_folder
            the_final_global_model_path = f'{test_folder}/{the_final_epoch}/epoch_{the_final_epoch}_global_model'
        else:
            test_folder = self.params['test_folder']
            the_final_global_model_path = f'{test_folder}/models_and_localupdates/{the_final_epoch}/epoch_{the_final_epoch}_global_model'

        logger.info('-'*40)
        logger.info(f'Using the final model.')
        logger.info(f'load from {the_final_global_model_path}')
        logger.info('-'*40)

        the_final_global_model_dict = torch.load(the_final_global_model_path)
        the_final_global_model = copy.deepcopy(self.local_model)
        the_final_global_model.load_state_dict(the_final_global_model_dict)
        the_final_global_model.eval()

        test_data_ = copy.deepcopy(self.test_dataset)

        # backdoor success samples
        success_samples_idx = copy.deepcopy(self.poison_test_data_idx)
        backdoor_correct_idx,  backdoor_fail_idx= [], []
        # print(len(success_samples_idx))
        for i in range(len(success_samples_idx)):
            # test whether the sample is correct
            input_tensor, label_tensor = self.process_backdoor_samples(test_data_[success_samples_idx[i]][0])

            out_put = the_final_global_model(input_tensor)
            pred = out_put.data.max(1)[1]

            if not pred == label_tensor:
                backdoor_fail_idx.append(success_samples_idx[i])
            else:
                backdoor_correct_idx.append(success_samples_idx[i])


        # sample the fixed number of samples from the samples which successfully backdoor the global model
        selected_samples_idx = random.sample(backdoor_correct_idx, self.params['failure_samples_num'])
        backdoor_success_samples_dataloader = self.get_backdoor_success_samples(selected_samples_idx)

        self.backdoor_success_samples_dataloader = backdoor_success_samples_dataloader


def generate_attack_time(on_, off_, numbers, start_epoch):
    attack_list = []
    for n in range(numbers):
        for i in range(on_):
            attack_list.append(i + start_epoch)
        start_epoch = start_epoch + on_ + off_
    
    return attack_list

        


        
        
        








        
    
        