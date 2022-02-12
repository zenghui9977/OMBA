import torch
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MNIST_TYPE = 'mnist'
CIFAR_TYPE = 'cifar'


DETECTION_METRICS_FILE = 'output_eval_result.csv'
DETECTION_MODEL_FOLDER = 'models_and_localupdates'

