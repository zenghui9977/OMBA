:: python main.py --params ./params_folder/mnist.yaml
@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 3 7 3 40
@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 3 7 4 40
@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 4 6 4 40
@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 5 5 4 40
@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 4 7 4 40
@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 5 7 4 40
@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 6 7 4 40
@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 3 7 3 40 --poison_local_epochs 4
@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 3 7 3 40 --poison_local_epochs 5
@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 3 7 3 40 --poison_local_epochs 6

@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 3 7 3 40 --poison_local_epochs 10


@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 3 7 3 40 --poison_local_epochs 3 --scale_at_start_epoch --scale_weight_at_start_epoch 2
@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 3 7 3 40 --poison_local_epochs 3 --scale_at_start_epoch --scale_weight_at_start_epoch 4
@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 3 7 3 40 --poison_local_epochs 6 --scale_at_start_epoch --scale_weight_at_start_epoch 2
@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 3 7 3 40 --poison_local_epochs 6 --scale_at_start_epoch --scale_weight_at_start_epoch 4


@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 3 7 3 40 --poison_local_epochs 3 --scale_at_start_epoch --scale_weight_at_start_epoch 2 --scale_rounds_at_start_epoch 1
@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 3 7 3 40 --poison_local_epochs 3 --scale_at_start_epoch --scale_weight_at_start_epoch 2 --scale_rounds_at_start_epoch 2
@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 3 7 3 40 --poison_local_epochs 3 --scale_at_start_epoch --scale_weight_at_start_epoch 2 --scale_rounds_at_start_epoch 3
@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 3 7 3 40 --poison_local_epochs 3 --scale_at_start_epoch --scale_weight_at_start_epoch 2 --scale_rounds_at_start_epoch 4

@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 1 5 2 40 --1_poison_epochs 1 5 2 40 --poison_local_epochs 3 
@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 1 5 2 40 --1_poison_epochs 1 5 2 40 --poison_local_epochs 3 

@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 2 5 3 40 --1_poison_epochs 2 5 3 40 --poison_local_epochs 3 
@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 2 5 3 40 --1_poison_epochs 2 5 3 40 --poison_local_epochs 3 

@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 2 5 3 40 --1_poison_epochs 2 5 3 40 --2_poison_epochs 2 5 3 40 --poison_local_epochs 3 
@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 2 5 3 40 --1_poison_epochs 2 5 3 40 --2_poison_epochs 2 5 3 40 --poison_local_epochs 3 
@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 2 5 3 40 --1_poison_epochs 2 5 3 40 --2_poison_epochs 2 5 3 40 --poison_local_epochs 3 


@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 1 3 3 40 --1_poison_epochs 1 3 3 40 --2_poison_epochs 1 3 3 40 --poison_local_epochs 4

@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 1 3 3 40 --1_poison_epochs 1 3 3 40 --2_poison_epochs 1 3 3 40 --poison_local_epochs 4

@REM python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 1 3 3 40 --1_poison_epochs 1 3 3 40 --2_poison_epochs 1 3 3 40 --poison_local_epochs 4


python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 1 3 4 40 --1_poison_epochs 1 3 4 40 --2_poison_epochs 1 3 4 40 --poison_local_epochs 3 

python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 1 3 4 40 --1_poison_epochs 1 3 4 40 --2_poison_epochs 1 3 4 40 --poison_local_epochs 3 

python main.py --params ./params_folder/mnist_on_and_off_attack.yaml --0_poison_epochs 1 3 4 40 --1_poison_epochs 1 3 4 40 --2_poison_epochs 1 3 4 40 --poison_local_epochs 3 


