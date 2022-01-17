import argparse

def Args_parse():
    parser = argparse.ArgumentParser(description='DK')
    parser.add_argument('--params', dest='params')
    parser.add_argument('--exp_code', dest='exp_code')

    parser.add_argument('--poison_local_epochs', type=int, default=-1)
    parser.add_argument('--scale_at_start_epoch', action="store_true")
    parser.add_argument('--scale_weight_at_start_epoch', type=float, default=1.0)
    parser.add_argument('--scale_rounds_at_start_epoch', type=int, default=1)

    parser.add_argument('--0_poison_epochs', nargs='+', type=int, default=[])
    parser.add_argument('--1_poison_epochs', nargs='+', type=int, default=[])
    parser.add_argument('--2_poison_epochs', nargs='+', type=int, default=[])
    parser.add_argument('--3_poison_epochs', nargs='+', type=int, default=[])
    parser.add_argument('--4_poison_epochs', nargs='+', type=int, default=[])
    parser.add_argument('--5_poison_epochs', nargs='+', type=int, default=[])
    parser.add_argument('--6_poison_epochs', nargs='+', type=int, default=[])




    args = parser.parse_args()
    return args


    