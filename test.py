from args_setting import Args_parse
from helper import Helper, generate_attack_time


args = Args_parse()

vars_args = vars(args)


print(vars_args)


print(generate_attack_time(3, 7, 4, 40))