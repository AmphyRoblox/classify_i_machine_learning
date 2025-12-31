import numpy as np

##training params
seed = 100  # 100
train_model = True
nb_epochs = 100
adv_train = False
lr = 1e-4
batch_size = 128
gpu_index = 0
signal_repre = 'origin'
patience = 20
# model_archi = ['resnet18_ghost', 'resnet18', 'mobilenet', ]  # mobilenet resnet18 mcldnn conformer resnet25
# model_archi = ['resnet10', "resnet10_ghost", 'resnet18']
# model_archi = ["resnet10", 'resnet50', "resnet18", 'mobilenet', 'shufflenet']
# model_archi = ["res2net18"]
# model_archi = ["res2netgc18"]
model_archi = ['res2netgc18', 'resnet18', 'mobilenet', 'shufflenet']
initial_flag = False
data_seed = 1
sparse_flag = False
sparse_scale = 0.00001
loss = 'focal_loss'
metric = 'linear'
margin_s = 16
margin_m = 0.35
easy_margin = False
dataset = 'wifi'
if dataset == 'adsb':
    class_nums = 20
else:
    class_nums = 16
# snr_list = np.arange(-10, 21, 2)
snr_list = np.array([0])
label_loss_weight = 0.9
soft_target_loss_weight = 0.1
temperature = 2
teacher_model = 'resnet10'
student_model = 'resnet10_ghostds'
teacher_initial_flag = True
scale_list = [4]
sample_num = 500


def get_all_variables_as_string():
    all_variables = globals()
    variable_str = ""
    for var_name, var_value in all_variables.items():
        if not var_name.startswith("__"):  # Exclude special variables
            variable_str += f"{var_name}: {var_value}\n"
    return variable_str
