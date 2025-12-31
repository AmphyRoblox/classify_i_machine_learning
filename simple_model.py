import copy
import torch.nn.functional as F
import numpy as np
import torch
import util.params as params
import random
from util.model_prepare import get_model_archi
from dataset.ads_b import get_dataloader
from util.traintest import train_model
import sys
import logging
import os
from util.traintest import validation_evaluation, validation_evaluation_all_snr
from dataset.ads_b import get_dataloader as get_adsb
from dataset.wifi_data import get_dataloader as get_wifi
import torch.optim as optim

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)

os.environ['NUMEXPR_MAX_THREADS'] = '16'

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device {device}')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)
    for archi in params.model_archi:
        print(f'Using net {archi}')
        ## add log
        if not os.path.exists(
                '../checkpoint/{}_{}'.format(params.signal_repre, archi)):
            os.makedirs(
                '../checkpoint/{}_{}/'.format(params.signal_repre, archi))
        defense_enhanced_saver = f'../checkpoint/{params.signal_repre}_{archi}/'
        fh = logging.FileHandler(os.path.join(defense_enhanced_saver, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        param_str = params.get_all_variables_as_string()
        # print(param_str)
        logging.info(param_str)

        net = get_model_archi(device, archi)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        optimizer = torch.optim.Adam(net.parameters(), lr=params.lr, weight_decay=1e-5)
        # Initialize ExponentialLR Scheduler, set the decay factor gamma as 0.95
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        if params.dataset == 'adsb':
            data, val_x, val_y = get_adsb(params.signal_repre, np.arange(params.class_nums))
        else:
            data, val_x, val_y = get_wifi(params.signal_repre, np.arange(params.class_nums))
        train_param = {'loss_fn': loss_fn, 'optimizer': optimizer, 'train_loader': data.train,
                       'validation_loader': data.test, 'device': device, 'num_epochs': params.nb_epochs,
                       'scheduler': scheduler, 'sparse_flag': params.sparse_flag, 'sparse_scale': params.sparse_scale,
                       'temperature': params.temperature, 'soft_target_loss_weight': params.soft_target_loss_weight,
                       'label_loss_weight': params.label_loss_weight}
        if not params.initial_flag:
            train_model(defense_enhanced_saver, net, train_param)
        net.load_state_dict(torch.load(defense_enhanced_saver + 'model_best.pth.tar'))

        # val_acc = validation_evaluation_all_snr(model_saver_root=defense_enhanced_saver, model=net, epoch=0,
        #                                         val_x=val_x, val_y=val_y, device=device,
        #                                         return_features=True)

        # print('validation dataset accuracy is {:.4f}'.format(val_acc))
