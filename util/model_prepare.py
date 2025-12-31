import util.params as params
from models.CNN1D import ResNet1D
from models.mobilenet import MobileNet
from models.mcldnn import MCLDNN
from models.resnet2d import ResNet8, ResNet18, ResNet25
from torchinfo import summary
from models.discriminator import Discriminator
from models.encoder_and_projection import Encoder_and_projection
from models.ghost_resnet import resnet18_ghost, resnet18_ghost_backbone
from models.resnet50_metric import resnet50, ghost_resnet50
from models.resnet18_metric import resnet18, ghost_resnet18, resnet10, ghost_resnet10, ghostds_resnet10, \
    res2net_resnet18, res2netgc_resnet18, res2net_resnet10, res2netgc_resnet10
from models.shufflenetv2 import shufflenet
from models.resnet18_metric_complex import complex_resnet18
from models.resnet1d import res2netgc_resnet18 as res2netgc_resnet18_1d
from models.resnet1d import res2netgc_resnet10 as res2netgc_resnet10_1d
from models.resnet1d import resnet18 as resnet18d1
from models.resnet1d import resnet10 as resnet10d1
from models.mobilenet1d import mobilenet
from models.shufflenet1d import shufflenet1d


def get_discriminator(device):
    print('#----------Preparing for the discriminator----------#')
    net = Discriminator(input_dims=params.d_input_dims,
                        hidden_dims=params.d_hidden_dims,
                        output_dims=params.d_output_dims)
    if device == "cuda":
        net = net.cuda()
    return net


def get_res2net_scale(device, scale):
    net = res2netgc_resnet18(num_classes=params.class_nums, scale=scale, scale_flag=True)
    if device == "cuda":
        net = net.cuda()
    summary(net, (16, 2, 4800))
    return net


def get_res2net_scale_1d(device, scale=4):
    net = res2netgc_resnet18_1d(num_classes=params.class_nums, scale=scale, scale_flag=True)
    if device == "cuda":
        net = net.cuda()
    summary(net, (16, 2, 4800))
    return net


def get_res2net10_scale_1d(device, scale=4):
    net = res2netgc_resnet10_1d(num_classes=params.class_nums, scale=scale, scale_flag=True)
    if device == "cuda":
        net = net.cuda()
    summary(net, (16, 2, 4800))
    return net


def get_model_archi(device, model_archi):
    print('#----------Preparing for the untrained network----------#')
    if model_archi == 'resnet1d':
        net = ResNet1D(num_classes=params.class_nums)
    elif model_archi == 'mobilenet':
        # net = MobileNet(num_classes=params.class_nums)
        net = mobilenet(class_num=params.class_nums)
    elif model_archi == 'mcldnn':
        net = MCLDNN(num_classes=params.class_nums)
    elif model_archi == 'resnet10':
        net = resnet10(num_classes=params.class_nums)
    elif model_archi == 'resnet10d1':
        net = resnet10d1(num_classes=params.class_nums)
    elif model_archi == 'resnet10_ghost':
        net = ghost_resnet10(num_classes=params.class_nums)
    elif model_archi == 'resnet10_ghostds':
        net = ghostds_resnet10(num_classes=params.class_nums)
    elif model_archi == 'resnet10_res2net':
        net = res2net_resnet10(num_classes=params.class_nums)
    elif model_archi == 'resnet10_res2netgc':
        net = res2netgc_resnet10(num_classes=params.class_nums)
    elif model_archi == 'resnet18':
        # net = ResNet18(n_class=params.class_nums)
        net = resnet18(num_classes=params.class_nums)
    elif model_archi == 'resnet18d1':
        net = resnet18d1(num_classes=params.class_nums)
    elif model_archi == 'complex_resnet18':
        # net = ResNet18(n_class=params.class_nums)
        net = complex_resnet18(num_classes=params.class_nums)
    elif model_archi == 'res2net18':
        net = res2net_resnet18(num_classes=params.class_nums)
    elif model_archi == 'res2netgc18':
        # net = res2netgc_resnet18(num_classes=params.class_nums)
        net = res2netgc_resnet18_1d(num_classes=params.class_nums, scale_flag=True)
    elif model_archi == 'resnet25':
        net = ResNet25(n_class=params.class_nums)
    elif model_archi == 'resnet18_ghost':
        # net = resnet18_ghost(num_classes=params.class_nums)
        net = ghost_resnet18(num_classes=params.class_nums)
    elif model_archi == 'resnet50':
        net = resnet50(num_classes=params.class_nums)
    elif model_archi == 'resnet50_ghost':
        net = ghost_resnet50(num_classes=params.class_nums)
    elif model_archi == 'shufflenet':
        net = shufflenet1d(num_classes=params.class_nums)
    elif model_archi == 'SA2SEI':
        net = Encoder_and_projection()
    else:
        net = ResNet8(num_classes=params.class_nums)
    if device == "cuda":
        net = net.cuda()
    summary(net, (16, 2, 4800))
    return net


def get_model_backbone(device, model_archi):
    print('#----------Preparing for the untrained network----------#')
    if model_archi == 'resnet1d':
        net = ResNet1D(num_classes=params.class_nums)
    elif model_archi == 'mobilenet':
        net = MobileNet(num_classes=params.class_nums, backbone=True)
    elif model_archi == 'resnet18':
        net = ResNet18(n_class=params.class_nums, backbone=True)
    elif model_archi == 'resnet18_ghost':
        net = resnet18_ghost_backbone(num_classes=params.class_nums)
    elif model_archi == 'SA2SEI':
        net = Encoder_and_projection()
    else:
        net = ResNet8(num_classes=params.class_nums)
    if device == "cuda":
        net = net.cuda()
    summary(net, (16, 2, 4800))
    return net
