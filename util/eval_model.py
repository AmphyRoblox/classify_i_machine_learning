import copy
import math
import random
import time
from collections import OrderedDict, defaultdict
from typing import Union, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm

from torchprofile import profile_macs
from models.ghost_resnet import resnet18_ghost
from models.mobilenet import MobileNet
from models.resnet2d import ResNet18


def get_model_macs(model, inputs) -> int:
    return profile_macs(model, inputs)


def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements


def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width


def plot_bn_weight_distribution(model, bins=256, count_nonzero_only=False):
    fig, axes = plt.subplots(3, 3, figsize=(10, 6))
    axes = axes.ravel()
    plot_index = 0

    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):  # Check if it is a BatchNorm layer
            for param_name, param in module.named_parameters():
                if plot_index >= len(axes):  # Prevent insufficient number of subgraphs
                    break
                ax = axes[plot_index]
                if count_nonzero_only:
                    param_cpu = param.detach().view(-1).cpu()
                    param_cpu = param_cpu[param_cpu != 0].view(-1)
                    ax.hist(param_cpu, range=(-0.5, 0.5), bins=bins, density=True,
                            color='blue', alpha=0.5)
                else:
                    ax.hist(param.detach().view(-1).cpu(), range=(-0.5, 0.5), bins=bins, density=True,
                            color='blue', alpha=0.5)
                ax.set_xlabel(f'{name}.{param_name}')
                ax.set_ylabel('density')
                plot_index += 1

    fig.suptitle('Histogram of BatchNorm Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()


def plot_weight_distribution(model, bins=4, count_nonzero_only=False):
    fig, axes = plt.subplots(4, 4, figsize=(10, 6))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            if 'conv' in name:
                if plot_index >= len(axes):  # Prevent insufficient number of subgraphs
                    break
                ax = axes[plot_index]
                if count_nonzero_only:
                    param_cpu = param.detach().view(-1).cpu()
                    param_cpu = param_cpu[param_cpu != 0].view(-1)
                    ax.hist(param_cpu, range=(-0.5, 0.5), bins=bins, density=True,
                            color='blue', alpha=0.5)
                else:
                    ax.hist(param.detach().view(-1).cpu(), range=(-0.5, 0.5), bins=bins, density=True,
                            color='blue', alpha=0.5)
                ax.set_xlabel(name)
                ax.set_ylabel('density')
                plot_index += 1

    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()


def test_fine_grained_prune(
        test_tensor=torch.tensor([[-0.46, -0.40, 0.39, 0.19, 0.37],
                                  [0.00, 0.40, 0.17, -0.15, 0.16],
                                  [-0.20, -0.23, 0.36, 0.25, 0.03],
                                  [0.24, 0.41, 0.07, 0.13, -0.15],
                                  [0.48, -0.09, -0.36, 0.12, 0.45]]),
        test_mask=torch.tensor([[True, True, False, False, False],
                                [False, True, False, False, False],
                                [False, False, False, False, False],
                                [False, True, False, False, False],
                                [True, False, False, False, True]]),
        target_sparsity=0.75, target_nonzeros=None):
    def plot_matrix(tensor, ax, title):
        ax.imshow(tensor.cpu().numpy() == 0, vmin=0, vmax=1, cmap='tab20c')
        ax.set_title(title)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        for i in range(tensor.shape[1]):
            for j in range(tensor.shape[0]):
                text = ax.text(j, i, f'{tensor[i, j].item():.2f}',
                               ha="center", va="center", color="k")

    test_tensor = test_tensor.clone()
    fig, axes = plt.subplots(1, 2, figsize=(6, 10))
    ax_left, ax_right = axes.ravel()
    plot_matrix(test_tensor, ax_left, 'dense tensor')

    sparsity_before_pruning = get_sparsity(test_tensor)
    mask = fine_grained_prune(test_tensor, target_sparsity)
    sparsity_after_pruning = get_sparsity(test_tensor)
    sparsity_of_mask = get_sparsity(mask)

    plot_matrix(test_tensor, ax_right, 'sparse tensor')
    fig.tight_layout()
    plt.show()

    print('* Test fine_grained_prune()')
    print(f'    target sparsity: {target_sparsity:.2f}')
    print(f'        sparsity before pruning: {sparsity_before_pruning:.2f}')
    print(f'        sparsity after pruning: {sparsity_after_pruning:.2f}')
    print(f'        sparsity of pruning mask: {sparsity_of_mask:.2f}')

    if target_nonzeros is None:
        if test_mask.equal(mask):
            print('* Test passed.')
        else:
            print('* Test failed.')
    else:
        if mask.count_nonzero() == target_nonzeros:
            print('* Test passed.')
        else:
            print('* Test failed.')


if __name__ == "__main__":
    model = resnet18_ghost(num_classes=90)
    model = MobileNet(num_classes=90)
    model = ResNet18(n_class=90)
    model.load_state_dict(torch.load('model_best.pth.tar'))
    plot_weight_distribution(model)
    plot_bn_weight_distribution(model)
    # plot_weight_distribution(model, count_nonzero_only=True)
    # plot_bn_weight_distribution(model, count_nonzero_only=True)
