import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from util.data_prepare import get_dataloader
import util.params as params
import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--signal_shape", type=tuple, default=(2, 5000), help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, np.prod(opt.signal_shape)),
            nn.Tanh()
        )

    def forward(self, z):
        signal = self.model(z)
        signal = signal.view(signal.size(0), *opt.signal_shape)
        return signal


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(opt.signal_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------
def wgan_train(dataloader, netd, netg, lambda_gp):
    # Optimizers
    optimizer_G = torch.optim.Adam(netg.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(netd.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            fake_imgs = netg(z)

            # Real images
            real_validity = netd(real_imgs)
            # Fake images
            fake_validity = netd(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(netd, real_imgs.data, fake_imgs.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs = netg(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = netd(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                if batches_done % opt.sample_interval == 0:
                    save_aggregated_signals_and_constellations(fake_imgs)

                batches_done += opt.n_critic


def save_aggregated_signals_and_constellations(fake_imgs):
    """
    Save the time-domain waveforms and constellation diagrams of the first 4 samples into two separate files.
    Parameters:
    fake_imgs: Generated samples, assuming the dimension is [batch_size, 2, 5000]
    """
    # Select the first 4 samples
    signals = fake_imgs[:4].cpu().detach().numpy()  # Ensure conversion to a numpy array

    # Create a large time-domain waveform graph
    plt.figure(figsize=(12, 8))
    for i, signal in enumerate(signals):
        I = signal[0]  # I-channel weight
        Q = signal[1]  # Q-channel weight
        plt.subplot(4, 1, i + 1)
        plt.plot(I, label='I Channel')
        plt.plot(Q, label='Q Channel')
        plt.title(f'Sample {i + 1} - Time Domain Waveform')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/time_domain_all_samples.png')  # Save all time-domain waveform graphs into one file
    plt.close()

    # Create a large star chart
    plt.figure(figsize=(12, 12))
    for i, signal in enumerate(signals):
        I = signal[0]
        Q = signal[1]
        plt.subplot(2, 2, i + 1)
        plt.scatter(I, Q, color='blue', s=1)
        plt.title(f'Sample {i + 1} - Constellation Diagram')
        plt.xlabel('I')
        plt.ylabel('Q')
        plt.axis('equal')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/constellation_all_samples.png')  # Save all the constellation maps to a single file
    plt.close()


if __name__ == '__main__':
    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    # Loss weight for gradient penalty
    lambda_gp = 10
    if cuda:
        generator.cuda()
        discriminator.cuda()
    train_dir = "C:/Users/zhiwei/Desktop/xf/signal/part2/1/group2/train_data"
    data = get_dataloader(params.signal_repre, train_dir)
    wgan_train(data.train, discriminator, generator, lambda_gp)
