"""Adversarial adaptation to train target encoder."""

import os

import torch
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import util.params as params


def train_tgt(src_encoder, tgt_encoder, critic,
              src_data_loader, tgt_data_loader, device, save_root):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()
    writer = SummaryWriter("logs")

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))
    cur_i = 0

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.adapt_epoch):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, _), (images_tgt,)) in data_zip:
            ###########################
            # 2.1 train discriminator #
            ###########################

            # make images variable
            images_src = images_src.to(device)
            images_tgt = images_tgt.to(device)

            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(images_src)
            feat_tgt = tgt_encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())

            # prepare real and fake label
            label_src = torch.ones(feat_src.size(0)).long().to(device)
            label_tgt = torch.zeros(feat_tgt.size(0)).long().to(device)
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # optimize critic
            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################

            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_encoder(images_tgt)

            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            label_tgt = torch.ones(feat_tgt.size(0)).long().to(device)

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()

            #######################
            # 2.3 print step info #
            #######################
            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                      .format(epoch + 1,
                              params.adapt_epoch,
                              step + 1,
                              len_data_loader,
                              loss_critic.item(),
                              loss_tgt.item(),
                              acc.item()))
            writer.add_scalars("Domain adaptation",
                               {'Target loss': loss_tgt.item(), 'Discriminator loss': loss_critic.item(),
                                'Discriminator accuracy': acc.item()}, cur_i)
            cur_i = cur_i + 1

        #############################
        # 2.4 save model parameters #
        #############################
        if (epoch + 1) % params.save_step == 0:
            save_path = os.path.join(save_root, 'critic.pth_{}.tar'.format(epoch))
            torch.save(critic.state_dict(), save_path)

            save_path = os.path.join(save_root, 'tgt_encoder.pth_{}.tar'.format(epoch))
            torch.save(tgt_encoder.state_dict(), save_path)
            # best_save_path = os.path.join(save_root, 'model_best.pth.tar')
            # shutil.copyfile(save_path, best_save_path)

            # torch.save(critic.state_dict(), os.path.join(
            #     params.model_root,
            #     "ADDA-critic-{}.pt".format(epoch + 1)))
            # torch.save(tgt_encoder.state_dict(), os.path.join(
            #     params.model_root,
            #     "ADDA-target-encoder-{}.pt".format(epoch + 1)))
    writer.close()
    torch.save(critic.state_dict(), os.path.join(
        save_root, 'critic.pth_final.tar'))
    torch.save(tgt_encoder.state_dict(), os.path.join(
        save_root, 'tgt_encoder.pth_final.tar'
    ))
    return tgt_encoder
