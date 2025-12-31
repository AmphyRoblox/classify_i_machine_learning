import torch
import torch.nn.functional as F
import numpy as np
import util.params as params
import logging
import sys
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
from util.model_prepare import get_model_archi
from util.data_prepare import get_dataloader, norm_data
from util.preprocessing import gendwt
import pickle
import time
import torch.nn as nn
from dataset.ads_b import trans_gaussian_noise, norm_data
from torch.utils.data import TensorDataset, DataLoader

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, save_root, epoch):

        score = -val_loss
        self.trace_func(
            f'Validation loss: {val_loss:.6f}.')

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_root, epoch)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_root, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, save_root, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        save_path = os.path.join(save_root, 'checkpoint.pth_{}.tar'.format(epoch))
        torch.save(model.state_dict(), save_path)
        best_save_path = os.path.join(save_root, 'model_best.pth.tar')
        shutil.copyfile(save_path, best_save_path)
        self.val_loss_min = val_loss


def updateBN(model, sparse_scale):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(sparse_scale * torch.sign(m.weight.data))  # L1


def train_model(model_saver_root, model=None, train_param=None):
    """

    :param model_saver_root:
    :param train_param:
    :param model:
    :return:
    """
    print("\nTraining the initial model ......\n")
    writer = SummaryWriter()
    best_val_acc = None
    num_epochs = train_param['num_epochs']
    train_loader = train_param['train_loader']
    device = train_param['device']
    optimizer = train_param['optimizer']
    validation_loader = train_param['validation_loader']
    scheduler = train_param['scheduler']
    sparse_scale = train_param['sparse_scale']
    sparse_flag = train_param['sparse_flag']
    loss_fn = train_param['loss_fn']
    fh = logging.FileHandler(os.path.join(model_saver_root, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    early_stopping = EarlyStopping(patience=params.patience, verbose=True, delta=0.001)
    val_losses = []
    train_losses = []
    train_acces = []
    val_acces = []
    # Calculate training time
    start_time = time.time()  # Record start time
    for epoch in range(num_epochs):
        model.train()  # set the model in the train mode before every epoch
        for index, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images, labels=labels, train_mode=True)
            train_loss = loss_fn(logits, labels.to(torch.int64))
            optimizer.zero_grad()
            train_loss.backward()
            if sparse_flag:
                updateBN(model, sparse_scale)
            optimizer.step()

            print('\rTrain Epoch {:>3}: [batch:{:>4}/{:>4}({:>3.0f}%)]  \tLoss: {:.4f} ===> '. \
                  format(epoch, index, len(train_loader), index / len(train_loader) * 100.0, train_loss.item()),
                  end=' ')

        print('\n')
        scheduler.step()
        train_acc, train_loss = validation_evaluation(model, epoch, train_loader, device, data_type='Training')
        val_acc, val_loss = validation_evaluation(model, epoch, validation_loader, device)
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Acc/Train', train_acc, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Acc/Val', val_acc, epoch)
        val_losses.append(val_loss)
        train_losses.append(train_loss)
        train_acces.append(train_acc)
        val_acces.append(val_acc)
        logging.info(f'Epoch: {epoch}, Accuracy: {val_acc}')

        early_stopping(val_loss, model, model_saver_root, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    writer.close()

    plot_loss_acc(train_losses, val_losses, train_acces, val_acces)
    # Training ending time
    end_time = time.time()  # Record ending time
    # Calculate the total training time
    total_time = end_time - start_time
    # Convert time into hours, minutes, and seconds
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total training time: {int(hours)}hours {int(minutes)}minutes {int(seconds)}seconds")


def plot_loss_acc(train_losses, val_losses, train_acces, val_acces):
    # Save the variables to a .mat file
    sio.savemat('./loss_plot/loss_acc_data.mat', {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_acces': train_acces,
        'val_acces': val_acces
    })

    # Plotting the loss and accuracy curves
    plt.figure(figsize=(12, 6))

    # Plot Losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", color='blue')
    plt.plot(val_losses, label="Validation Loss", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.grid()

    # Plot Accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_acces, label="Train Accuracy", color='blue')
    plt.plot(val_acces, label="Validation Accuracy", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.legend()
    plt.grid()

    # Show the plot
    plt.tight_layout()
    plt.show()


def train_model_metric(model_saver_root, model=None, metric_fc=None, train_param=None):
    """

    :param metric_fc:
    :param model_saver_root:
    :param train_param:
    :param model:
    :return:
    """
    print("\nTraining the initial model ......\n")
    writer = SummaryWriter()
    best_val_acc = None
    num_epochs = train_param['num_epochs']
    train_loader = train_param['train_loader']
    device = train_param['device']
    optimizer = train_param['optimizer']
    validation_loader = train_param['validation_loader']
    scheduler = train_param['scheduler']
    loss_fn = train_param['loss_fn']

    # Calculate training time
    start_time = time.time()  # Record start time
    for epoch in range(num_epochs):
        model.train()  # set the model in the train mode before every epoch
        for index, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            feature = model(images)
            logits = metric_fc(feature, labels.to(torch.int64))
            train_loss = loss_fn(logits, labels.to(torch.int64))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            print('\rTrain Epoch {:>3}: [batch:{:>4}/{:>4}({:>3.0f}%)]  \tLoss: {:.4f} ===> '. \
                  format(epoch, index, len(train_loader), index / len(train_loader) * 100.0, train_loss.item()),
                  end=' ')
            writer.add_scalar('Loss/Train', train_loss.item(), epoch)

        print('\n')
        scheduler.step()
        # validation_evaluation(model, epoch, train_loader, device, data_type='Training')
        val_acc, _ = validation_evaluation(model, epoch, validation_loader, device, metric_fc=metric_fc)
        if best_val_acc is None or val_acc > best_val_acc:
            best_val_acc = val_acc
            save_backbone_fc(model, metric_fc, model_saver_root)
        writer.close()

    # Training end time
    end_time = time.time()  # Record the end time
    # Calculate the total training time
    total_time = end_time - start_time
    # Convert time into hours, minutes, and seconds
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total training time: {int(hours)}hours {int(minutes)}minutes {int(seconds)}seconds")


def train_model_with_knowledge(model_saver_root, teacher=None, student=None, train_param=None):
    """
    :param student:
    :param teacher:
    :param model_saver_root:
    :param train_param:
    :return:
    """
    print("\nTraining the initial model ......\n")
    writer = SummaryWriter()
    best_val_acc = None
    num_epochs = train_param['num_epochs']
    train_loader = train_param['train_loader']
    device = train_param['device']
    optimizer = train_param['optimizer']
    validation_loader = train_param['validation_loader']
    scheduler = train_param['scheduler']
    sparse_scale = train_param['sparse_scale']
    sparse_flag = train_param['sparse_flag']
    loss_fn = train_param['loss_fn']
    T = train_param['temperature']
    soft_target_loss_weight = train_param['soft_target_loss_weight']
    label_loss_weight = train_param['label_loss_weight']
    fh = logging.FileHandler(os.path.join(model_saver_root, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    early_stopping = EarlyStopping(patience=params.patience, verbose=True)

    # Calculate training time
    start_time = time.time()  # Record start time
    for epoch in range(num_epochs):
        student.train()  # set the model in the train mode before every epoch
        for index, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                teacher_logits, _ = teacher(images, labels=labels)
            student_logits = student(images, labels=labels, train_mode=True)
            # Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling
            # the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (
                    T ** 2)
            label_loss = loss_fn(student_logits, labels.to(torch.int64))
            loss = soft_target_loss_weight * soft_targets_loss + label_loss_weight * label_loss
            optimizer.zero_grad()
            loss.backward()
            if sparse_flag:
                updateBN(student, sparse_scale)
            optimizer.step()

            print('\rTrain Epoch {:>3}: [batch:{:>4}/{:>4}({:>3.0f}%)]  \tLoss: {:.4f} ===> '. \
                  format(epoch, index, len(train_loader), index / len(train_loader) * 100.0, loss.item()),
                  end=' ')
            writer.add_scalar('Loss/Train', loss.item(), epoch)

        print('\n')
        scheduler.step()
        validation_evaluation(student, epoch, train_loader, device, data_type='Training')
        val_acc, val_loss = validation_evaluation(student, epoch, validation_loader, device)
        logging.info(f'Epoch: {epoch}, Accuracy: {val_acc}')

        if early_stopping.early_stop:
            print("Early stopping")
            break

        early_stopping(val_acc, student, model_saver_root, epoch)
        writer.close()

    # End time for training
    end_time = time.time()  # Record the end time
    # Calculate the total training time
    total_time = end_time - start_time
    # Convert time into hours, minutes, and seconds
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total training time: {int(hours)}hours {int(minutes)}seconds {int(seconds)}minutes")


def save_backbone_fc(model, fc, model_dir):
    save_model_dir = os.path.join(model_dir, 'backbone_max_acc.pth.tar')
    torch.save(model.state_dict(), save_model_dir)
    save_fc_dir = os.path.join(model_dir, 'metric_fc.pth.tar')
    torch.save(fc.state_dict(), save_fc_dir)


def validation_evaluation(model, epoch, validation_loader, device, return_features=False, data_type='Validation',
                          snr=1000, archi=' '):
    model = model.to(device)
    model.eval()
    val_losses = []
    total = 0.0
    correct = 0.0
    all_labels = []
    all_preds = []
    all_features = []
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    with torch.no_grad():
        for index, (inputs, labels) in enumerate(validation_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            if return_features:
                outputs, features = model(inputs)
                all_features.extend(features.cpu().numpy())
            else:
                outputs, _ = model(inputs)

            loss = F.cross_entropy(outputs, labels.to(torch.int64))
            val_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    ratio = correct / total
    print(f'{data_type} Accuracy: {ratio:.4f}')
    print(f'{data_type} Loss: {np.average(val_losses):.4f}')
    import matplotlib.ticker as mticker

    
    if return_features:
        tsne = TSNE(n_components=2, random_state=0)
        t_features = tsne.fit_transform(np.array(all_features))

        # Choose a different color for each category
        unique_labels = np.unique(all_labels)
        colors = plt.cm.get_cmap('tab20', len(unique_labels))

        # Create a scatter plot
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(t_features[:, 0], t_features[:, 1], c=all_labels, cmap=colors, s=100, edgecolors='white')
        # plt.grid(True)  # Optional: add grid lines to improve readability
        plt.savefig(f"./res_plot/dot_plot/{snr}_{archi}_{params.metric}.svg", format="svg")
        plt.show()

    return ratio, np.average(val_losses)
    # return ratio, val_losses


def validation_evaluation_all_snr(model_saver_root, model, val_x, val_y, epoch, device, return_features=False,
                                  data_type='Validation'):
    fh = logging.FileHandler(os.path.join(model_saver_root, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    model = model.to(device)
    model.eval()
    val_losses = []
    total = 0.0
    correct = 0.0
    all_labels = []
    all_preds = []
    all_features = []
    snr_acc = []
    for i, snr in enumerate(params.snr_list):
        X_val, Y_val = trans_gaussian_noise(val_x, val_y, snr)
        X_val, val_norm_par = norm_data(X_val)
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
        validation_loader = DataLoader(val_dataset, batch_size=params.batch_size,
                                       num_workers=0, shuffle=False)
        with torch.no_grad():
            for index, (inputs, labels) in enumerate(validation_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                if return_features:
                    outputs, features = model(inputs)
                    all_features.extend(features.cpu().numpy())
                else:
                    outputs, _ = model(inputs)

                loss = F.cross_entropy(outputs, labels.to(torch.int64))
                val_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        ratio = correct / total
        print(f'data: {data_type} snr: {snr} Accuracy: {ratio:.4f}')
        print(f'data: {data_type} snr: {snr} Loss: {np.average(val_losses):.4f}')
        logging.info(f'data: {data_type} snr: {snr} Accuracy: {ratio:.4f}')
        snr_acc.append(ratio)

        if epoch % 10 == 0:
            # Confusion Matrix
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.show()

        tsne = TSNE(n_components=2, random_state=0)
        t_features = tsne.fit_transform(np.array(all_features))

        # Choose a different color for each category
        unique_labels = np.unique(all_labels)
        colors = plt.cm.get_cmap('tab20', len(unique_labels))  # Use the 'tab10' color map, or choose others such as 'set1', 'viridis', etc.

        # Create a scatter plot
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(t_features[:, 0], t_features[:, 1], c=all_labels, cmap=colors, s=1)

        # Add a color bar
        cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)))
        cbar.set_label('Class Labels')
        cbar.set_ticklabels(unique_labels)  # Ensure that the color bar labels match the categories

        plt.title('t-SNE visualization of features')
        plt.xlabel('t-SNE Feature 1')
        plt.ylabel('t-SNE Feature 2')
        plt.grid(True)  # Optional: Add grid lines to improve readability
        plt.show()

    return snr_acc


def validation_evaluation_metric(model, epoch, validation_loader, device, metric_fc=None, return_features=False,
                                 data_type='Validation'):
    model = model.to(device)
    model.eval()
    val_losses = []
    total = 0.0
    correct = 0.0
    all_labels = []
    all_preds = []
    all_features = []

    with torch.no_grad():
        for index, (inputs, labels) in enumerate(validation_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            if return_features:
                outputs, features = model(inputs, return_feature=True)
                all_features.extend(features.cpu().numpy())
            else:
                outputs = model(inputs)
            if metric_fc is None:
                pass
            else:
                cosine = F.linear(F.normalize(outputs), F.normalize(metric_fc.weight))  # Calculate cosine similarity
                # Get the predicted value
                pred = cosine.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.size(0)

            # loss = F.cross_entropy(outputs, labels.to(torch.int64))
            # val_losses.append(loss.item())
            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())

    ratio = correct / total
    print(f'{data_type} Accuracy: {ratio:.4f}')
    # print(f'{data_type} Loss: {np.average(val_losses):.4f}')

    if epoch % 10 == 0:
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    if return_features:
        tsne = TSNE(n_components=2, random_state=0)
        t_features = tsne.fit_transform(np.array(all_features))

        # Choose a different color for each category
        unique_labels = np.unique(all_labels)
        colors = plt.cm.get_cmap('tab20', len(unique_labels))  # Use the 'tab10' color map, or choose others such as 'set1', 'viridis', etc.

        # Create a scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(t_features[:, 0], t_features[:, 1], c=all_labels, cmap=colors, s=1)

        # Add a color bar
        cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)))
        cbar.set_label('Class Labels')
        cbar.set_ticklabels(unique_labels)  # Ensure that the color bar labels match the categories

        plt.title('t-SNE visualization of features')
        plt.xlabel('t-SNE Feature 1')
        plt.ylabel('t-SNE Feature 2')
        plt.grid(True)  # Optional: add grid lines to improve readability
        plt.show()

    return ratio, np.average(val_losses)


def tsne_visualization(model, validation_loader, device):
    model = model.to(device)
    model.eval()
    all_features = []

    with torch.no_grad():
        for (inputs, _) in validation_loader:
            inputs = inputs.to(device)

            # Suppose the model has been modified to only return features
            features = model(inputs)
            all_features.extend(features.cpu().numpy())

    # Use t-SNE to reduce the dimensionality of features
    tsne = TSNE(n_components=2, random_state=0)
    t_features = tsne.fit_transform(np.array(all_features))

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(t_features[:, 0], t_features[:, 1])

    plt.title('t-SNE Visualization of Features')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.grid(True)  # Optional: add grid lines to improve readability
    plt.show()


if __name__ == "__main__":
    print('hello')
