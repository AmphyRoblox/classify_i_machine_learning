import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from models.mobilenet import mobilenet
from ads_models.resnet import resnet18
from scipy.io import loadmat
import h5py


# Data loading function
def TrainDataset(num):
    X = np.load(f"F:/FS-SEI_4800/Dataset/X_train_{num}Class.npy")
    Y = np.load(f"F:/FS-SEI_4800/Dataset/Y_train_{num}Class.npy").astype(np.uint8)
    return train_test_split(X, Y, test_size=0.2, random_state=30)


def TestDataset(num):
    X = np.load(f"Dataset/X_test_{num}Class.npy")
    Y = np.load(f"Dataset/Y_test_{num}Class.npy").astype(np.uint8)
    return X, Y


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

        # Define the convolutional layer
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.pool4 = nn.MaxPool1d(2)

        self.conv5 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(64)
        self.pool5 = nn.MaxPool1d(2)

        self.conv6 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(64)
        self.pool6 = nn.MaxPool1d(2)

        self.conv7 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm1d(64)
        self.pool7 = nn.MaxPool1d(2)

        self.conv8 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm1d(64)
        self.pool8 = nn.MaxPool1d(2)

        self.conv9 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm1d(64)
        self.pool9 = nn.MaxPool1d(2)

        # Define a fully connected layer
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1152, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1152, 90)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x, train_mode=True):
        # If the input is (batch_size, length, channels)，Need to be adjusted to (batch_size, channels, length)
        # x = x.permute(0, 2, 1)

        # Pass through the convolutional blocks layer by layer
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        x = torch.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)

        x = torch.relu(self.bn6(self.conv6(x)))
        x = self.pool6(x)

        x = torch.relu(self.bn7(self.conv7(x)))
        x = self.pool7(x)

        x = torch.relu(self.bn8(self.conv8(x)))
        x = self.pool8(x)

        x = torch.relu(self.bn9(self.conv9(x)))
        # x = self.pool9(x)
        # Fully connected layer
        x = self.flatten(x)
        # x = self.fc1(x)
        embedding = x
        x = self.relu(x)
        x = self.fc2(x)
        if train_mode:
            return x
        else:
            return embedding, x


class NETCONV(nn.Module):
    def __init__(self):
        super(NETCONV, self).__init__()

        # Define the convolutional layer
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(2)

        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.pool5 = nn.MaxPool1d(2)

        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(256)
        self.pool6 = nn.MaxPool1d(2)

        self.conv7 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm1d(512)
        self.pool7 = nn.MaxPool1d(2)

        self.conv8 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm1d(512)
        self.pool8 = nn.MaxPool1d(2)

        self.conv9 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm1d(1024)
        self.pool9 = nn.MaxPool1d(2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.fc1 = nn.Linear(1152, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 90)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x, train_mode=True):
        # If the input is (batch_size, length, channels)，Need to be adjusted to (batch_size, channels, length)
        # x = x.permute(0, 2, 1)

        # Pass through the convolutional blocks layer by layer
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        x = torch.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)

        x = torch.relu(self.bn6(self.conv6(x)))
        x = self.pool6(x)

        x = torch.relu(self.bn7(self.conv7(x)))
        x = self.pool7(x)

        x = torch.relu(self.bn8(self.conv8(x)))
        x = self.pool8(x)

        x = torch.relu(self.bn9(self.conv9(x)))
        # x = self.pool9(x)
        # Fully connected layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc1(x)
        embedding = x
        x = self.relu(x)
        x = self.fc2(x)
        if train_mode:
            return x
        else:
            return embedding, x


# Accuracy calculation function
def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def get_num_class_pretraindata():
    x = np.load(f"F:/FS-SEI_4800/Dataset/X_train_90Class.npy")
    y = np.load(f"F:/FS-SEI_4800/Dataset/Y_train_90Class.npy")
    x = x.transpose(0, 2, 1)
    return x, y
    # train_index_shot = []
    # for i in range(90):
    #     index_classi = [index for index, value in enumerate(y) if value == i]
    #     train_index_shot += index_classi[0:100]
    # return x[train_index_shot], y[train_index_shot]


# def PreTrainDataset_prepared(class_index, sample_num=100):
#     X_train_ul, Y_train_ul = get_num_class_pretraindata()
#     train_index_shot = []
#     for i in class_index:
#         index_classi = [index for index, value in enumerate(Y_train_ul) if value == i]
#         train_index_shot += index_classi[0:sample_num]
#     X_train_shot = X_train_ul[train_index_shot]
#     Y_train_shot = Y_train_ul[train_index_shot].astype(np.uint8)
#     return train_test_split(X_train_shot, Y_train_shot, test_size=0.2, random_state=30)
def PreTrainDataset_prepared(class_index, sample_num=100):
    # X_train_ul = np.load('F:/ADS-B/signal.npy')
    with h5py.File('F:/ADS-B/signal_data.mat', 'r') as f:
        signal_data_dataset = f['signalData']  # Return HDF5 Dataset type
        signal_data = np.array(signal_data_dataset)  # Recommend method
        print(f"Signal data shape: {signal_data.shape}")
    X_train_ul = signal_data.transpose()
    Y_train_ul = loadmat('F:/ADS-B/labels.mat')['labels'].squeeze()
    train_index_shot = []
    for i in class_index:
        index_classi = [index for index, value in enumerate(Y_train_ul) if value == i]
        train_index_shot += index_classi[0:sample_num]
    X_train_shot = X_train_ul[train_index_shot]
    Y_train_shot = Y_train_ul[train_index_shot].astype(np.uint8)
    return train_test_split(X_train_shot, Y_train_shot, test_size=0.2, random_state=30)


# Training Code
if __name__ == "__main__":
    num_classes = 90
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_index = np.arange(num_classes)
    # Loading Data
    # X_train, X_val, Y_train, Y_val = TrainDataset(num_classes)
    X_train, X_val, Y_train, Y_val = PreTrainDataset_prepared(class_index, sample_num=100)
    min_value = min(X_train.min(), X_val.min())
    max_value = max(X_train.max(), X_val.max())
    X_train = (X_train - min_value) / (max_value - min_value)
    X_val = (X_val - min_value) / (max_value - min_value)

    Y_train = torch.tensor(Y_train, dtype=torch.long)
    Y_val = torch.tensor(Y_val, dtype=torch.long)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), Y_train)
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), Y_val)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Model initialization
    model = NETCONV().to(device)
    # model = mobilenet(class_num=90).to(device)
    # model = resnet18(num_classes=60).to(device)
    # model = res2netgc_resnet10(num_classes=90).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Phase
    best_val_loss = float('inf')
    for epoch in range(200):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        # Tranining Steps
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X, train_mode=True)  # Adjust the entire dimension to adapt Conv1d
            # outputs = model(batch_X, train_mode=True)  # Adjust the entire dimension to adapt Conv1d
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == batch_Y).sum().item()
            total_train += batch_Y.size(0)

        train_accuracy = train_correct / total_train

        # Test Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                # outputs, _ = model(batch_X)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)

                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == batch_Y).sum().item()
                total_val += batch_Y.size(0)

        val_accuracy = val_correct / total_val

        print(f"Epoch {epoch + 1}/200, "
              f"Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, "
              f"Val Acc: {val_accuracy:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoint/CVCNN.pth")
            print("Model saved.")
