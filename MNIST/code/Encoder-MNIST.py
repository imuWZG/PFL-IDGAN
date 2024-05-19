import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)  # 输入通道数为3，输出通道数为16
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.convT1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.convT2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.convT3 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = F.relu(self.bn1(self.convT1(x)))
        x = F.relu(self.bn2(self.convT2(x)))
        x = torch.sigmoid(self.convT3(x))  # 使用sigmoid激活函数以匹配图像的值范围
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x 
def model_loading(model,file_path, backup_path):#模型加载
    try:
        # 尝试从主路径加载模型
        model.load_state_dict(torch.load(file_path))
        #loaded_model = load_model(file_path, model)  # 你可能需要根据你的框架调整这个调用
        print("模型成功从主路径加载。")
    except Exception as e:
        print(f"从主路径加载模型失败，原因：{e}。尝试备用路径。")
        try:
            # 尝试从备用路径加载模型
            model.load_state_dict(torch.load(backup_path))
            #loaded_model = load_model(backup_path, model)  # 同样需要调整
            print("模型成功从备用路径加载。")
        except Exception as e:
            # 如果两个路径都失败了，处理失败情况
            print(f"从主路径和备用路径加载模型都失败了，原因：{e}")

    return model
def train(model, train_loader, val_loader, epochs, optimizer, device, opt):
    criterion = nn.MSELoss()
    model.train()

    # 早停参数设置
    patience = 10  # 验证损失未改善的epoch数量
    min_val_loss = np.Inf
    patience_counter = 0

    for epoch in range(epochs):
        model.train()  # 确保模型处于训练模式
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()

        # 验证过程
        model.eval()  # 确保模型处于评估模式
        val_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                outputs = model(data)
                batch_loss = criterion(outputs, data)
                val_loss += batch_loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}, Loss: {loss.item()}, Val Loss: {val_loss}")

        # 早停逻辑
        if val_loss < min_val_loss:
            print("Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...".format(min_val_loss, val_loss))
            torch.save(model.state_dict(), opt.model_save_path + 'Encoder3.pth')
            min_val_loss = val_loss
            patience_counter = 0  # 重置patience计数器
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                return  # 停止训练

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam:Generator learning rate")
    parser.add_argument("--model_save_path", type=str, default='/workspace/algorithm/code/models/MNIST-Encoder', help="Generator model save path")
    parser.add_argument("--dataset_root", type=str, default='/workspace/dataset/private/wzg_dataset/MNIST/', help="Path to dataset")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(32),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root=opt.dataset_root, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

    # 加载验证集
    val_dataset = datasets.MNIST(root=opt.dataset_root, train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

    model = Autoencoder().to(device)
    model=model_loading(model,opt.model_save_path,opt.model_save_path)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # 调用训练函数时添加验证加载器
    train(model, train_loader, val_loader, opt.n_epochs, optimizer, device, opt)

if __name__=='__main__':
    main()