import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import os

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention
    
class Generator(nn.Module):
    def __init__(self, z_dim, channel, img_size, n_classes):  # z_dim（噪声向量的维度）, n_classes为类别数
        super(Generator, self).__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(z_dim + n_classes, 128 * self.init_size ** 2))  # 直接将one-hot编码的类别加到输入z_dim上
        # 初始化自注意力层
        self.self_attention = SelfAttention(64)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_conv = nn.Conv2d(64, channel, 3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z, labels):
        z = torch.cat([z, labels], 1)  # 这里labels应为one-hot编码形式
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        out = self.conv_blocks(out)
        # 应用自注意力层
        out, _ = self.self_attention(out)  # 只获取需要的输出
        out = self.final_conv(out)
        img = self.tanh(out)
        return img
class Generator2(nn.Module):
    def __init__(self, z_dim, channel, img_size, n_classes):  # z_dim（噪声向量的维度）, n_classes为类别数
        super(Generator2, self).__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(z_dim + n_classes, 128 * self.init_size ** 2))  # 直接将one-hot编码的类别加到输入z_dim上
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_conv = nn.Conv2d(64, channel, 3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z, labels):
        z = torch.cat([z, labels], 1)  # 这里labels应为one-hot编码形式
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        out = self.conv_blocks(out)
        out = self.final_conv(out)
        img = self.tanh(out)
        return img
class FeatureExtractor(nn.Module):
    """共享的特征提取部分"""
    def __init__(self, channel):
        super(FeatureExtractor, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channel, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    """鉴别器的顶层"""
    def __init__(self, feature_extractor, img_size, n_classes,channel):
        super(Discriminator, self).__init__()
        self.feature_extractor = feature_extractor
        # 计算卷积后的图像大小
        self.conv_output_size = 128 * (img_size // 4) * (img_size // 4)
        self.label_embedding = nn.Embedding(n_classes, self.conv_output_size)
        self.discriminator = nn.Sequential(
            nn.Linear(self.conv_output_size *(n_classes + 1) , 256),  # 注意这里的修改
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels_onehot):
        #labels = self.label_embedding(labels_onehot)
        labels= self.label_embedding(labels_onehot).view(labels_onehot.size(0),  -1)
        features = self.feature_extractor(x)
        # 确保拼接后的向量d_in传递给鉴别器
        d_in = torch.cat((features, labels), dim=1)
        return self.discriminator(d_in)

class Classifier(nn.Module):
    """分类器的顶层"""
    def __init__(self, feature_extractor, num_classes,img_size):
        super(Classifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = nn.Sequential(
            nn.Linear(128 * (img_size // 4) * (img_size // 4), 256),  # 修改为三层全连接层
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        output=self.classifier(features)
        #result = torch.unsqueeze(torch.argmax(output, dim=1),dim=1)
        return output
    
class LocalDataset(Dataset):
    def __init__(self):
        self.data = []
        self.labels = []

    def append(self, data, label):
        self.data.append(data)
        self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)  # 输入通道数为3，输出通道数为16
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

class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.convT1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.convT2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.convT3 = nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = F.relu(self.bn1(self.convT1(x)))
        x = F.relu(self.bn2(self.convT2(x)))
        x = torch.sigmoid(self.convT3(x))  # 使用sigmoid激活函数以匹配图像的值范围
        return x

class Autoencoder2(nn.Module):
    def __init__(self):
        super(Autoencoder2, self).__init__()
        self.encoder = Encoder2()
        self.decoder = Decoder2()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x 
