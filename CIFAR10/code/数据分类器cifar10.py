import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from models import Autoencoder2
from sklearn.cluster import DBSCAN

from torchvision import datasets, transforms
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle
from sklearn.metrics.pairwise import cosine_similarity
def select_eps(features_scaled, n_neighbors=5):
    # 计算每个点到其第n_neighbors个最近邻的距离
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors_fit = neighbors.fit(features_scaled)
    distances, indices = neighbors_fit.kneighbors(features_scaled)

    # 对距离进行排序
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    # 绘制KNN距离图
    plt.plot(distances)
    plt.xlabel('Points sorted by distance to the n-th nearest neighbor')
    plt.ylabel(f'Distance to {n_neighbors}-th nearest neighbor')
    plt.title('KNN Distance Plot to Help Choose eps')
    plt.show()
def calculate_centroids(features_scaled, clusters): #计算簇心，features_scaled是一个包含所有低维表示的NumPy数组，clusters是DBSCAN聚类结果，我们可以如下计算每个簇的质心：
    centroids = []
    for cluster_id in np.unique(clusters):
        if cluster_id != -1:  # 排除噪声点
            # 计算每个簇的质心
            centroid = np.mean(features_scaled[clusters == cluster_id], axis=0)
            centroids.append(centroid)
    return np.array(centroids)

def calculate_cosine_similarity(centroids): #计算簇心之间的余弦相似度，similarity_matrix是一个矩阵，其中similarity_matrix[i, j]表示第i个簇和第j个簇之间的余弦相似度。
    # 计算所有质心之间的余弦相似度
    similarity_matrix = cosine_similarity(centroids)
    return similarity_matrix

def Data_classifier(Encoder,data,real_c):
    features=Encoder(data)
    #features=F.normalize(features, p=2, dim=-1)
    # 将特征数据从GPU内存复制到CPU内存
    #features_cpu = features.detach().cpu().numpy()
    features_scaled = features.detach().cpu().numpy()
    features_scaled = features_scaled.reshape(features_scaled.shape[0], -1)
    #features_scaled = StandardScaler().fit_transform(features_cpu)

    with open('/workspace/algorithm/Minist-c/Encoder_model/scaled_feature.pkl', 'wb') as f:
        pickle.dump(features_scaled, f)
    #select_eps(features_scaled)
    # 使用DBSCAN聚类算法
    n_clusters_=0
    e=7.001
    bb=0
    while n_clusters_!=real_c:

        dbscan = DBSCAN(eps=e, min_samples=1)
        clusters = dbscan.fit_predict(features_scaled)

        # 打印簇的数量（不包括噪声点）
        n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)
        if n_clusters_==real_c or bb>20:
            break
        elif n_clusters_>real_c:
            e+=0.01
            bb+=0.01
        else:
            e-=0.01

    print(e)
    print(f'Estimated number of clusters: {n_clusters_}')

    # 查看噪声点的比例
    n_noise_ = list(clusters).count(-1)
    print(f'Estimated number of noise points: {n_noise_}')

    centroids = calculate_centroids(features_scaled, clusters)
    similarity_matrix = calculate_cosine_similarity(centroids)

    print(similarity_matrix)
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam:Generator learning rate")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--model_save_path", type=str, default='/workspace/algorithm/code/models/CIFAR10-Encoder.pth', help="Generator model save path")
    parser.add_argument("--dataset_root", type=str, default='/workspace/dataset/private/wzg_dataset/CIFAR10', help="Path to dataset")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10(root=opt.dataset_root, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

    model = Autoencoder2().to(device)

    model.load_state_dict(torch.load(opt.model_save_path))
    print("模型加载成功！")
    model.eval()

    CC=[]
    for images, labels in train_loader:
        print(labels)
        unique_values = torch.unique(labels)
        real_classes = len(unique_values)
        c=Data_classifier(model.encoder,images.to(device),real_classes)
        CC.append(c)
        print(c)
    print("e的均值为:"+str(sum(CC) / len(CC)))
if __name__=='__main__':
    main()