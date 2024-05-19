import torch
import torch
import copy
import os
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Subset
from scipy.linalg import sqrtm
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import DBSCAN
import torch.nn as nn
import models
import torch.optim as optim
import math


random.seed(4) 
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

def save_model_checkpoint(model, file_path, backup_path): #模型保存
    '''
    # 检查当前模型文件是否存在，如果存在，则移动到备份位置
    if os.path.exists(file_path):
        # 如果备份文件已存在，先删除
        if os.path.exists(backup_path):
            os.remove(backup_path)
        os.rename(file_path, backup_path)
        print(f"旧模型参数已备份到 {backup_path}")
    '''
    # 保存当前模型参数
    torch.save(model.state_dict(), file_path)
    print(f"模型参数已保存到 {file_path}")

def generate_proportions(num_clients, num_classes):
    proportions = []
    for _ in range(num_clients):
        # 初始化类别比例为零
        client_proportions = {class_idx: 0 for class_idx in range(num_classes)}
        
        # 计算标签为0的类别数量
        zero_labels_count = random.randint(int(num_classes * 0.3), int(num_classes * 0.4))
        
        # 确定有数据的类别数
        nonzero_classes_count = num_classes - zero_labels_count
        
        # 选择非零类别
        nonzero_classes = random.sample(range(num_classes), nonzero_classes_count)
        
        # 分配随机比例
        remaining = 1.0
        for class_idx in nonzero_classes[:-1]:
            # 保留足够的比例给剩余的非零类别，每个至少0.02
            max_available = remaining - 0.01 * (len(nonzero_classes) - nonzero_classes.index(class_idx) - 1)
            if max_available > 0.01:
                proportion = random.uniform(0.01, max_available)
                client_proportions[class_idx] = proportion
                remaining -= proportion
            else:
                # 如果不可能分配至少0.08给每个剩余类别，分配所有剩余给当前类别
                client_proportions[class_idx] = remaining
                remaining = 0
                break
        
        # 最后一个选中的类别分配剩余的比例
        if remaining > 0:
            client_proportions[nonzero_classes[-1]] = remaining
        proportions.append(client_proportions)
    
    return proportions

def generate_proportions2(num_clients, num_classes, num_common_classes):
    if num_common_classes > num_classes:
        raise ValueError("The number of common classes cannot be greater than the total number of classes.")
    
    # 确定公共类别
    common_classes = random.sample(range(num_classes), num_common_classes)

    # 初始化所有客户端的比例列表
    proportions = []
    
    for _ in range(num_clients):
        client_proportions = {class_idx: 0 for class_idx in range(num_classes)}
        
        # 随机选择有数据的类别，包括所有公共类别
        selected_classes = common_classes + random.sample(
            [cls for cls in range(num_classes) if cls not in common_classes],
            random.randint(0, num_classes - num_common_classes)
        )
        
        # 分配随机比例
        remaining = 1.0
        random.shuffle(selected_classes)
        for class_idx in selected_classes[:-1]:
            max_available = remaining - 0.01 * (len(selected_classes) - selected_classes.index(class_idx) - 1)
            if max_available > 0.01:
                proportion = random.uniform(0.01, max_available)
                client_proportions[class_idx] = proportion
                remaining -= proportion
            else:
                client_proportions[class_idx] = remaining
                remaining = 0
                break

        # 最后一个选中的类别分配剩余的比例
        if remaining > 0:
            client_proportions[selected_classes[-1]] = remaining
        
        proportions.append(client_proportions)
    
    return proportions

def generate_proportions3(num_clients, num_classes, num_common_classes):
    if num_common_classes > num_classes:
        raise ValueError("The number of common classes cannot be greater than the total number of classes.")
    
    # 确定公共类别
    common_classes = random.sample(range(num_classes), num_common_classes)

    # 初始化所有客户端的比例列表
    proportions = []
    
    for _ in range(num_clients):
        client_proportions = {class_idx: 0 for class_idx in range(num_classes)}
        
        # 确定标签样本为0的类别数量
        zero_labels_count = random.randint(int(num_classes * 0.4), int(num_classes * 0.5))
        
        # 随机选择标签样本为0的类别
        zero_label_classes = random.sample([cls for cls in range(num_classes) if cls not in common_classes], zero_labels_count)
        
        # 确定有数据的类别，包括所有公共类别
        selected_classes = [cls for cls in range(num_classes) if cls not in zero_label_classes]
        
        # 分配随机比例
        remaining = 1.0
        random.shuffle(selected_classes)
        for class_idx in selected_classes[:-1]:
            max_available = remaining - 0.01 * (len(selected_classes) - selected_classes.index(class_idx) - 1)
            if max_available > 0.01:
                proportion = random.uniform(0.01, max_available)
                client_proportions[class_idx] = proportion
                remaining -= proportion
            else:
                client_proportions[class_idx] = remaining
                remaining = 0
                break

        # 最后一个选中的类别分配剩余的比例
        if remaining > 0:
            client_proportions[selected_classes[-1]] = remaining
        
        proportions.append(client_proportions)
    
    return proportions,common_classes



def generate_iid_proportions(num_clients, num_classes):
    proportions = []
    # 生成一个全局统一的类别比例
    total_proportion = 1.0
    class_proportions = []
    remaining = total_proportion

    for class_idx in range(num_classes - 1):
        # 为每个类别分配一个随机的比例，确保每个类别至少有一个最小比例，这里我们也设置为0.02
        max_available = remaining - 0.02 * (num_classes - class_idx - 1)
        if max_available > 0.02:
            proportion = random.uniform(0.02, max_available)
            class_proportions.append(proportion)
            remaining -= proportion
        else:
            class_proportions.append(remaining)
            remaining = 0
            break

    # 确保最后一个类别分配所有剩余比例
    if remaining > 0:
        class_proportions.append(remaining)
    else:
        class_proportions.append(0.02)  # 至少保留最小值

    # 每个客户端获得相同的类别比例
    for _ in range(num_clients):
        client_proportions = {class_idx: prop for class_idx, prop in enumerate(class_proportions)}
        proportions.append(client_proportions)

    return proportions

def load_and_process_dataset(dataset_name, dataset_root, num_clients,args):

    # 数据集的类别数
    num_classes = 10 if 'CIFAR' not in dataset_name else 100 if dataset_name == 'CIFAR100' else 10
    
    # 根据数据集名称设置图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * (1 if 'MNIST' in dataset_name else 3), (0.5,) * (1 if 'MNIST' in dataset_name else 3))
    ])
    
    # 动态加载数据集
    dataset_class = {
        'MNIST': datasets.MNIST,
        'FashionMNIST': datasets.FashionMNIST,
        'CIFAR10': datasets.CIFAR10,
        'CIFAR100': datasets.CIFAR100
    }[dataset_name]
    
    dataset = dataset_class(root=dataset_root, train=True, transform=transform, download=True)
    targets = np.array(dataset.targets)
    indices_per_class = {class_idx: np.where(targets == class_idx)[0] for class_idx in range(num_classes)}

    # 生成自定义客户端的数据比例
    proportions = generate_proportions(num_clients, num_classes)
    for pro in proportions:
        print(pro)
        print('---------------------------------------------------------------')
    testlables = []
    # Iterate through each dictionary in proportions
    for prop in proportions:
        # Use a list comprehension to get labels (keys) where the value is non-zero
        non_zero_labels = [label for label, proportion in prop.items() if proportion != 0]
        # Append the list of non-zero labels to testlables
        testlables.append(non_zero_labels)
    # 生成非独立同分布的数据子集
    subsets = []
    subset_targets = []
    for proportion in proportions:
        indices = np.hstack([np.random.choice(indices_per_class[class_idx], int(len(indices_per_class[class_idx]) * proportion[class_idx]), replace=False)
                            for class_idx in proportion if class_idx in indices_per_class and proportion[class_idx] > 0])
        subset = Subset(dataset, indices)
        subsets.append(subset)
        subset_targets.append(targets[indices])

    intersection_label= common_labels(testlables)
    Public_Set=generate_Public_set(intersection_label,'MNIST',args.dataset_root,args)

    return subsets, testlables,Public_Set

def load_and_process_dataset2(dataset_name, dataset_root, num_clients, samples_per_client, args):
    # 数据集的类别数
    num_classes = args.num_classes
    
    # 根据数据集名称设置图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * (1 if 'MNIST'or'FashionMNIST' in dataset_name else 3), (0.5,) * (1 if 'MNIST'or'FashionMNIST' in dataset_name else 3))
    ])
    
    # 动态加载数据集
    dataset_class = {
        'MNIST': datasets.MNIST,
        'FashionMNIST': datasets.FashionMNIST,
        'CIFAR10': datasets.CIFAR10,
        'CIFAR100': datasets.CIFAR100
    }[dataset_name]
    
    dataset = dataset_class(root=dataset_root, train=True, transform=transform, download=True)
    targets = np.array(dataset.targets)
    indices_per_class = {class_idx: np.where(targets == class_idx)[0] for class_idx in range(num_classes)}

    # 生成自定义客户端的数据比例
    num_common_classes=math.ceil(num_classes*0.1)
    proportions,common_classes = generate_proportions3(num_clients, num_classes,num_common_classes)

    for pro in proportions:
        print(pro)
        print('---------------------------------------------------------------')

    # 收集客户端的标签集合
    testlables = []
    for prop in proportions:
        non_zero_labels = [label for label, proportion in prop.items() if proportion != 0]
        testlables.append(non_zero_labels)

    # 生成非独立同分布的数据子集
    subsets = []
    subset_targets = []
    for proportion in proportions:
        # 调整比例以确保总样本数等于samples_per_client
        total_samples = sum([int(len(indices_per_class[class_idx]) * proportion[class_idx]) for class_idx in proportion if class_idx in indices_per_class])
        adjustment_factor = samples_per_client / total_samples if total_samples > 0 else 0
        
        indices = np.hstack([
            np.random.choice(indices_per_class[class_idx], int((len(indices_per_class[class_idx]) * proportion[class_idx]) * adjustment_factor), replace=True)
            for class_idx in proportion if class_idx in indices_per_class and proportion[class_idx] > 0
        ])
        subset = Subset(dataset, indices)
        subsets.append(subset)
        subset_targets.append(targets[indices])
    
    # 生成公共数据集
    #intersection_labels = set(testlables[0]).intersection(*testlables[1:])
    Public_Set = generate_Public_set2(common_classes, dataset_name, dataset_root, args,args.p_num)

    return subsets, testlables, Public_Set


def generate_testset(test_lable,dataset_name,dataset_root,args):
    # 根据数据集名称设置图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * (1 if 'MNIST' in dataset_name else 3), (0.5,) * (1 if 'MNIST' in dataset_name else 3))
    ])
    
    # 动态加载数据集
    dataset_class = {
        'MNIST': datasets.MNIST,
        'FashionMNIST': datasets.FashionMNIST,
        'CIFAR10': datasets.CIFAR10,
        'CIFAR100': datasets.CIFAR100
    }[dataset_name]
    
    testdataset = dataset_class(root=dataset_root, train=False, transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=True)

    custom_dataset = models.LocalDataset()
    # 创建一个空字典
    sample_dict = {}
    for images, labels in testloader:
        for image, label in zip(images, labels):
            label = label.unsqueeze(0)
            if label.item() in test_lable:
                if label.item() in sample_dict:
                    if sample_dict[label.item()] <args.testset_sample_num:
                        sample_dict[label.item()] += 1
                        custom_dataset.append(image, label)
                else:
                    sample_dict[label.item()] = 1
                    custom_dataset.append(image, label)
    data_loader = DataLoader(custom_dataset, batch_size=args.batch_size, shuffle=True)
    return data_loader

def generate_testset2(test_label, dataset_name, dataset_root, batch_size, x):
    # 根据数据集名称设置图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * (1 if 'MNIST' in dataset_name else 3), (0.5,) * (1 if 'MNIST' in dataset_name else 3))
    ])
    
    # 动态加载数据集
    dataset_class = {
        'MNIST': datasets.MNIST,
        'FashionMNIST': datasets.FashionMNIST,
        'CIFAR10': datasets.CIFAR10,
        'CIFAR100': datasets.CIFAR100
    }[dataset_name]
    
    testdataset = dataset_class(root=dataset_root, train=True, transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size, shuffle=True)

    custom_dataset = models.LocalDataset()
    # 创建一个空字典
    sample_dict = {label: 0 for label in range(10)}

    for images, labels in testloader:
        for image, label in zip(images, labels):
            if label.item() in sample_dict and sample_dict[label.item()] < x:
                sample_dict[label.item()] += 1
                custom_dataset.append(image, label)
                # 检查是否所有标签都已经收集到足够的样本
                if all(count >= x for count in sample_dict.values()):
                    break
        if all(count >= x for count in sample_dict.values()):
            break

    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def generate_Public_set(test_lable,dataset_name,dataset_root,args):
    # 根据数据集名称设置图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * (1 if 'MNIST' in dataset_name else 3), (0.5,) * (1 if 'MNIST' in dataset_name else 3))
    ])
    
    # 动态加载数据集
    dataset_class = {
        'MNIST': datasets.MNIST,
        'FashionMNIST': datasets.FashionMNIST,
        'CIFAR10': datasets.CIFAR10,
        'CIFAR100': datasets.CIFAR100
    }[dataset_name]
    
    testdataset = dataset_class(root=dataset_root, train=True, transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=True)

    custom_dataset = models.LocalDataset()
    # 创建一个空字典
    sample_dict = {}
    for images, labels in testloader:
        for image, label in zip(images, labels):
            label = label.unsqueeze(0)
            if label.item() in test_lable:
                if label.item() in sample_dict:
                    if sample_dict[label.item()] <args.testset_sample_num:
                        sample_dict[label.item()] += 1
                        custom_dataset.append(image, label)
                else:
                    sample_dict[label.item()] = 1
                    custom_dataset.append(image, label)
    data_loader = DataLoader(custom_dataset, batch_size=args.batch_size, shuffle=True)
    return data_loader

def generate_Public_set2(test_lable, dataset_name, dataset_root, args, P_num):
    # 根据数据集名称设置图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * (1 if 'MNIST' in dataset_name else 3), (0.5,) * (1 if 'MNIST' in dataset_name else 3))
    ])
    
    # 动态加载数据集
    dataset_class = {
        'MNIST': datasets.MNIST,
        'FashionMNIST': datasets.FashionMNIST,
        'CIFAR10': datasets.CIFAR10,
        'CIFAR100': datasets.CIFAR100
    }[dataset_name]
    
    testdataset = dataset_class(root=dataset_root, train=True, transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=True)

    custom_dataset = models.LocalDataset()
    # 创建一个空字典用于跟踪每个标签的样本数量
    sample_dict = {}
    for images, labels in testloader:
        for image, label in zip(images, labels):
            label = label.unsqueeze(0)
            label_item = label.item()
            if label_item in test_lable:
                if label_item not in sample_dict:
                    sample_dict[label_item] = 0
                if sample_dict[label_item] < P_num:
                    sample_dict[label_item] += 1
                    custom_dataset.append(image, label)

    data_loader = DataLoader(custom_dataset, batch_size=args.batch_size, shuffle=True)
    return data_loader

def collect_unique_labels(labels):
    unique_labels = set()  # 创建一个空集合，用于存储唯一的标签
    for row in labels:
        unique_labels.update(row)  # 将每一行的元素添加到集合中，自动去重
    return list(unique_labels)  # 将集合转换成列表并返回

def create_custom_cnn_2(dataset_name,algorithm_name):
    # 根据数据集确定输入通道数和类别数
    if dataset_name in ['MNIST', 'FashionMNIST']:
        input_channels = 1  # MNIST 和 Fashion-MNIST 是灰度图像
        num_classes = 10
        img_size=28
    elif dataset_name in ['CIFAR10']:
        input_channels = 3  # CIFAR-10 是彩色图像
        num_classes = 10
        img_size=32
    elif dataset_name in ['CIFAR100']:
        input_channels = 3  # CIFAR-100 也是彩色图像
        num_classes = 100
        img_size=32
    # 根据算法名称选择网络结构
    if algorithm_name in ['FedAvg', 'FedProx']:
        num_layers = 2
        #filters = [32, 64]  # 固定的滤波器配置
    else:
        num_layers = random.choice([2, 3])
        #filters = sorted(random.choices([20, 24, 32, 40, 48, 56, 80, 96], k=num_layers))
    
    
    k = 64
    layers = []
    layer_in_channels = input_channels
    current_size = img_size

    for i in range(num_layers):
        layer_out_channels = (i + 1) * k
        layers.append(nn.Conv2d(layer_in_channels, layer_out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(layer_out_channels))
        layers.append(nn.ReLU())
        layer_in_channels = layer_out_channels

    # 添加展平层前计算特征图的尺寸
    layers.append(nn.Flatten())

    # 计算卷积层输出的维度，假设没有尺寸变化
    fc_input_features = layer_out_channels * current_size * current_size
    layers.append(nn.Linear(fc_input_features, 256))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(256, 128))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(128, num_classes))
    layers.append(nn.Softmax(dim=1))

    model = nn.Sequential(*layers)
    return model

def calculate_average(lst):
    total = sum(lst)
    length = len(lst)
    average = total / length
    return average

def train(model, device, train_loader,args):
    optimizer = optim.Adam(model.parameters(), args.c_lr)
    model.train()
    model.to(device)
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        images=images.to(device)
        outputs=model(images).float()
        #labels=torch.squeeze(labels,dim=0).to(device)
        loss = F.nll_loss(outputs, labels)

        loss.backward()
        optimizer.step()

    #print("[loss: %f]" % (loss.item()))
    return model.state_dict()       

def test(model,  test_loader,args):
    model.eval()
    # 测试模型的精度
    correct = 0
    total = 0
    for images, labels in test_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            labels=labels.view(-1)
            r=(predicted == labels).sum().item()
            correct += r
    accuracy = 100 * correct / total
    return accuracy

def average_model_weights(global_model, client_models):
    global_state_dict = global_model.state_dict()
    avg_state_dict = copy.deepcopy(global_state_dict)
    for key in avg_state_dict.keys():
        sum_weight = torch.zeros_like(avg_state_dict[key])
        for client_model in client_models:
            client_state_dict = client_model.state_dict()
            sum_weight += client_state_dict[key]
        avg_state_dict[key] = sum_weight / len(client_models)
    global_model.load_state_dict(avg_state_dict)
    return global_model

def fed_prox(models, global_model, mu=0.02):
    """实现 FedProx，添加正则化"""
    prox_model = copy.deepcopy(models[0].state_dict())  # 使用深拷贝以避免修改原始模型
    for key in prox_model.keys():
        for i in range(1, len(models)):
            prox_model[key] += models[i].state_dict()[key]
        # 在 FedProx 中，我们在这里添加了 L2 正则化项
        prox_model[key] = (prox_model[key] + mu * global_model[key]) / (len(models) + mu)
    return prox_model

def fedmd_step(client_models, public_dataloader, args):
    # 将所有模型切换到训练模式
    for model in client_models:
        model.train()

    # 优化器列表，假设每个模型都用相同的学习率
    optimizers = [torch.optim.Adam(model.parameters(), args.c_lr) for model in client_models]

    # 迭代公共数据集
    for data, target in public_dataloader:
        data, target = data.to(args.device), target.to(args.device)
        
        # 存储所有模型的输出
        outputs = []

        # 首次获取所有客户端模型的输出
        for model in client_models:
            output = model(data)
            outputs.append(output)
        
        # 计算软输出和平均软输出
        #soft_outputs = [F.softmax(output / 1.0, dim=1) for output in outputs]
        avg_soft_outputs = torch.stack(outputs).mean(dim=0)

        # 对每个模型进行一次优化
        for model, optimizer, output in zip(client_models, optimizers, outputs):
            optimizer.zero_grad()
            #soft_output = F.softmax(output / 1.0, dim=1)
            soft_output=output
            loss = F.kl_div(soft_output.log(), avg_soft_outputs.detach(), reduction='batchmean')
            #loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
    return client_models

def common_labels(labels):
    if not labels:
        return []  # 如果输入列表为空，返回空列表
    
    # 将第一个列表转换为集合
    common_set = set(labels[0])
    
    # 遍历剩余的列表，用交集更新公共元素集合
    for label_list in labels[1:]:
        common_set.intersection_update(label_list)
        
    # 将结果集合转换回列表形式
    return list(common_set)

def generator_local_train(local_generator,local_discriminator,trainloader,pid,args,algorithm_name):#生成器本地迭代

    optimizer_G=torch.optim.Adam(local_generator.parameters(), lr=args.g_lr)
    optimizer_D=torch.optim.Adam(local_discriminator.parameters(), lr=args.d_lr)
    local_generator.train()
    local_discriminator.train()

    adversarial_loss = torch.nn.BCELoss()
    for epoch in range(args.gen_pro_epochs):
        # 使用数据加载器

        for images, labels in trainloader:

            labels_one_hot=F.one_hot(labels, num_classes=args.num_classes).to(args.device)

            # 生成分类随机整数
            unique_elements = torch.unique(labels)
            y = np.random.choice(unique_elements,size=args.batch_size)
            y = torch.tensor(y)
            # 转换为one-hot编码
            y_one_hot = F.one_hot(y, num_classes=args.num_classes).to(args.device)

            real_imgs = images
            real_imgs=real_imgs.to(args.device)
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            z = torch.from_numpy(np.random.normal(0, 1, (args.batch_size, args.z_dim))).float().to(args.device)
            gen_imgs = local_generator(z,y_one_hot)
            # Generate a batch of images
            g_output=local_discriminator(gen_imgs,y_one_hot)
            g_label = torch.ones_like(g_output) 
            g_loss = adversarial_loss(g_output, g_label)

            g_loss.backward()
            optimizer_G.step()

           # 判别器的损失
            optimizer_D.zero_grad()

            # 对于真实数据，判别器应该输出接近1的值
            d_output_real = local_discriminator(real_imgs, labels_one_hot)  # 将真实数据x和条件y输入判别器
            d_label_real = torch.ones_like(d_output_real)  # 真实数据的标签是1
            d_loss_real = adversarial_loss(d_output_real, d_label_real)

            # 对于生成的数据，判别器应该输出接近0的值
            d_output_fake = local_discriminator(gen_imgs.detach(), y_one_hot)  # 使用.detach()防止梯度传回生成器
            d_label_fake = torch.zeros_like(d_output_fake)  # 生成数据的标签是0
            d_loss_fake = adversarial_loss(d_output_fake, d_label_fake)

            # 判别器的总损失
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
        
        
        if epoch%10==0:
            save_image(gen_imgs.data[:args.batch_size], args.img_save_path+algorithm_name+"/%d/%d_客户端.png" % (int(pid),int(epoch)), nrow=int(pow(args.batch_size,0.5)), normalize=True)
            torch.save(local_generator.state_dict(), args.model_save_path +algorithm_name+ '/participant'+str(pid)+'Generator'+str(epoch)+'.pth')
            torch.save(local_discriminator.state_dict(), args.model_save_path + algorithm_name+'/participant'+str(pid)+'Discriminator'+str(epoch)+'.pth')
        torch.save(local_generator.state_dict(), args.model_save_path +algorithm_name+ '/participant'+str(pid)+'Generator.pth')
        torch.save(local_discriminator.state_dict(), args.model_save_path + algorithm_name+'/participant'+str(pid)+'Discriminator.pth')
    return args.model_save_path +algorithm_name+ '/participant'+str(pid)+'Generator.pth',args.model_save_path + algorithm_name+'/participant'+str(pid)+'Discriminator.pth'
    #return_queue.put((args.model_save_path + 'participant'+str(pid)+'Generator.pth',  args.model_save_path + 'participant'+str(pid)+'Discriminator.pth'))
    
def generator_local_train2(local_generator,local_discriminator,trainloader,pid,args,algorithm_name):#生成器本地迭代

    optimizer_G=torch.optim.Adam(local_generator.parameters(), lr=args.g_lr)
    optimizer_D=torch.optim.Adam(local_discriminator.parameters(), lr=args.d_lr)
    local_generator.train()
    local_discriminator.train()

    adversarial_loss = torch.nn.BCELoss()
    for epoch in range(args.gen_pro_epochs):
        # 使用数据加载器

        for images, labels in trainloader:

            labels_one_hot=F.one_hot(labels, num_classes=args.num_classes).to(args.device)

            # 生成分类随机整数
            unique_elements = torch.unique(labels)
            y = np.random.choice(unique_elements,size=args.batch_size)
            y = torch.tensor(y)
            # 转换为one-hot编码
            y_one_hot = F.one_hot(y, num_classes=args.num_classes).to(args.device)

            real_imgs = images
            real_imgs=real_imgs.to(args.device)
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            z = torch.from_numpy(np.random.normal(0, 1, (args.batch_size, args.z_dim))).float().to(args.device)
            gen_imgs = local_generator(z,y_one_hot)
            # Generate a batch of images
            g_output=local_discriminator(gen_imgs,y_one_hot)
            g_label = torch.ones_like(g_output) 
            g_loss = adversarial_loss(g_output, g_label)

            g_loss.backward()
            optimizer_G.step()

           # 判别器的损失
            optimizer_D.zero_grad()

            # 对于真实数据，判别器应该输出接近1的值
            d_output_real = local_discriminator(real_imgs, labels_one_hot)  # 将真实数据x和条件y输入判别器
            d_label_real = torch.ones_like(d_output_real)  # 真实数据的标签是1
            d_loss_real = adversarial_loss(d_output_real, d_label_real)

            # 对于生成的数据，判别器应该输出接近0的值
            d_output_fake = local_discriminator(gen_imgs.detach(), y_one_hot)  # 使用.detach()防止梯度传回生成器
            d_label_fake = torch.zeros_like(d_output_fake)  # 生成数据的标签是0
            d_loss_fake = adversarial_loss(d_output_fake, d_label_fake)

            # 判别器的总损失
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
        
        
        
        save_image(gen_imgs.data[:args.batch_size], args.img_save_path+algorithm_name+"/%d/%d_客户端.png" % (int(pid),int(epoch)), nrow=int(pow(args.batch_size,0.5)), normalize=True)
            
        torch.save(local_generator.state_dict(), args.model_save_path +algorithm_name+ '/participant'+str(pid)+'Generator.pth')
        torch.save(local_discriminator.state_dict(), args.model_save_path + algorithm_name+'/participant'+str(pid)+'Discriminator.pth')
    return args.model_save_path +algorithm_name+ '/participant'+str(pid)+'Generator.pth',args.model_save_path + algorithm_name+'/participant'+str(pid)+'Discriminator.pth'

def generate_samples(generator, num_samples,testlables, args): #生成总数为num_samples的样本
    generator.eval()  # 将生成器置于评估模式
    a_labels=[]
    a_samples=[]
    for i in range(num_samples):
        with torch.no_grad():  # 关闭梯度计算
            # 生成分类随机整数
            y = np.random.choice(testlables,size=args.batch_size)
            labels=copy.deepcopy(y)
            y = torch.tensor(y)
            # 转换为one-hot编码
            y_one_hot = F.one_hot(y, num_classes=args.num_classes).to(args.device)
            z = torch.from_numpy(np.random.normal(0, 1, (args.batch_size, args.z_dim))).float().to(args.device)
            samples = generator(z,y_one_hot)  
            for i, tensor in enumerate(torch.split(samples, 1)):
                a_labels.append(labels[i])
                a_samples.append(tensor)
    return a_labels,a_samples

def generate_samples2(generator, num_samples, test_labels, args):#生成样本，每个标签的样本数都为num_samples
    generator.eval()  # 将生成器置于评估模式
    a_labels = []
    a_samples = []
    for label in test_labels:  # 遍历每一个标签
        for _ in range(num_samples):
            with torch.no_grad():  # 关闭梯度计算
                # 创建当前标签的张量
                y = torch.tensor([label] * args.batch_size)
                # 转换为one-hot编码
                y_one_hot = F.one_hot(y, num_classes=args.num_classes).to(args.device)
                z = torch.from_numpy(np.random.normal(0, 1, (args.batch_size, args.z_dim))).float().to(args.device)
                samples = generator(z, y_one_hot)
                samples = samples.cpu()
                y_one_hot=y_one_hot.cpu()
                for i, tensor in enumerate(torch.split(samples, 1)):
                    a_labels.append(label)  # 保存当前标签
                    a_samples.append(torch.squeeze(tensor, dim=0))  # 保存生成的样本
    return a_labels, a_samples

def distribute_samples_equally(all_labels, all_samples, num_parts,args):
    # 确保数据可以被均匀分配
    total_batches = len(all_labels)
    batches_per_part = total_batches // num_parts
    distributed_labels = [[] for _ in range(num_parts)]
    distributed_samples = [[] for _ in range(num_parts)]

    # 分配每个部分
    for part_idx in range(num_parts):
        start_idx = part_idx * batches_per_part
        end_idx = start_idx + batches_per_part
        # 直接使用批次数据，避免额外的列表层
        distributed_labels[part_idx] = all_labels[start_idx:end_idx]
        distributed_samples[part_idx] = all_samples[start_idx:end_idx]

    data_loaders=[]
    for i in range(len(distributed_labels)):
        custom_dataset = models.LocalDataset()
        for j, tensor in enumerate(distributed_samples[i]):
            custom_dataset.append(tensor.squeeze(1), distributed_labels[i][j])

        data_loader = DataLoader(custom_dataset, batch_size=args.batch_size, shuffle=True)
        data_loaders.append(data_loader)
    return data_loaders

def data_classifier(Encoder, two_datas_with_labels, e):

    data1=two_datas_with_labels[0][0]
    data2=two_datas_with_labels[1][0]
    lable_one1=two_datas_with_labels[0][1]
    lable_one2=two_datas_with_labels[1][1]


    d1_copy=data1.detach().clone()
    d2_copy=data2.detach().clone()
    lable_one1_copy=lable_one1.detach().clone()
    lable_one2_copy=lable_one2.detach().clone()

    # 将所有数据和标签转换为张量，并且合并以进行聚类
    all_datas = torch.cat([d1_copy, d2_copy], dim=0)
    all_labels_one_hot = torch.cat([lable_one1_copy, lable_one2_copy], dim=0)
    
    # 生成数据来源标识
    source_labels = torch.cat([torch.zeros(d1_copy.size(0)),
                               torch.ones(d2_copy.size(0))])

    # 使用Encoder进行特征提取
    features = Encoder(all_datas).detach().cpu()
    features_scaled = features.view(features.size(0), -1).numpy()

    # 使用DBSCAN进行聚类
    dbscan = DBSCAN(eps=e, min_samples=1)
    clusters = dbscan.fit_predict(features_scaled)


    # 初始化存储结构
    classification_results = [[[], []], [[], []]]  # [[公共数据, 个性化数据], [公共数据, 个性化数据]]

    # 根据簇和来源标识进行数据分类
    for i, cluster_label in enumerate(clusters):
        data,one_hot=0,0
        #data = all_datas[i]
        # 查找all_datas[i]在data1中的索引
        for j,d in enumerate(data1):
            if torch.equal(d,all_datas[i])==True:
                data=data1[j]
                one_hot=lable_one1[j]
                break
        for j,d in enumerate(data2):
            if torch.equal(d,all_datas[i])==True:
                data=data2[j]
                one_hot=lable_one2[j]
                break
 
        source = source_labels[i].item()

        # 检查当前簇的其他数据点的来源
        other_sources = source_labels[clusters == cluster_label].unique()

        # 如果存在多个不同的来源，则认为是公共数据
        if len(other_sources) > 1:
            classification_results[int(source)][0].append((data, one_hot))
        else:
            classification_results[int(source)][1].append((data, one_hot))
    
    return classification_results

def adaptive_learning(generator1,discriminator1,generator2,discriminator2,lables1,lables2,args,Encoder):
    Encoder.eval()
    adversarial_loss = torch.nn.BCELoss()


    optimizer_G1=torch.optim.Adam(generator1.parameters(), lr=args.g_lr)
    optimizer_G2=torch.optim.Adam(generator2.parameters(), lr=args.g_lr)
    generator1.train()
    generator2.train()

    for i in range(args.adversarial_learning_epoch):
        y1_one_hot = F.one_hot(torch.tensor(np.random.choice(lables1,size=args.batch_size)), num_classes=args.num_classes).to(args.device)
        z1 = torch.from_numpy(np.random.normal(0, 1, (args.batch_size, args.z_dim))).float().to(args.device)

        y2_one_hot = F.one_hot(torch.tensor(np.random.choice(lables2,size=args.batch_size)), num_classes=args.num_classes).to(args.device)
        z2 = torch.from_numpy(np.random.normal(0, 1, (args.batch_size, args.z_dim))).float().to(args.device)

        datas1=generator1(z1,y1_one_hot)
        datas2=generator2(z2,y2_one_hot)

        two_datas=[]
        two_datas=[(datas1,y1_one_hot),(datas2,y2_one_hot)]

        classification_results=data_classifier(Encoder, two_datas,args.e)

        #加练公共部分，让生成器1骗鉴别器2，将生成器1的数据送入鉴别器2，返回损失
        optimizer_G1.zero_grad()
        if classification_results[0][0]:
            d1=torch.cat([d.unsqueeze(0) for d,_ in classification_results[0][0]], dim=0)
            l1=torch.cat([l.unsqueeze(0) for _, l in classification_results[0][0]], dim=0)
            d2_output=discriminator2(d1,l1)
            g1_label = torch.ones_like(d2_output)
            g1_loss = adversarial_loss(d2_output, g1_label)
        else:
            g1_loss = torch.tensor(0.).to(args.device)  # 或者设定一个合适的默认损失值
        #加练个性化部分
        if classification_results[0][1]:
            g_d1=torch.cat([d.unsqueeze(0) for d,_ in classification_results[0][1]], dim=0)
            g_l1=torch.cat([l.unsqueeze(0) for _, l in classification_results[0][1]], dim=0)
            g_d1_output=discriminator1(g_d1,g_l1)
            g_g1_label = torch.ones_like(g_d1_output)
            g1_personalized_loss = adversarial_loss(g_d1_output, g_g1_label)
        else:
            g1_personalized_loss = torch.tensor(0.).to(args.device)

        total_g1_loss = g1_loss + g1_personalized_loss
        total_g1_loss.backward()
        optimizer_G1.step()


        #加练公共部分，让生成器2骗鉴别器1，将生成器2的数据送入鉴别器1，返回损失
        optimizer_G2.zero_grad()
        if classification_results[1][0]:
            d2=torch.cat([d.unsqueeze(0) for d,_ in classification_results[1][0]], dim=0)
            l2=torch.cat([l.unsqueeze(0) for _, l in classification_results[1][0]], dim=0)
            d1_output=discriminator1(d2,l2)
            g2_label = torch.ones_like(d1_output)
            g2_loss = adversarial_loss(d1_output, g2_label)
        else:
            g2_loss = torch.tensor(0.).to(args.device)
        #加练个性化部分
        if classification_results[1][1]:
            g_d2=torch.cat([d.unsqueeze(0) for d,_ in classification_results[1][1]], dim=0)
            g_l2=torch.cat([l.unsqueeze(0) for _, l in classification_results[1][1]], dim=0)
            g_d2_output=discriminator2(g_d2,g_l2)
            g_g2_label = torch.ones_like(g_d2_output)
            g2_personalized_loss = adversarial_loss(g_d2_output, g_g2_label)
        else:
            g2_personalized_loss = torch.tensor(0.).to(args.device)
        total_g2_loss = g2_loss + g2_personalized_loss
        total_g2_loss.backward()
        optimizer_G2.step()

    return generator1,generator2

def data_generator(local_generator,local_trainloader,args):
    local_generator.eval()
    # 创建一个空字典
    sample_dict = {}
    # 使用自定义 Dataset
    custom_dataset = models.LocalDataset()
    # 遍历数据集
    for images, labels in local_trainloader:
        for image, label in zip(images, labels):
            #label = label.unsqueeze(0).unsqueeze(0)
            custom_dataset.append(image, label)
            if label.item() in sample_dict:
                sample_dict[label.item()] += 1
            else:
                sample_dict[label.item()] = 1

    max_sample_size=0
    for label, count in sample_dict.items():
        if count>max_sample_size:
            max_sample_size=count
    max_sample_size+=math.ceil(max_sample_size)#增强数据集
    # 计算每个标签需要生成的额外样本数量
    extra_samples = {}
    for label, count in sample_dict.items():
        extra_samples[label] = max_sample_size - count


    # 生成额外样本
    local_generator.to('cuda')
    for label, count in extra_samples.items():
        label = F.one_hot(torch.tensor(label), num_classes=args.num_classes).unsqueeze(0).to('cuda')
        for _ in range(count):
            z = torch.from_numpy(np.random.normal(0, 1, (1, args.z_dim))).float().to('cuda')
            generated_sample = local_generator(z, label)  # 假设label_tensor已经在循环外正确处理
            # 将生成的样本和标签转移到 CPU，以释放 GPU 显存
            generated_sample_cpu = torch.squeeze(generated_sample.detach().to('cpu'),dim=0)

            label_cpu = label.to('cpu')
            class_indices = torch.argmax(label_cpu, dim=1)
            # 示例添加数据
            class_indices=class_indices.squeeze(0)
            #generated_sample_cpu=generated_sample_cpu.squeeze(0)#[3,32,32]
            custom_dataset.append(generated_sample_cpu, class_indices)

    # 创建 DataLoader
    data_loader = DataLoader(custom_dataset, batch_size=args.batch_size, shuffle=True)
    return data_loader

def data_ganerator2(local_generators,gen_lables,generate_samples_num,args):

    gan_datasets =[]
    models.LocalDataset()
    #生成样本
    for i in range(args.clients_num):
        gan_dataset=models.LocalDataset()
        lables,samples = generate_samples2(local_generators[i], generate_samples_num, gen_lables[i], args)
        for i, tensor in enumerate(samples):
            gan_dataset.append(tensor.squeeze(1), lables[i])

        # 创建 DataLoader
        data_loader = DataLoader(gan_dataset, batch_size=args.batch_size, shuffle=True)
        gan_datasets.append(data_loader)
    return gan_datasets

def data_generator3(local_generator,local_trainloader,args):
    local_generator.eval()
    # 创建一个空字典
    sample_dict = {}
    # 使用自定义 Dataset
    custom_dataset = models.LocalDataset()
    # 遍历数据集
    for images, labels in local_trainloader:
        for image, label in zip(images, labels):
            if label.item() in sample_dict:
                sample_dict[label.item()] += 1
            else:
                sample_dict[label.item()] = 1

    max_sample_size=0
    for label, count in sample_dict.items():
        if count>max_sample_size:
            max_sample_size=count
    max_sample_size+=math.ceil(max_sample_size*0.1)#增强数据集
    # 计算每个标签需要生成的额外样本数量
    extra_samples = {}
    for label, count in sample_dict.items():
        extra_samples[label] = max_sample_size - count

    # 生成额外样本
    local_generator.to(args.device)
    for label, count in extra_samples.items():
        label = F.one_hot(torch.tensor(label), num_classes=args.num_classes).unsqueeze(0).to(args.device)
        for _ in range(count):
            z = torch.from_numpy(np.random.normal(0, 1, (1, args.z_dim))).float().to(args.device)
            generated_sample = local_generator(z, label)  # 假设label_tensor已经在循环外正确处理
            # 将生成的样本和标签转移到 CPU，以释放 GPU 显存
            generated_sample_cpu = torch.squeeze(generated_sample.detach().to('cpu'),dim=1)

            label_cpu = label.to('cpu')
            class_indices = torch.argmax(label_cpu, dim=1)
            # 示例添加数据
            class_indices=class_indices.squeeze(0)
            custom_dataset.append(generated_sample_cpu, class_indices)

    # 创建 DataLoader
    data_loader = DataLoader(custom_dataset, batch_size=args.batch_size, shuffle=True)
    return data_loader

def calculate_fid(real_images, generated_images, model):
    """计算两组图像的Fréchet Inception Distance (FID)"""
    model.eval()
    real_images = torch.stack(real_images).squeeze(0)
    generated_images = torch.stack(generated_images)
    generated_images = torch.squeeze(generated_images, dim=2)

    # 计算特征
    with torch.no_grad():
        act1 = model(real_images.to('cuda')).detach()
        act2 = model(generated_images.to('cuda')).detach()
        act1=torch.reshape(act1, (100, 1024)).cpu().numpy()
        act2=torch.reshape(act2, (100, 1024)).cpu().numpy()
    distances = np.sqrt(np.sum((act1 - act2)**2))
    # 计算均值和协方差
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
    
    # 计算FID
    ssdiff = np.sum((mu1 - mu2)**2)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    
    return distances

def data_fetch(local_trainloader, num=100):
    # 使用 data_fetch 函数获取标签为1的数据样本
    #label_1_samples = data_fetch(local_trainloader, num=100).get(1, [])
    # 创建一个空字典来保存每个标签下的样本
    sample_dict = {}
    
    # 遍历数据加载器中的每个样本
    for images, labels in local_trainloader:
        for image, label in zip(images, labels):
            # 转换标签为整数
            label = label.item()
            
            # 如果字典中已经有该标签对应的样本列表，且样本数量小于 num，则将样本添加到列表中
            if label in sample_dict and len(sample_dict[label]) < num:
                sample_dict[label].append(image)
            # 如果字典中没有该标签对应的样本列表，则创建一个新列表，并将样本添加到其中
            elif label not in sample_dict:
                sample_dict[label] = [image]
                
            # 检查是否每个标签的样本数量都已达到 num，如果是则返回结果
            if all(len(samples) >= num for samples in sample_dict.values()):
                return sample_dict
    
    # 如果所有样本都被遍历了但是仍然没有每个标签达到 num 个样本，则返回当前的样本字典
    return sample_dict

def generate_data_for_specified_labels(local_generator, specified_label, args, num=100):
    # 将标签转换为 one-hot 编码，并移到 GPU
    label = F.one_hot(torch.tensor(specified_label), num_classes=args.num_classes).unsqueeze(0).to('cuda')
    
    # 初始化一个空列表来存储生成的样本
    generated_samples = []
    
    # 循环生成指定数量的样本
    for _ in range(num):
        # 生成随机的噪声向量 z
        z = torch.from_numpy(np.random.normal(0, 1, (1, args.z_dim))).float().to('cuda')
        
        # 使用生成器生成样本
        with torch.no_grad():
            generated_sample = local_generator(z, label)
        
        # 将生成的样本添加到列表中
        generated_samples.append(generated_sample)
    
    return generated_samples