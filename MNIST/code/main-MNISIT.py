import copy
import torch
from torch.utils.data import DataLoader
import utils2
import argparse
from models import Generator,Discriminator,FeatureExtractor,Autoencoder
from multiprocessing import Queue
import torch.multiprocessing as mp
import csv
import math
from torchvision.models import inception_v3

def Local_train(train_dataloaders,test_dataloaders,clients_models,args):
    all_clients_accuracys=[]
    for index,clients_model in enumerate(clients_models):
        for i in range(args.local_epochs):
            clients_models[index].load_state_dict(utils2.train(clients_model,args.device,train_dataloaders[index],args))
        accuracy = utils2.test(clients_models[index],test_dataloaders[index],args)
        #print('客户端%d 的精度:%f%%' % (index,accuracy))
        all_clients_accuracys.append(accuracy)
    avg_acc=utils2.calculate_average(all_clients_accuracys)
    print('Local客户端平均精度:%f%%' % (avg_acc))

    return clients_models,avg_acc

def FedAvg_train(train_dataloaders,test_dataloaders,global_model,clients_models,args):
    all_clients_accuracys=[]
    for index,clients_model in enumerate(clients_models):
        for i in range(args.local_epochs):
            clients_models[index].load_state_dict(utils2.train(clients_model,args.device,train_dataloaders[index],args))
        accuracy = utils2.test(clients_models[index],test_dataloaders[index],args)
        #print('客户端%d 的精度:%f%%' % (index,accuracy))
        all_clients_accuracys.append(accuracy)
    avg_acc=utils2.calculate_average(all_clients_accuracys)
    print('FedAvg客户端平均精度:%f%%' % (avg_acc))

    global_model=utils2.average_model_weights(global_model.to(args.device),clients_models)
    return global_model,clients_models,avg_acc

def FedProx_train(train_dataloaders,test_dataloaders,global_model,clients_models,args):
    all_clients_accuracys=[]
    for index,clients_model in enumerate(clients_models):
        for i in range(args.local_epochs):
            clients_models[index].load_state_dict(utils2.train(clients_model,args.device,train_dataloaders[index],args))
        accuracy = utils2.test(clients_models[index],test_dataloaders[index],args)
        #print('客户端%d 的精度:%f%%' % (index,accuracy))
        all_clients_accuracys.append(accuracy)
    avg_acc=utils2.calculate_average(all_clients_accuracys)
    print('FedProx客户端平均精度:%f%%' % (avg_acc))

    new_global_state_dict=utils2.fed_prox(clients_models, global_model.to(args.device).state_dict())
    global_model.load_state_dict(new_global_state_dict)

    return global_model,clients_models,avg_acc

def FedMD_train(train_dataloaders,test_dataloaders,Public_Set,clients_models,args):
    all_clients_accuracys=[]
    for index,clients_model in enumerate(clients_models):
        for i in range(args.local_epochs):
            clients_models[index].load_state_dict(utils2.train(clients_model,args.device,train_dataloaders[index],args))
        accuracy = utils2.test(clients_models[index],test_dataloaders[index],args)
        #print('客户端%d 的精度:%f%%' % (index,accuracy))
        all_clients_accuracys.append(accuracy)
    avg_acc=utils2.calculate_average(all_clients_accuracys)
    print('FedMD客户端平均精度:%f%%' % (avg_acc))

    clients_models=utils2.fedmd_step(clients_models,Public_Set,args)

    return clients_models,avg_acc

def PerFEDGAN_train(train_dataloaders,test_labels,test_dataloaders,clients_models,local_generators,local_discriminators,args,e):
    for xx in range(args.local_epochs):
        #本地生成器训练
        for pid in range(args.clients_num):
            g,d=utils2.generator_local_train2(local_generators[pid],local_discriminators[pid],train_dataloaders[pid],pid,args,'PerFEDGAN')
            local_generators[pid].load_state_dict(torch.load(g))
            local_discriminators[pid].load_state_dict(torch.load(d))
    '''   
        #本地训练
        all_clients_accuracys=[]
        for index,clients_model in enumerate(clients_models):
            clients_models[index].load_state_dict(utils2.train(clients_model,args.device,train_dataloaders[index],args))
            accuracy = utils2.test(clients_models[index],test_dataloaders[index],args)
            all_clients_accuracys.append(accuracy)
        avg_acc=utils2.calculate_average(all_clients_accuracys)
        print('PerFEDGAN客户端平均精度:%f%%' % (avg_acc))

    #生成样本
    all_samples=[]
    all_lables=[]
    for i in range(args.clients_num):
        lables,samples = utils2.generate_samples2(local_generators[i], args.PerFEDGAN_generate_samples_num, testlables[i], args)
        for i, tensor in enumerate(samples):
            all_samples.append(tensor)
            all_lables.append(lables[i])
    

    #混合样本和本地数据，训练本地任务模型
    gan_data_loaders=utils2.distribute_samples_equally(all_lables,all_samples, args.clients_num,args)
    for index,clients_model in enumerate(clients_models):
        clients_models[index].load_state_dict(utils2.train(clients_model,args.device,gan_data_loaders[index],args))
    ''' 
    #生成样本
    all_samples=[]
    all_labels=[]
    for i in range(args.clients_num):
        lables,samples = utils2.generate_samples2(local_generators[i], args.PerFEDGAN_generate_samples_num, test_labels[i], args)
        for i, tensor in enumerate(samples):
            all_samples.append(tensor)
            all_labels.append(lables[i])
    # 混合样本和本地数据，创建新的数据加载器
    mixed_dataloaders = []
    for i in range(args.clients_num):
        # 获取本地数据
        local_dataset = train_dataloaders[i].dataset
        local_samples, local_labels = [], []

        for img, lbl in local_dataset:
            local_samples.append(img)
            local_labels.append(lbl)

        # 获取生成的数据
        generated_samples = [sample for j, sample in enumerate(all_samples) if all_labels[j] in test_labels[i]]
        generated_labels = [label for label in all_labels if label in test_labels[i]]

        # 合并本地数据和生成数据
        mixed_samples = local_samples + generated_samples
        mixed_labels = local_labels + generated_labels

        # 创建新的数据加载器
        mixed_dataset = torch.utils.data.TensorDataset(torch.stack(mixed_samples), torch.tensor(mixed_labels))
        mixed_dataloader = torch.utils.data.DataLoader(mixed_dataset, batch_size=args.batch_size, shuffle=True)
        mixed_dataloaders.append(mixed_dataloader)

    # 使用混合数据集训练本地任务模型
    all_clients_accuracys=[]
    for index, clients_model in enumerate(clients_models):
        clients_models[index].load_state_dict(utils2.train(clients_model, args.device, mixed_dataloaders[index], args))
        accuracy = utils2.test(clients_models[index],test_dataloaders[index],args)
        all_clients_accuracys.append(accuracy)
    avg_acc=utils2.calculate_average(all_clients_accuracys)
    print('PerFEDGAN客户端平均精度:%f%%' % (avg_acc))
    return clients_models,local_generators,local_discriminators,avg_acc
       
def PFLDGAN_train(train_dataloaders,testlables,test_dataloaders,clients_models,local_generators,local_discriminators,autoencoder,args,e):
    for xx in range(args.local_epochs):
        #本地生成器迭代
        for pid in range(args.clients_num):
            g,d=utils2.generator_local_train2(local_generators[pid],local_discriminators[pid],train_dataloaders[pid],pid,args,'PFLDGAN')
            local_generators[pid].load_state_dict(torch.load(g))
            local_discriminators[pid].load_state_dict(torch.load(d))
       
        #本地训练
        all_clients_accuracys=[]
        for index,clients_model in enumerate(clients_models):
            new_dataloader=utils2.data_generator(local_generators[index],train_dataloaders[index],args)
            clients_models[index].load_state_dict(utils2.train(clients_model,args.device,new_dataloader,args))
            accuracy = utils2.test(clients_models[index],test_dataloaders[index],args)
            all_clients_accuracys.append(accuracy)
        avg_acc=utils2.calculate_average(all_clients_accuracys)
        print('PFLDGAN客户端平均精度:%f%%' % (avg_acc))
        
        #生成器相互学习
        for i in range(len(local_generators)):#相互学习
            j=i+1
            while j<=len(local_generators)-1:
                local_generators[i],local_generators[j]=utils2.adaptive_learning(local_generators[i],
                                                                                local_discriminators[i],
                                                                                local_generators[j],
                                                                                local_discriminators[j],
                                                                                testlables[i],
                                                                                testlables[j],
                                                                                args,
                                                                                autoencoder.encoder)
                j+=1
      
    return clients_models,local_generators,local_discriminators,avg_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--FL_epochs", type=int, default=20, help="Number of FL epochs")
    parser.add_argument("--local_epochs", type=int, default=1, help="Number of local epochs")
    parser.add_argument("--gen_pro_epochs", type=int, default=4, help="Number of generator process epochs")
    parser.add_argument("--device", type=str, default='cuda', help=' ')
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--g_lr", type=float, default=0.0005, help="adam:Generator learning rate")
    parser.add_argument("--d_lr", type=float, default=0.0005, help="adam:Discriminator learning rate")
    parser.add_argument("--c_lr", type=float, default=0.000005, help="adam:Classifier learning rate")
    parser.add_argument("--z_dim", type=int, default=150, help="dimensionality of the latent space")
    parser.add_argument("--clients_num", type=int, default=5, help="number of clients")
    parser.add_argument("--img_save_path", type=str, default='/workspace/algorithm/code/gan_images/MNIST/', help="image save path")
    parser.add_argument("--model_save_path", type=str, default='/workspace/algorithm/code/models/MNIST/', help="Generator model save path")
    parser.add_argument("--testset_sample_num", type=int, default=1000, help="Number of test sets")
    parser.add_argument("--encoder_path", type=str, default="/workspace/algorithm/code/models/MNIST-Encoder.pth", help="Encoder path")
    parser.add_argument("--adversarial_learning_epoch", type=int, default=100, help="The number of rounds in adversarial learning")
    parser.add_argument("--PerFEDGAN_generate_samples_num", type=int, default=100, help="")
    parser.add_argument("--PFLDGAN_generate_samples_num", type=int, default=100, help="")


    parser.add_argument("--dataset_name", type=str, default='MNIST', help="Path to dataset")
    parser.add_argument("--dataset_root", type=str, default='/workspace/dataset/private/wzg_dataset/', help="Path to dataset")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--e", type=int, default=5.922, help="")
    parser.add_argument("--p_num", type=int, default=50, help="")
    args = parser.parse_args()

    #分割数据集，生成每个客户端的本地数据集
    subsets,testlables,Public_Set=utils2.load_and_process_dataset2(args.dataset_name, args.dataset_root+args.dataset_name, args.clients_num,3000,args)
    local_trainloaders = [DataLoader(subset, batch_size=args.batch_size, shuffle=True) for subset in subsets]
    #Public_Set=utils2.generate_testset2(testlables, args.dataset_name, args.dataset_root+args.dataset_name, args.batch_size, 50)
    #根据客户端标签，生成每个客户端的测试集
    test_sets=[]
    for testlable in testlables:
        test_sets.append(utils2.generate_testset(testlable,args.dataset_name,args.dataset_root+args.dataset_name,args))

    #生成所有算法的客户端模型，依赖全局模型的生成全局模型clear

   #FedAvg
    FedAvg_global_model = utils2.create_custom_cnn_2(args.dataset_name,'FedAvg')
    FedAvg_clients_models=[]
    for i in range(args.clients_num):
        FedAvg_clients_model=copy.deepcopy(FedAvg_global_model)
        FedAvg_clients_models.append(FedAvg_clients_model)
    #Local
    Local_clients_models=[]
    for i in range(args.clients_num):
        Local_clients_model=copy.deepcopy(FedAvg_global_model)
        Local_clients_models.append(Local_clients_model)
    #FedProx
    FedProx_global_model = utils2.create_custom_cnn_2(args.dataset_name,'FedProx')
    FedProx_clients_models=[]
    for i in range(args.clients_num):
        FedProx_clients_model=copy.deepcopy(FedProx_global_model)
        FedProx_clients_models.append(FedProx_clients_model)
    #FedMD
    FedMD_clients_models=[]
    for i in range(args.clients_num):
        FedMD_clients_model=utils2.create_custom_cnn_2(args.dataset_name,'FedMD')
        FedMD_clients_models.append(FedMD_clients_model)
    #PerFEDGAN
    PerFEDGAN_clients_models=[]
    PerFEDGAN_local_generators=[]
    PerFEDGAN_local_discriminators=[]
    PerFEDGAN_feature_extractor=FeatureExtractor(args.channels).to(args.device)
    for i in range(args.clients_num):
        PerFEDGAN_local_generator=Generator(args.z_dim, args.channels, args.img_size, args.num_classes).to(args.device)
        PerFEDGAN_local_discriminator=Discriminator(PerFEDGAN_feature_extractor,args.img_size,args.num_classes,args.channels).to(args.device)
        '''
        utils2.model_loading(PerFEDGAN_local_generator,
                            '/workspace/algorithm/code/models/MNIST/loading_model/participant'+str(i)+'Generator30.pth',
                            '/workspace/algorithm/code/models/MNIST/loading_model/participant'+str(i)+'Generator30.pth')
        utils2.model_loading(PerFEDGAN_local_discriminator,
                            '/workspace/algorithm/code/models/MNIST/loading_model/participant'+str(i)+'Discriminator30.pth',
                            '/workspace/algorithm/code/models/MNIST/loading_model/participant'+str(i)+'Discriminator30.pth')
        '''
        PerFEDGAN_local_generators.append(PerFEDGAN_local_generator)
        PerFEDGAN_local_discriminators.append(PerFEDGAN_local_discriminator)
        PerFEDGAN_clients_model=utils2.create_custom_cnn_2(args.dataset_name,'PerFEDGAN')
        PerFEDGAN_clients_models.append(PerFEDGAN_clients_model)
    #PFLDGAN
    PFLDGAN_clients_models=[]
    PFLDGAN_local_generators=[]
    PFLDGAN_local_discriminators=[]
    PFLDGAN_feature_extractor=FeatureExtractor(args.channels).to(args.device)
    PFLDGAN_autoencoder = Autoencoder().to(args.device)
    PFLDGAN_autoencoder.load_state_dict(torch.load(args.encoder_path))
    PFLDGAN_autoencoder.eval()
    for i in range(args.clients_num):
        PFLDGAN_clients_model=utils2.create_custom_cnn_2('MNIST','PFLDGAN')
        PFLDGAN_clients_models.append(PFLDGAN_clients_model)
        PFLDGAN_local_generator=Generator(args.z_dim, args.channels, args.img_size, args.num_classes).to(args.device)
        PFLDGAN_local_discriminator=Discriminator(PFLDGAN_feature_extractor,args.img_size,args.num_classes,args.channels).to(args.device)
        '''
        utils2.model_loading(PFLDGAN_local_generator,
                            '/workspace/algorithm/code/models/MNIST/loading_model/participant'+str(i)+'Generator30.pth',
                            '/workspace/algorithm/code/models/MNIST/loading_model/participant'+str(i)+'Generator30.pth')
        utils2.model_loading(PFLDGAN_local_discriminator,
                            '/workspace/algorithm/code/models/MNIST/loading_model/participant'+str(i)+'Discriminator30.pth',
                            '/workspace/algorithm/code/models/MNIST/loading_model/participant'+str(i)+'Discriminator30.pth')
        '''
        PFLDGAN_local_generators.append(PFLDGAN_local_generator)
        PFLDGAN_local_discriminators.append(PFLDGAN_local_discriminator)
        

    for e in range(args.FL_epochs):

        Local_clients_models,FedAvg_avg_acc=Local_train(local_trainloaders,test_sets,Local_clients_models,args)

        
        FedAvg_global_model,FedAvg_clients_models,FedAvg_avg_acc=FedAvg_train(local_trainloaders,test_sets,FedAvg_global_model,FedAvg_clients_models,args)
        with open('/workspace/algorithm/code/FedAvg_avg_acc.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow( [format(FedAvg_avg_acc, '.4f')])

        FedProx_global_model,FedProx_clients_models,FedProx_avg_acc=FedProx_train(local_trainloaders,test_sets,FedProx_global_model,FedProx_clients_models,args)
        with open('/workspace/algorithm/code/FedProx_avg_acc.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([format(FedProx_avg_acc, '.4f')])

        FedMD_clients_models,FedMD_avg_acc=FedMD_train(local_trainloaders,test_sets,Public_Set,FedMD_clients_models,args)
        with open('/workspace/algorithm/code/FedMD_avg_acc.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([format(FedMD_avg_acc, '.4f')])


        PerFEDGAN_clients_models,PerFEDGAN_local_generators,PerFEDGAN_local_discriminators,PerFEDGAN_avg_acc=PerFEDGAN_train(local_trainloaders,testlables,test_sets,PerFEDGAN_clients_models,PerFEDGAN_local_generators,PerFEDGAN_local_discriminators,args,e)
        with open('/workspace/algorithm/code/PerFEDGAN_avg_acc.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([format(PerFEDGAN_avg_acc, '.4f')])


        PFLDGAN_clients_models,PFLDGAN_local_generators,PFLDGAN_local_discriminators,PFLDGAN_avg_acc=PFLDGAN_train(local_trainloaders,
                                      testlables,
                                      test_sets,
                                      PFLDGAN_clients_models,
                                      PFLDGAN_local_generators,
                                      PFLDGAN_local_discriminators,
                                      PFLDGAN_autoencoder,
                                      args,
                                      e)
        with open('/workspace/algorithm/code/PFLDGAN_avg_acc.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([format(PFLDGAN_avg_acc, '.4f')])
           
        print('-------------------------')
if __name__=='__main__':
    main()
