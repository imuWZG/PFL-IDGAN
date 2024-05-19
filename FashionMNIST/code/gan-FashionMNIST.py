import copy
import torch
from torch.utils.data import DataLoader
import utils
import argparse
from models import Generator,Discriminator,FeatureExtractor,Autoencoder
from multiprocessing import Queue
import torch.multiprocessing as mp
import csv
import math
from torchvision.models import inception_v3


       
def PFLDGAN_train(train_dataloaders,testlables,test_dataloaders,clients_models,local_generators,local_discriminators,autoencoder,args):


    for pid in range(args.clients_num):
        g,d=utils.generator_local_train(local_generators[pid],local_discriminators[pid],train_dataloaders[pid],pid,args,'PFLDGAN')
        local_generators[pid].load_state_dict(torch.load(g))
        local_discriminators[pid].load_state_dict(torch.load(d))
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--FL_epochs", type=int, default=30, help="Number of FL epochs")
    parser.add_argument("--local_epochs", type=int, default=1, help="Number of local epochs")
    parser.add_argument("--gen_pro_epochs", type=int, default=53, help="Number of generator process epochs")
    parser.add_argument("--device", type=str, default='cuda', help=' ')
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--g_lr", type=float, default=0.00005, help="adam:Generator learning rate")
    parser.add_argument("--d_lr", type=float, default=0.0005, help="adam:Discriminator learning rate")
    parser.add_argument("--c_lr", type=float, default=0.0002, help="adam:Classifier learning rate")
    parser.add_argument("--z_dim", type=int, default=150, help="dimensionality of the latent space")
    parser.add_argument("--clients_num", type=int, default=5, help="number of clients")
    parser.add_argument("--img_save_path", type=str, default='/workspace/algorithm/code/gan_images/FashionMNIST/', help="image save path")
    parser.add_argument("--model_save_path", type=str, default='/workspace/algorithm/code/models/FashionMNIST/', help="Generator model save path")
    parser.add_argument("--testset_sample_num", type=int, default=1000, help="Number of test sets")
    parser.add_argument("--encoder_path", type=str, default="/workspace/algorithm/code/models/MNIST-Encoder.pth", help="Encoder path")
    parser.add_argument("--adversarial_learning_epoch", type=int, default=100, help="The number of rounds in adversarial learning")
    parser.add_argument("--PerFEDGAN_generate_samples_num", type=int, default=300, help="")
    parser.add_argument("--PFLDGAN_generate_samples_num", type=int, default=100, help="")


    parser.add_argument("--dataset_name", type=str, default='FashionMNIST', help="Path to dataset")
    parser.add_argument("--dataset_root", type=str, default='/workspace/dataset/private/wzg_dataset/', help="Path to dataset")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--e", type=int, default=5.922, help="")
    parser.add_argument("--p_num", type=int, default=100, help="")
    args = parser.parse_args()

    #分割数据集，生成每个客户端的本地数据集
    subsets,testlables,Public_Set=utils.load_and_process_dataset2(args.dataset_name, args.dataset_root+args.dataset_name, args.clients_num,2000,args)
    local_trainloaders = [DataLoader(subset, batch_size=args.batch_size, shuffle=True) for subset in subsets]
    Public_Set=utils.generate_testset2(testlables, args.dataset_name, args.dataset_root+args.dataset_name, args.batch_size, 100)
    #根据客户端标签，生成每个客户端的测试集
    test_sets=[]
    for testlable in testlables:
        test_sets.append(utils.generate_testset(testlable,args.dataset_name,args.dataset_root+args.dataset_name,args))

   
    #PFLDGAN
    PFLDGAN_clients_models=[]
    PFLDGAN_local_generators=[]
    PFLDGAN_local_discriminators=[]
    PFLDGAN_feature_extractor=FeatureExtractor(args.channels).to(args.device)
    PFLDGAN_autoencoder = Autoencoder().to(args.device)
    PFLDGAN_autoencoder.load_state_dict(torch.load(args.encoder_path))
    PFLDGAN_autoencoder.eval()
    for i in range(args.clients_num):
        PFLDGAN_clients_model=utils.create_custom_cnn_2('MNIST','PFLDGAN')
        PFLDGAN_clients_models.append(PFLDGAN_clients_model)
        PFLDGAN_local_generator=Generator(args.z_dim, args.channels, args.img_size, args.num_classes).to(args.device)
        PFLDGAN_local_discriminator=Discriminator(PFLDGAN_feature_extractor,args.img_size,args.num_classes,args.channels).to(args.device)

        PFLDGAN_local_generators.append(PFLDGAN_local_generator)
        PFLDGAN_local_discriminators.append(PFLDGAN_local_discriminator)



      

    PFLDGAN_clients_models,PFLDGAN_local_generators,PFLDGAN_local_discriminators,PFLDGAN_avg_acc=PFLDGAN_train(local_trainloaders,
                                    testlables,
                                    test_sets,
                                    PFLDGAN_clients_models,
                                    PFLDGAN_local_generators,
                                    PFLDGAN_local_discriminators,
                                    PFLDGAN_autoencoder,
                                    args)
     
if __name__=='__main__':
    mp.set_start_method('spawn')
    main()
