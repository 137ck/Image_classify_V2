import torch
import os
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torchvision import transforms,datasets,models
import pandas as pd
import matplotlib.pyplot as plt
import json
from PIL import Image
import argparse
    
#data_path ='ImageClassifier/flowers'
# data_dir = 'flowers'
# train_dir = data_dir + '/train'
# valid_dir = data_dir + '/valid'
# test_dir = data_dir + '/test'
def main():
    args = get_arguments()
    if args.model == 'densenet121':
       model = models.densenet121(pretrained=True)
       for parameter in model.parameters():
           parameter.requires_grad=False
       classifier_new= nn.Sequential(nn.Linear(1024,256),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.2),
                                     nn.Linear(256,102),
                                     nn.LogSoftmax(dim=1))       
       input_num=1024

    if args.model == 'vgg19':
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier_new = nn.Sequential(OrderedDict([
                                        ('input',  nn.Linear(25088, 6000)),
                                        ('relu1',  nn.ReLU()),
                                        ('dropout1',  nn.Dropout(p=0.2)),
                                        ('linear2',  nn.Linear(6000, 400)),
                                        ('relu2',  nn.ReLU()),
                                        ('linear3', nn.Linear(400, 102)),
                                        ('output', nn.LogSoftmax(dim=1))
                                       ])) 
        input_num=25088
    
    model.classifier = classifier_new
    data_loaders, image_datasets, data_transforms = data_parser(args.data_path)
        
    if args.cuda:
       model = model.cuda()
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    
    train_model(model, data_loaders[0], criterion=criterion, optimizer=optimizer, epochs=int(args.epochs), cuda=args.cuda)
    
    test_model(model, data_loaders[2], cuda=args.cuda)

    checkpoint = {'input_size': input_num,
                  'output_size': 102,
                  'epochs_nums': args.epochs,
                  'learning_rate':args.lr,
                  'batch_size': 64,
                  'data_transforms': data_transforms,
                  'model': model,
                  'classifier': classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': image_datasets[0].class_to_idx
                }

    torch.save(checkpoint, 'checkpoint.pth')
    
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", action="store", dest="save_dir", default="." , help = "Choose directory to save                                       checkpoints")
    parser.add_argument("--model", action="store", dest="model", default="densenet121" , help = "Choose architechture('densenet121'                         or'vgg19')")
    parser.add_argument("--learning_rate", action="store", dest="lr", default=0.003 , help = "Set learning rate into system")
    parser.add_argument("--hidden_layers", action="store", dest="hidden_layers", default=512 , help = "Set number of hidden layers")
    parser.add_argument("--epochs", action="store", dest="epochs", default=5 , help = "Set number of epochs in system")
    parser.add_argument("--gpu", action="store_true", dest="cuda", default=False , help = "Use CUDA for AI training")
    parser.add_argument('--data_path', action="store",dest='data_path',default='ImageClassifier/flowers')
    return parser.parse_args()

def data_parser(data_path):  
    train_dir = data_path + '/train'
    valid_dir = data_path + '/valid'
    test_dir = data_path + '/test'
    
    batch_size = 64
    
    train_transform = transforms.Compose( [transforms.Resize(256),
                                          transforms.RandomRotation(45),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],
                                                               [0.229,0.224,0.225])]) 
    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485,0.456,0.406],
                                                               [0.229,0.224,0.225])])
    valid_transform=test_transform
    data_transforms=[train_transform,test_transform,valid_transform]
    
    train_datasets = datasets.ImageFolder(train_dir,transform=train_transform)
    valid_datasets = datasets.ImageFolder(valid_dir,transform=valid_transform)
    test_datasets=datasets.ImageFolder(test_dir,transform=valid_transform)
    image_datasets=[train_datasets,valid_datasets,test_datasets]

    train_dataloader = torch.utils.data.DataLoader(train_datasets,batch_size=64,shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_datasets,batch_size=64)
    test_dataloader = torch.utils.data.DataLoader(test_datasets,batch_size=64)
    dataloaders = [train_dataloader,valid_dataloader,test_dataloader]     
    
    return dataloaders, image_datasets, data_transforms

def train_model(model, dataloaders, criterion, optimizer, epochs=15, cuda=False):
    counts = 0   
    screen_every = 30
    running_loss=0
    
    if cuda and torch.cuda.is_available:
        model.cuda()
    else:
        model.cpu()

    for epoch in range(epochs):
        for images,labels in dataloaders[0]:
            epoch+= 1
            if cuda:
                images, labels = images.cuda(), labels.cuda()
            else:
                images, labels = images.cpu(), labels.cpu()
 
            optimizer.zero_grad()
            logps= model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            
            if steps % screen_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for images,labels in dataloaders[1]:
                        logps = model.forward(inputs)
                        valid_loss += criterion(logps, labels).item()
                        ps = torch.exp(logps)
                        top_ps, top_class=ps.topk(1,dim=1)
                        equality=top_class==labels.view(*top_class.shape)                                                                     
                        accuracy=accuracy+torch.mean(equality.type(torch.FloatTensor)).item()

                print(f"Num {num+1}/{nums};"
                      f"Train_loss: {present_loss/screen_every:.3f};"
                      f"Valid_loss: {valid_loss/len(valid_dataloader):.3f};"
                      f"Valid_accuracy: {accuracy/len(valid_dataloader)*100:.3f}%")

                running_loss = 0
                model.train()

def test_model(model, dataloaders, cuda=False):
    model.eval()
    test_accuracy=0
    test_loss=0
    if cuda and torch.cuda.is_available:
        model.cuda()
    else:
        model.cpu()
    with torch.no_grad():
        for images,labels in test_dataloader:
            if cuda:
               images, labels = images.cuda(), labels.cuda()
            else:
               images, labels = images.cpu(), labels.cpu()
            
            logps=model.forward(images)
            loss=criterion(logps,labels)
            test_loss=test_loss+loss.item()
            ps=torch.exp(logps)
            top_ps, top_class=ps.topk(1,dim=1)
            equality=top_class==labels.view(*top_class.shape)
            test_accuracy=test_accuracy+torch.mean(equality.type(torch.FloatTensor)).item()

    print(f"test_loss: {test_loss/len(test_dataloader):.3f};"
          f"test_accuracy: {test_accuracy/len(test_dataloader)*100:.3f}%") 

main()
