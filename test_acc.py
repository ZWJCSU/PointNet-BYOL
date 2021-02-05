# -*- coding: utf-8 -*-
"""UCMerced_linear_feature_eval.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1866eTCYkCpZd_9XeIJ54MatLGHhtBGsF
"""



import torch
import sys
import yaml
from torchvision import transforms, datasets
import torchvision
import numpy as np
import os
import provider
import importlib
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import preprocessing
from torch.utils.data.dataloader import DataLoader
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from models.resnet_base_network import ResNet18
from models.pointnet2_cls_msg import get_model
from models.pointnet2_cls_msg import get_loss
import pointnet2_cls_msg_concat
import pointnet2_cls_msg_raw



# train_dataset = datasets.STL10('/home/thalles/Downloads/', split='train', download=False,
#                                transform=data_transforms)

# test_dataset = datasets.STL10('/home/thalles/Downloads/', split='test', download=False,
#                                transform=data_transforms)

from torchvision import datasets
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18





class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x=self.linear(x)
        return F.log_softmax(x, -1)



def get_features_from_encoder(encoder, loader):
    raw_model = pointnet2_cls_msg_raw.get_model(num_class=40,normal_channel=True).cuda()
    x_train = []
    y_train = []
    print(type(loader))
    # get the features from the pre-trained model
    for batch_id, data in tqdm(enumerate(loader, 0), total=len(loader), smoothing=0.9):
           points, target = data
           points = points.data.numpy()
           points = provider.random_point_dropout(points)
           points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
           points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
           points = torch.Tensor(points)
           target = target[:, 0]
           points = points.transpose(2, 1)
           points, target = points.cuda(), target.cuda()
           with torch.no_grad():
                raw_feature,raw_cls = raw_model(points)
                feature_vector,cls = encoder(points)
                feature_vector = torch.cat((raw_feature,feature_vector),1)
                #这里要用extend,append会把[12*128]一块放进去
                x_train.extend(feature_vector.cpu().numpy())
                y_train.extend(target.cpu().numpy())
    x_train = np.array(x_train)
    y_train = torch.tensor(y_train)
    print("success feature")


    # for i, (x,y) in enumerate(loader):
    #     # i=i.to(device)
    #     # x=x.to(device)
    #     # y=y.to(device)
    #     x1=torch.tensor([item.cpu().detach().numpy() for item in x1]).cuda() 
    #     with torch.no_grad():
    #         feature_vector = encoder(x1)
    #         x_train.extend(feature_vector)
    #         y_train.extend(y.numpy())
    return x_train, y_train



def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test):

    train = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True)

    test = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=8, shuffle=True)
    return train_loader, test_loader



def get_acc():
 batch_size = 8

 config = yaml.load(open("/content/PointNet-BYOL/config/config.yaml", "r"), Loader=yaml.FullLoader)
#这里normal_channel一定要改成True,不然channel会变成3，无法与6匹配
 TRAIN_DATASET = ModelNetDataLoader(root='data/modelnet40_normal_resampled/', npoint=1024, split='train',
                                                  normal_channel=True)
 TEST_DATASET = ModelNetDataLoader(root='data/modelnet40_normal_resampled/', npoint=1024, split='test',
                                                 normal_channel=True)


 print("Input shape:", len(TRAIN_DATASET))

 train_loader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=8, shuffle=True, num_workers=12)
 test_loader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=8, shuffle=False, num_workers=12)
 device = 'cuda' if torch.cuda.is_available() else 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
 encoder = get_model(num_class=40,normal_channel=True)
#  output_feature_dim = encoder.projetion.net[0].in_features# 



#  load pre-trained parameters
 load_params = torch.load(os.path.join('/content/PointNet-BYOL/checkpoints/model.pth'),
                          map_location=torch.device(torch.device(device)))

 if 'online_network_state_dict' in load_params:
     encoder.load_state_dict(load_params['online_network_state_dict'])
     print("Parameters successfully loaded.")

 # remove the projection head
 encoder = encoder.to(device)
 encoder.eval()
 
#  x_train, y_train = get_features_from_encoder(encoder, train_loader)
#  x_test, y_test = get_features_from_encoder(encoder, test_loader)

 
#  x_train = torch.mean(x_train, dim=[2, 3])
#  x_test = torch.mean(x_test, dim=[2, 3])
     
#  print("Training data shape:", x_train.shape, y_train.shape)
#  print("Testing data shape:", x_test.shape, y_test.shape)
#  x_train=np.array(x_train)
#  scaler = preprocessing.StandardScaler()
#  scaler.fit(x_train)
#  x_train = scaler.transform(x_train).astype(np.float32)
#  x_test = scaler.transform(x_test).astype(np.float32)

#  train_loader, test_loader = create_data_loaders_from_arrays(torch.tensor([item.cpu().detach().numpy() for item in x_train]).cuda(), \
#  y_train, torch.from_numpy(x_test), y_test)
#  train_loader, test_loader = create_data_loaders_from_arrays(torch.from_numpy(x_train), \
#  y_train, torch.from_numpy(x_test), y_test)

 criterion = get_loss()
 classifier = pointnet2_cls_msg_concat.get_model(num_class=40,normal_channel=True).cuda()
 optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
 eval_every_n_epochs = 1
 scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
 best_instance_acc = 0.0
 best_class_acc = 0.0
 mean_correct = []


 for epoch in range(10):
       print('Epoch %d ' % ( epoch + 1))

       scheduler.step()
       for batch_id, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
           points, target = data
           points = points.data.numpy()
           points = provider.random_point_dropout(points)
           points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
           points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
           points = torch.Tensor(points)
           target = target[:, 0]

           points = points.transpose(2, 1)
           points, target = points.cuda(), target.cuda()
           optimizer.zero_grad()

           classifier = classifier.train()
           feature_vector,cls = encoder(points)
           pred,cls = classifier(points,feature_vector)
           loss = criterion(pred, target.long(), target)
           pred_choice = pred.data.max(1)[1]
           correct = pred_choice.eq(target.long().data).cpu().sum()
           mean_correct.append(correct.item() / float(points.size()[0]))
           loss.backward()
           optimizer.step()
       train_instance_acc = np.mean(mean_correct)
       print('Train Instance Accuracy: %f' % train_instance_acc)


       
 print('End of training...')  


 
#  train = torch.utils.data.TensorDataset(pred1, target)
#  train_loader = torch.utils.data.DataLoader(train, batch_size=96, shuffle=True)
#  for epoch in range(10):
#     train_acc = []
#     for x, y in train_loader:
#         x = x.to(device)
#         y = y.to(device)
#         # zero the parameter gradients
#         optimizer.zero_grad()    
#         classifier = classifier.train()
#         pred = classifier(x)    
#         predictions = torch.argmax(pred, dim=1)
#         loss = criterion(pred, y.long(),y)
#         loss.backward(retain_graph=True)
#         optimizer.step()
    
    
#     if epoch % eval_every_n_epochs == 0:
#         train_total,total = 0,0
#         train_correct,correct = 0,0
#         for x, y in train_loader:
#             x = x.to(device)
#             y = y.to(device)
#             classifier = classifier.train()
#             pred = classifier(x) 
#             predictions = torch.argmax(pred, dim=1)
            
#             train_total += y.size(0)
#             train_correct += (predictions == y).sum().item()
#         for x, y in test_loader:
#             x = x.to(device)
#             y = y.to(device)
            
#             classifier = classifier.train()
#             pred = classifier(x) 
#             predictions = torch.argmax(pred, dim=1)
            
#             total += y.size(0)
#             correct += (predictions == y).sum().item()
#         train_acc=  train_correct / train_total 
#         acc =  correct / total
#         print(f"Training accuracy: {np.mean(train_acc)}")
#         print(f"Testing accuracy: {np.mean(acc)}")
 return train_acc
