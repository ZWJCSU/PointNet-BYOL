
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms
from models.mlp_head import MLPHead
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
import os
import logging
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import _create_model_training_folder


class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter()
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "main.py", 'trainer.py'])

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        loss=0
        for i in range(len(x)):
            loss+=abs(x[i]-y[i])
        return loss

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
    def train_pointnet(self,trainDataLoader):
        def log_string(str):
            logger.info(str)
            print(str)

        '''LOG'''
        logger = logging.getLogger("Model")
        logger.setLevel(logging.INFO)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.7)
        global_epoch = 0
        global_step = 0
        best_instance_acc = 0.0
        best_class_acc = 0.0
        mean_correct = []
       
        for epoch in range(0,self.max_epochs):
            log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, self.max_epochs))

            scheduler.step()
            for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
               points, target = data
               points = points.data.numpy()
               points = provider.random_point_dropout(points)
               points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
               points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
               points = torch.Tensor(points)
               target = target[:, 0]

               points = points.transpose(2, 1)
               points, target = points.cuda(), target.cuda()
               loss = self.update(points, points)
               self.optimizer.zero_grad()
               loss.backward()
               self.optimizer.step()

               self._update_target_network_parameters() 
              #  classifier = classifier.train()
              #  pred, trans_feat = classifier(points)
              #  loss = criterion(pred, target.long(), trans_feat)
              #  pred_choice = pred.data.max(1)[1]
              #  correct = pred_choice.eq(target.long().data).cpu().sum()
              #  mean_correct.append(correct.item() / float(points.size()[0]))
               

              


            with torch.no_grad():
               instance_acc, class_acc = test(classifier.eval(), testDataLoader)

               if (instance_acc >= best_instance_acc):
                   best_instance_acc = instance_acc
                   best_epoch = epoch + 1

               if (class_acc >= best_class_acc):
                   best_class_acc = class_acc
               log_string('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
               log_string('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

               if (instance_acc >= best_instance_acc):
                   logger.info('Save model...')
                   savepath = str(checkpoints_dir) + '/best_model.pth'
                   log_string('Saving at %s'% savepath)
                   state = {
                       'epoch': best_epoch,
                       'instance_acc': instance_acc,
                       'class_acc': class_acc,
                       'model_state_dict': classifier.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                   }
                   torch.save(state, savepath)
               global_epoch += 1



 
    
    
    
    
    def train(self, train_dataset):
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()

        for epoch_counter in range(self.max_epochs):

            for (batch_view_1), _ in train_loader:

                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_1.to(self.device)

                if niter == 0:
                    grid = torchvision.utils.make_grid(batch_view_1[:32])
#                     self.writer.add_image('views_1', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(batch_view_2[:32])
#                     self.writer.add_image('views_2', grid, global_step=niter)

                loss = self.update(batch_view_1, batch_view_2)
                self.writer.add_scalar('loss', loss, global_step=niter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network_parameters()  # update the key encoder
                niter += 1

            print("End of epoch {}".format(epoch_counter))

        # save checkpoints
        self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))

        
        
    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        online_net=self.online_network.train()
        pred1, trans_feat1=online_net(batch_view_1)
        pred2, trans_feat2=online_net(batch_view_2)
        predictor=self.predictor.train()
        predictions_from_view_1 = predictor(pred1)
        predictions_from_view_2 = predictor(pred2)
        # pred_choice = pred1.data.max(1)[1]
        # correct = pred_choice.eq(target.long().data).cpu().sum()
        # mean_correct.append(correct.item() / float(points.size()[0]))
        # train_instance_acc = np.mean(mean_correct)
        # log_string('Train Instance Accuracy: %f' % train_instance_acc)
        # compute key features
        with torch.no_grad():
            target_network=self.target_network.train()
            targets_to_view_2,trans_feat_target_1 = target_network(batch_view_1)
            targets_to_view_1,trans_feat_target_2 = target_network(batch_view_2)
        
        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)
