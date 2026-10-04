"""DI-VNN utility functions for model training, prediction, and explainability.

Contains: ontonet, OntoDataset, train_model, best_configuration,
model_predict, model_grad_cam.
Ported from colab_uti.ipynb.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils import resample
# from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from joblib import dump, parallel_backend, load
import os
# os.environ['CUDA_LAUNCH_BLOCKING']='1'
import matplotlib.pyplot as plt

import math
import regex as re
from divnn import utils, TidySet
from divnn.ExpressionSet import *
from dfply import X, mutate, mask, select, rename, left_join
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.utils.data import random_split
from torch.utils.data import Subset
import torch.optim as optim
from progressbar import ProgressBar
from tqdm.notebook import tqdm
# import gc

import shap
#from lime.lime_tabular import LimeTabularExplainer

class ontonet(nn.Module):
  
  """
  Make an ontonet generator for deep-insight visible neural network (DI-VNN) 
  modeling

  This function creates a function that generates a PyTorch Convolutional Neural
  Network (CNN) model with a specific layer architecture for each path in the
  hierarchy of the given ontology.
  :param TidySet: TidySet, an ExpressionSet with three tables.
  :param path: A character of file path if the model json file is saved.
  :param init_seed: An integer of random seed for ReLU initializer.
  :param init2_seed: An integer of random seed for tanh initializer.
  :param l2_norm: A floating number of L2-norm regularization factor.
  :return: output Pytorch model object, a pointer to Pytorch model object in 
  python environment, which will be an input to train DI-VNN model using 
  Pytorch.
  """
  
  def __init__(self
               ,TidySet
               ,device='cpu'
               ,init_seed=888
               ,init2_seed=9999
               ,l2_norm=0
               ,output_unit=1
               ,output_activation='sigmoid'):
    
    super(ontonet,self).__init__()
    
    # Recall ontomap
    ontomap=notes(TidySet.experimentData)['ontomap']
    
    # Recall ontotype
    ontotype=notes(TidySet.experimentData)['ontotype']
    
    # Recall ontology
    ontology=notes(TidySet.experimentData)['ontology']
    
    def namer(name,suffix):
      if not name:
        return suffix
      else:
        return name+'_'+suffix
    
    # Build a class to insert an separable convolution 2D layer with initializer
    class CustomSeparableConv2d(nn.Module):
      def __init__(self
                   ,in_channels
                   ,out_channels
                   ,kernel_size
                   ,padding
                   ,depthwise_initializer
                   ,pointwise_initializer
                   ,name=None):
        
        super(CustomSeparableConv2d,self).__init__()
        
        if isinstance(kernel_size,int):
          kernel_size=(kernel_size,kernel_size)
        
        if isinstance(padding,int):
            padding=(padding,padding)
        
        self.depthwise=nn.Conv2d(
            in_channels
            ,in_channels
            ,kernel_size=kernel_size
            ,padding=padding
            ,groups=in_channels
            ,bias=False
          )
        self.pointwise=nn.Conv2d(
            in_channels
            ,out_channels
            ,kernel_size=1
            ,bias=False
          )
        
        self.init_weights(
            depthwise_initializer
            ,pointwise_initializer
          )
        
        self.depthwise_name=namer(name,'depthwise')
        self.pointwise_name=namer(name,'pointwise')
        self.to(device)
  
      def init_weights(self
                       ,depthwise_initializer
                       ,pointwise_initializer):
        
        depthwise_initializer(self.depthwise.weight)
        pointwise_initializer(self.pointwise.weight)
      
      def forward(self,x):
        x=self.depthwise(x)
        x=self.pointwise(x)
        return x
    
    # Build a function to insert a linear layer with initializer
    class CustomLinear(nn.Module):
      def __init__(self
                   ,in_channels
                   ,out_channels
                   ,kernel_initializer
                   ,name=None):
        
        super(CustomLinear,self).__init__()
        
        self.linear=nn.Linear(in_channels,out_channels)
        self.init_weights(kernel_initializer)
        self.linear_name=namer(name,'linear')
        self.to(device)
      
      def init_weights(self
                       ,kernel_initializer):
        
        kernel_initializer(self.linear.weight)
      
      def forward(self,x):
        x=self.linear(x)
        return x
    
    class CustomAdaptiveAvgPool2d(nn.Module):
      def __init__(self,output_size,name=None):
        
        super(CustomAdaptiveAvgPool2d,self).__init__()
        
        self.output_size=output_size
        self.adaptiveavgpool_name=namer(name,'adaptiveavgpool')
        self.to(device)
      
      def forward(self,x):
        return F.adaptive_avg_pool2d(x,self.output_size)
    
    # Build a function to insert an inception module along with a pre-activation residual unit
    class layer_inception_resnet(nn.Module):
      def __init__(self
                   ,filters
                   ,kernel_initializer
                   ,name=None):
        
        super(layer_inception_resnet,self).__init__()
        
        self.filters=filters
        self.kernel_initializer=kernel_initializer
        self.name=name
        self.in_channels=None
        self.layers=None
      
      def _initialize_layers(self
                             ,filters
                             ,kernel_initializer
                             ,name
                             ,in_channels):
        
        self.layers=nn.ModuleDict({
          
          namer(name,'pre_bn'):nn.BatchNorm2d(
              in_channels
              ,eps=1e-05
              ,momentum=0.1
              ,affine=True
              ,track_running_stats=True
            )
          ,namer(name,'pre_ac'):nn.ReLU(
              inplace=True
            )
          
          ,namer(name,'tower1_cv'):CustomSeparableConv2d(
              in_channels
              ,filters
              ,kernel_size=(1,1)
              ,padding=(0,0)
              ,depthwise_initializer=kernel_initializer
              ,pointwise_initializer=kernel_initializer
              ,name=namer(name,'tower1_cv')
            )
          ,namer(name,'tower1_bn'):nn.BatchNorm2d(
              filters
              ,eps=1e-05
              ,momentum=0.1
              ,affine=True
              ,track_running_stats=True
            )
          ,namer(name,'tower1_ac'):nn.ReLU(
              inplace=True
            )
          
          ,namer(name,'tower2a_mp'):nn.MaxPool2d(
              kernel_size=3
              ,stride=1
              ,padding=1#0
            )
          ,namer(name,'tower2b_cv'):CustomSeparableConv2d(
              in_channels
              ,filters
              ,kernel_size=(1,1)
              ,padding=(0,0)
              ,depthwise_initializer=kernel_initializer
              ,pointwise_initializer=kernel_initializer
              ,name=namer(name,'tower2b_cv')
            )
          ,namer(name,'tower2b_bn'):nn.BatchNorm2d(
              filters
              ,eps=1e-05
              ,momentum=0.1
              ,affine=True
              ,track_running_stats=True
            )
          ,namer(name,'tower2b_ac'):nn.ReLU(
              inplace=True
            )
          
          ,namer(name,'tower3a_cv'):CustomSeparableConv2d(
              in_channels
              ,filters
              ,kernel_size=(1,1)
              ,padding=(0,0)
              ,depthwise_initializer=kernel_initializer
              ,pointwise_initializer=kernel_initializer
              ,name=namer(name,'tower3a_cv')
            )
          ,namer(name,'tower3a_bn'):nn.BatchNorm2d(
              filters
              ,eps=1e-05
              ,momentum=0.1
              ,affine=True
              ,track_running_stats=True
            )
          ,namer(name,'tower3a_ac'):nn.ReLU(
              inplace=True
            )
          ,namer(name,'tower3b1_cv'):CustomSeparableConv2d(
              filters#in_channels
              ,filters
              ,kernel_size=(1,3)
              ,padding=(0,1)
              ,depthwise_initializer=kernel_initializer
              ,pointwise_initializer=kernel_initializer
              ,name=namer(name,'tower3b1_cv')
            )
          ,namer(name,'tower3b1_bn'):nn.BatchNorm2d(
              filters
              ,eps=1e-05
              ,momentum=0.1
              ,affine=True
              ,track_running_stats=True
            )
          ,namer(name,'tower3b1_ac'):nn.ReLU(
              inplace=True
            )
          
          ,namer(name,'tower3b2_cv'):CustomSeparableConv2d(
              filters#in_channels
              ,filters
              ,kernel_size=(3,1)
              ,padding=(1,0)
              ,depthwise_initializer=kernel_initializer
              ,pointwise_initializer=kernel_initializer
              ,name=namer(name,'tower3b2_cv')
            )
          ,namer(name,'tower3b2_bn'):nn.BatchNorm2d(
              filters
              ,eps=1e-05
              ,momentum=0.1
              ,affine=True
              ,track_running_stats=True
            )
          ,namer(name,'tower3b2_ac'):nn.ReLU(
              inplace=True
            )
          
          ,namer(name,'tower4a_cv'):CustomSeparableConv2d(
              in_channels
              ,filters
              ,kernel_size=(1,1)
              ,padding=(0,0)
              ,depthwise_initializer=kernel_initializer
              ,pointwise_initializer=kernel_initializer
              ,name=namer(name,'tower4a_cv')
            )
          ,namer(name,'tower4a_bn'):nn.BatchNorm2d(
              filters
              ,eps=1e-05
              ,momentum=0.1
              ,affine=True
              ,track_running_stats=True
            )
          ,namer(name,'tower4a_ac'):nn.ReLU(
              inplace=True
            )
          
          ,namer(name,'tower4b_cv'):CustomSeparableConv2d(
              filters#in_channels
              ,filters
              ,kernel_size=(3,3)
              ,padding=(1,1)
              ,depthwise_initializer=kernel_initializer
              ,pointwise_initializer=kernel_initializer
              ,name=namer(name,'tower4b_cv')
            )
          ,namer(name,'tower4b_bn'):nn.BatchNorm2d(
              filters
              ,eps=1e-05
              ,momentum=0.1
              ,affine=True
              ,track_running_stats=True
            )
          ,namer(name,'tower4b_ac'):nn.ReLU(
              inplace=True
            )
          
          ,namer(name,'tower4c1_cv'):CustomSeparableConv2d(
              filters#in_channels
              ,filters
              ,kernel_size=(1,3)
              ,padding=(0,1)
              ,depthwise_initializer=kernel_initializer
              ,pointwise_initializer=kernel_initializer
              ,name=namer(name,'tower4c1_cv')
            )
          ,namer(name,'tower4c1_bn'):nn.BatchNorm2d(
              filters
              ,eps=1e-05
              ,momentum=0.1
              ,affine=True
              ,track_running_stats=True
            )
          ,namer(name,'tower4c1_ac'):nn.ReLU(
              inplace=True
            )
          
          ,namer(name,'tower4c2_cv'):CustomSeparableConv2d(
              filters#in_channels
              ,filters
              ,kernel_size=(3,1)
              ,padding=(1,0)
              ,depthwise_initializer=kernel_initializer
              ,pointwise_initializer=kernel_initializer
              ,name=namer(name,'tower4c2_cv')
            )
          ,namer(name,'tower4c2_bn'):nn.BatchNorm2d(
              filters
              ,eps=1e-05
              ,momentum=0.1
              ,affine=True
              ,track_running_stats=True
            )
          ,namer(name,'tower4c2_ac'):nn.ReLU(
              inplace=True
            )
          
          ,namer(name,'sc_cv'):CustomSeparableConv2d(
              filters*6
              ,in_channels
              ,kernel_size=(1,1)
              ,padding=(0,0)
              ,depthwise_initializer=kernel_initializer
              ,pointwise_initializer=kernel_initializer
              ,name=namer(name,'sc_cv')
            )
          
        })
        
        self.to(device)
      
      def forward(self,x,residue):
        if self.layers is None:
          self.in_channels=residue.shape[1]
          self._initialize_layers(
              self.filters
              ,self.kernel_initializer
              ,self.name
              ,self.in_channels
            )
        
        pre_activation=self.layers[
            namer(self.name,'pre_bn')
          ](x)
        pre_activation=self.layers[
            namer(self.name,'pre_ac')
          ](pre_activation)
        
        tower_1=self.layers[
            namer(self.name,'tower1_cv')
          ](pre_activation)
        tower_1=self.layers[
            namer(self.name,'tower1_bn')
          ](tower_1)
        tower_1=self.layers[
            namer(self.name,'tower1_ac')
          ](tower_1)
        
        tower_2=self.layers[
            namer(self.name,'tower2a_mp')
          ](pre_activation)
        tower_2=self.layers[
            namer(self.name,'tower2b_cv')
          ](tower_2)
        tower_2=self.layers[
            namer(self.name,'tower2b_bn')
          ](tower_2)
        tower_2=self.layers[
            namer(self.name,'tower2b_ac')
          ](tower_2)
        
        tower_3a=self.layers[
            namer(self.name,'tower3a_cv')
          ](pre_activation)
        tower_3a=self.layers[
            namer(self.name,'tower3a_bn')
          ](tower_3a)
        tower_3a=self.layers[
            namer(self.name,'tower3a_ac')
          ](tower_3a)
        tower_3b1=self.layers[
            namer(self.name,'tower3b1_cv')
          ](tower_3a)
        tower_3b1=self.layers[
            namer(self.name,'tower3b1_bn')
          ](tower_3b1)
        tower_3b1=self.layers[
            namer(self.name,'tower3b1_ac')
          ](tower_3b1)
        tower_3b2=self.layers[
            namer(self.name,'tower3b2_cv')
          ](tower_3a)
        tower_3b2=self.layers[
            namer(self.name,'tower3b2_bn')
          ](tower_3b2)
        tower_3b2=self.layers[
            namer(self.name,'tower3b2_ac')
          ](tower_3b2)
        
        tower_4a=self.layers[
            namer(self.name,'tower4a_cv')
          ](pre_activation)
        tower_4a=self.layers[
            namer(self.name,'tower4a_bn')
          ](tower_4a)
        tower_4a=self.layers[
            namer(self.name,'tower4a_ac')
          ](tower_4a)
        tower_4b=self.layers[
            namer(self.name,'tower4b_cv')
          ](tower_4a)
        tower_4b=self.layers[
            namer(self.name,'tower4b_bn')
          ](tower_4b)
        tower_4b=self.layers[
            namer(self.name,'tower4b_ac')
          ](tower_4b)
        tower_4c1=self.layers[
            namer(self.name,'tower4c1_cv')
          ](tower_4b)
        tower_4c1=self.layers[
            namer(self.name,'tower4c1_bn')
          ](tower_4c1)
        tower_4c1=self.layers[
            namer(self.name,'tower4c1_ac')
          ](tower_4c1)
        tower_4c2=self.layers[
            namer(self.name,'tower4c2_cv')
          ](tower_4b)
        tower_4c2=self.layers[
            namer(self.name,'tower4c2_bn')
          ](tower_4c2)
        tower_4c2=self.layers[
            namer(self.name,'tower4c2_ac')
          ](tower_4c2)
        
        towers=torch.cat(
            [tower_1,tower_2,tower_3b1,tower_3b2,tower_4c1,tower_4c2]
            ,dim=1
          )
        
        scaling=self.layers[
            namer(self.name,'sc_cv')
          ](towers)
        
        inception_resnet=scaling+residue
        
        return inception_resnet
    
    # Build a function to insert auxiliary output layers
    class layer_aux_output(nn.Module):
      def __init__(self
                   ,filters
                   ,units
                   ,kernel_initializer
                   ,kernel_initializer2
                   ,l2_norm=None
                   ,output_unit=1
                   ,output_activation='sigmoid'
                   ,name=None):
        
        super(layer_aux_output,self).__init__()
        
        self.filters=filters
        self.units=units
        self.kernel_initializer=kernel_initializer
        self.kernel_initializer2=kernel_initializer2
        self.l2_norm=l2_norm
        self.output_unit=output_unit
        self.output_activation=output_activation
        self.name=name
        self.in_channels=None
        self.layers=None
      
      def _initialize_layers(self
                             ,filters
                             ,units
                             ,kernel_initializer
                             ,kernel_initializer2
                             ,l2_norm
                             ,output_unit
                             ,output_activation
                             ,name
                             ,in_channels):
        
        activation_layer=None
        if output_activation=='sigmoid':
          activation_layer=nn.Sigmoid()
        elif output_activation=='tanh':
          activation_layer=nn.Tanh()
        elif output_activation=='linear':
          activation_layer=nn.Identity()
        
        self.layers=nn.ModuleDict({
          
          namer(name,'ap'):CustomAdaptiveAvgPool2d(1)
          
          ,namer(name,'hl1_cv'):CustomSeparableConv2d(
              in_channels
              ,filters
              ,kernel_size=(1,1)
              ,padding=(0,0)
              ,depthwise_initializer=kernel_initializer
              ,pointwise_initializer=kernel_initializer
              ,name=namer(name,'hl1_cv')
            )
          ,namer(name,'hl1_bn'):nn.BatchNorm2d(
              filters
              ,eps=1e-05
              ,momentum=0.1
              ,affine=True
              ,track_running_stats=True
            )
          ,namer(name,'hl1_ac'):nn.ReLU(
              inplace=True
            )
          ,namer(name,'hl1_fl'):nn.Flatten()
          
          ,namer(name,'hl2_de'):CustomLinear(
              filters
              ,units
              ,kernel_initializer
              ,name=namer(name,'hl2_de')
            )
          ,namer(name,'hl2_bn'):nn.BatchNorm1d(
              units
              ,eps=1e-05
              ,momentum=0.1
              ,affine=True
              ,track_running_stats=True
            )
          ,namer(name,'hl2_ac'):nn.ReLU(
              inplace=True
            )
          
          ,namer(name,'hl3_de'):CustomLinear(
              units
              ,units
              ,kernel_initializer
              ,name=namer(name,'hl3_de')
            )
          ,namer(name,'hl3_bn'):nn.BatchNorm1d(
              units
              ,eps=1e-05
              ,momentum=0.1
              ,affine=True
              ,track_running_stats=True
            )
          ,namer(name,'hl3_ac'):nn.ReLU(
              inplace=True
            )
          
          ,namer(name,'ao_de'):CustomLinear(
              units
              ,2
              ,kernel_initializer2
              ,name=namer(name,'ao_de')
            )
          ,namer(name,'ao_tn'):nn.Tanh()
          ,namer(name,'ao_bn'):nn.BatchNorm1d(
              2
              ,eps=1e-05
              ,momentum=0.1
              ,affine=True
              ,track_running_stats=True
            )
          ,namer(name,'aux_output_de'):CustomLinear(
              2
              ,output_unit
              ,kernel_initializer2
              ,name=namer(name,'aux_output_de')
            )
          ,namer(name,'aux_output'):activation_layer
        })
        
        self.l2_norm=l2_norm
      
      def forward(self,x):
        if self.layers is None:
          self.in_channels=x.shape[1]
          self._initialize_layers(
              self.filters
              ,self.units
              ,self.kernel_initializer
              ,self.kernel_initializer2
              ,self.l2_norm
              ,self.output_unit
              ,self.output_activation
              ,self.name
              ,self.in_channels
            )
        
        self.to(device)
        
        aux_output=self.layers[
            namer(self.name,'ap')
          ](x)
        
        aux_output=self.layers[
            namer(self.name,'hl1_cv')
          ](aux_output)
        aux_output=self.layers[
            namer(self.name,'hl1_bn')
          ](aux_output)
        aux_output=self.layers[
            namer(self.name,'hl1_ac')
          ](aux_output)
        aux_output=self.layers[
            namer(self.name,'hl1_fl')
          ](aux_output.view(aux_output.size(0),-1))
        
        aux_output=self.layers[
            namer(self.name,'hl2_de')
          ](aux_output)
        aux_output=self.layers[
            namer(self.name,'hl2_bn')
          ](aux_output)
        aux_output=self.layers[
            namer(self.name,'hl2_ac')
          ](aux_output)
        
        aux_output=self.layers[
            namer(self.name,'hl3_de')
          ](aux_output)
        aux_output=self.layers[
            namer(self.name,'hl3_bn')
          ](aux_output)
        aux_output=self.layers[
            namer(self.name,'hl3_ac')
          ](aux_output)
        
        aux_output=self.layers[
            namer(self.name,'ao_de')
          ](aux_output)
        aux_output=self.layers[
            namer(self.name,'ao_tn')
          ](aux_output)
        aux_output=self.layers[
            namer(self.name,'ao_bn')
          ](aux_output)
        aux_output=self.layers[
            namer(self.name,'aux_output_de')
          ](aux_output)
        aux_output=self.layers[
            namer(self.name,'aux_output')
          ](aux_output)  
        
        return aux_output
      
      def l2_regularization(self):
        l2_reg=None
        for param in self.parameters():
          if l2_reg is None:
            l2_reg=0.5*self.l2_norm*torch.norm(param,2)**2
          else:
            l2_reg=l2_reg+0.5*self.l2_norm*torch.norm(param,2)**2
        return l2_reg
    
    # Build a function to insert output layers
    class layer_output(nn.Module):
      def __init__(self
                   ,units
                   ,kernel_initializer
                   ,kernel_initializer2
                   ,l2_norm=None
                   ,output_unit=1
                   ,output_activation='sigmoid'
                   ,name=None):
        
        super(layer_output,self).__init__()
        
        self.units=units
        self.kernel_initializer=kernel_initializer
        self.kernel_initializer2=kernel_initializer2
        self.l2_norm=l2_norm
        self.output_unit=output_unit
        self.output_activation=output_activation
        self.name=name
        self.in_channels=None
        self.layers=None
      
      def _initialize_layers(self
                             ,units
                             ,kernel_initializer
                             ,kernel_initializer2
                             ,l2_norm
                             ,output_unit
                             ,output_activation
                             ,name
                             ,in_channels):
        
        activation_layer=None
        if output_activation=='sigmoid':
          activation_layer=nn.Sigmoid()
        elif output_activation=='tanh':
          activation_layer=nn.Tanh()
        elif output_activation=='linear':
          activation_layer=nn.Identity()
        
        self.layers=nn.ModuleDict({
          
          namer(name,'ap'):CustomAdaptiveAvgPool2d(1)
          ,namer(name,'ap_fl'):nn.Flatten()
          
          ,namer(name,'hl1_de'):CustomLinear(
              in_channels
              ,units
              ,kernel_initializer
              ,name=namer(name,'hl1_de')
            )
          ,namer(name,'hl1_bn'):nn.BatchNorm1d(
              units
              ,eps=1e-05
              ,momentum=0.1
              ,affine=True
              ,track_running_stats=True
            )
          ,namer(name,'hl1_ac'):nn.ReLU(
              inplace=True
            )
          
          ,namer(name,'mo_de'):CustomLinear(
              units
              ,2
              ,kernel_initializer2
              ,name=namer(name,'mo_de')
            )
          ,namer(name,'mo_tn'):nn.Tanh()
          ,namer(name,'mo_bn'):nn.BatchNorm1d(
              2
              ,eps=1e-05
              ,momentum=0.1
              ,affine=True
              ,track_running_stats=True
            )
          ,namer(name,'de'):CustomLinear(
              2
              ,output_unit
              ,kernel_initializer2
              ,name=namer(name,'de')
            )
          ,name:activation_layer
        })
        
        self.l2_norm=l2_norm
      
      def forward(self,x):
        if self.layers is None:
          self.in_channels=x.shape[1]
          self._initialize_layers(
              self.units
              ,self.kernel_initializer
              ,self.kernel_initializer2
              ,self.l2_norm
              ,self.output_unit
              ,self.output_activation
              ,self.name
              ,self.in_channels
            )
        
        self.to(device)
        
        output=self.layers[
            namer(self.name,'ap')
          ](x)
        output=self.layers[
            namer(self.name,'ap_fl')
          ](output.view(output.size(0),-1))
        
        output=self.layers[
            namer(self.name,'hl1_de')
          ](output)
        output=self.layers[
            namer(self.name,'hl1_bn')
          ](output)
        output=self.layers[
            namer(self.name,'hl1_ac')
          ](output)
        
        output=self.layers[
            namer(self.name,'mo_de')
          ](output)
        output=self.layers[
            namer(self.name,'mo_tn')
          ](output)
        output=self.layers[
            namer(self.name,'mo_bn')
          ](output)
        output=self.layers[
            namer(self.name,'de')
          ](output)
        output=self.layers[
            self.name
          ](output) 
        
        return output
      
      def l2_regularization(self):
        l2_reg=None
        for param in self.parameters():
          if l2_reg is None:
            l2_reg=0.5*self.l2_norm*torch.norm(param,2)**2
          else:
            l2_reg=l2_reg+0.5*self.l2_norm*torch.norm(param,2)**2
        return l2_reg
    
    # Build a generator function to construct ontonet
    def init(tensor):
        torch.manual_seed(init_seed)
        return nn.init.kaiming_uniform_(tensor, nonlinearity='relu')
    
    def init2(tensor):
        torch.manual_seed(init2_seed)
        return nn.init.xavier_uniform_(tensor)
    
    feature=ontology.copy()
    while any('ONT' in s for s in feature.source):
      Z=pd.DataFrame()
      for i in np.arange(feature.shape[0]):
        if 'ONT' in feature.source[i]:
          Y=feature >> mask(X.target==feature.source[i],X.relation=='feature')
          if Y.shape[0]>0:
            Z=pd.concat([
                Z
                ,pd.DataFrame({
                    'source':Y.source
                    ,'target':feature.target[i]
                    ,'similarity':feature.similarity[i]
                    ,'relation':'feature'
                  })
            ])
          else:
            Z=pd.concat([Z,feature.iloc[[i]]])
        else:
          Z=pd.concat([Z,feature.iloc[[i]]])
      Z=Z >> select(X.source,X.target,X.similarity,X.relation)
      feature=Z
      feature=feature.reset_index(inplace=False)
    
    Y=feature >> select(X.source,X.target)
    Z=feature >> select(X.source)
    Z=Z.drop_duplicates()
    Z=Z >> mutate(target='root')
    feature=pd.concat([Y,Z])
    
    Y=feature.target.drop_duplicates()
    Z=pd.DataFrame()
    for i in np.arange(Y.shape[0]):
      K=feature >> mask(X.target==Y.iloc[i])
      Z=pd.concat([Z,pd.DataFrame.from_dict({'target':[Y.iloc[i]],'n':[K.shape[0]]})])
    Z.index=np.arange(Y.shape[0])
    Y=[]
    for i in Z.target.to_numpy():
      Y.append(re.sub('ONT\\:','ONT',i))
    feature=Z >> mutate(oid=Y) >> select(X.oid,X.n)
    del Y,Z,K
    
    I=[]
    for i in np.arange(ontology.shape[0]):
      if 'ONT:' in ontology.source[i]: I.append(i)
    Y=[]
    for i in ontology.iloc[I,].source.to_numpy():
      Y.append(re.sub('ONT\\:','ONT',i))
    Z=[]
    for i in ontology.iloc[I,].target.to_numpy():
      Z.append(re.sub('ONT\\:','ONT',i))
    
    Y=pd.DataFrame.from_dict({'from':Y,'to':Z})
    I=[]
    for i in Y['to']:
      if not i in ' '.join(Y['from']): I.append(i)
    Z=pd.DataFrame.from_dict({'from':I,'to':'root'})
    Z=Z.drop_duplicates()
    hierarchy=pd.concat([Y,Z])
    hierarchy.index=np.arange(hierarchy.shape[0])
    I=feature >> rename(to=X.oid)
    hierarchy=hierarchy >> left_join(I,by='to')
    del I
    
    self.feature=feature
    self.hierarchy=hierarchy
    
    self.terminal_nodes=[]
    for i in hierarchy['from'].drop_duplicates():
      if not i in ' '.join(hierarchy['to']): self.terminal_nodes.append(i)
    
    self.non_terminal_nodes=hierarchy['to'].drop_duplicates().to_list()
    
    pb=ProgressBar(1+len(self.terminal_nodes)+len(self.non_terminal_nodes))
    tick=0
    pb.start()
    
    self.ontoarray_shape=ontomap.shape[1:4]
        
    self.ontofilters=nn.ParameterDict()
    for key in ontotype.keys():
        Z=torch.zeros_like(torch.from_numpy(ontomap[0]))
        for i in range(len(ontotype[key])):
            x=int(ontotype[key]['x'].values[i])-1
            y=int(ontotype[key]['y'].values[i])-1
            z=int(ontotype[key]['z'].values[i])-1
            Z[x,y,z]=1
        self.ontofilters[key]=Parameter(
            Z.float()
            ,requires_grad=False
          )
    
    # self.inputs=nn.ModuleDict()
    self.hiddens=nn.ModuleDict()
    self.outputs=nn.ModuleDict()

    for A in self.terminal_nodes:
      tick+=1
      pb.update(tick)
      
      B=A
      C=self.feature.n[self.feature.oid==A].values.tolist()[0]
      
      self.hiddens[A]=layer_inception_resnet(
        filters=np.max([20,math.ceil(0.3*C)])
        ,kernel_initializer=init
        ,name=namer(A,'hidden')
      )
  
      self.outputs[A]=layer_aux_output(
        filters=np.max([20,math.ceil(0.3*C)])
        ,units=np.max([20,math.ceil(0.3*C)])
        ,kernel_initializer=init
        ,kernel_initializer2=init2
        ,l2_norm=l2_norm
        ,output_unit=output_unit
        ,output_activation=output_activation
        ,name=A
      )
    
    for A in self.non_terminal_nodes:
      tick+=1
      pb.update(tick)
      
      B=hierarchy['from'][hierarchy['to']==A].values.tolist()
      C=self.feature.n[self.feature.oid==A].values.tolist()[0]
      
      self.hiddens[A]=layer_inception_resnet(
        filters=np.max([20,math.ceil(0.3*C)])
        ,kernel_initializer=init
        ,name=namer(A,'hidden')
      )
      
      if A!=self.non_terminal_nodes[len(self.non_terminal_nodes)-1]:
        self.outputs[A]=layer_aux_output(
          filters=np.max([20,math.ceil(0.3*C)])
          ,units=np.max([20,math.ceil(0.3*C)])
          ,kernel_initializer=init
          ,kernel_initializer2=init2
          ,l2_norm=l2_norm
          ,output_unit=output_unit
          ,output_activation=output_activation
          ,name=A
        )
      else:
        self.outputs[A]=layer_output(
          units=np.max([20,math.ceil(0.3*C)])
          ,kernel_initializer=init
          ,kernel_initializer2=init2
          ,l2_norm=l2_norm
          ,output_unit=output_unit
          ,output_activation=output_activation
          ,name=A
        )
    
    self.to(device)
    
    tick+=1
    pb.update(tick)
  
  def forward(self,ontoarray):
    assert ontoarray.shape[1:]==self.ontoarray_shape,f"Input shape must be {self.ontoarray_shape}"
    
    inputs={}
    for key in self.ontofilters.keys():
      inputs[key]=ontoarray*self.ontofilters[key]
    
    hiddens={}
    outputs={}
    
    for A in self.terminal_nodes:
      B=A
      
      hiddens[A]=self.hiddens[A](inputs[B],inputs[A])
      outputs[A]=self.outputs[A](hiddens[A])
    
    for A in self.non_terminal_nodes:
      B=self.hierarchy['from'][self.hierarchy['to']==A].values.tolist()
      
      hiddens[A]=torch.cat(
        [hiddens[i] for i in B]
        ,dim=-1
      )
      hiddens[A]=self.hiddens[A](hiddens[A],inputs[A])
      
      outputs[A]=self.outputs[A](hiddens[A])
    
    return outputs

from torch.utils.data import Dataset, DataLoader

class OntoDataset(Dataset):
  def __init__(self,TidySet,indices,sample_weights=None):
    # Initialization code
    self.ontomap=notes(TidySet.experimentData)['ontomap'][indices,:]
    self.outcome=pData(TidySet).outcome[indices]
    self.ontotype=notes(TidySet.experimentData)['ontotype']
    self.sample_weights=sample_weights
    
  def __len__(self):
    # Return the total number of samples
    return self.ontomap.shape[0]
    
  def __getitem__(self, index):
    # Generate one sample of data
    x={'ontoarray':self.ontomap[index,:]}
    y={key: self.outcome[index] for key in self.ontotype}
    
    if self.sample_weights is not None:
      weight=self.sample_weights[index]
      return x,y,weight
    else:
      return x,y

def train_model(tidy_set_path
                ,save_dir=None
                ,l2_norm=0
                ,epochs=500
                ,patience=500/2
                ,batch_size=32
                ,warm_up=0.05
                ,lr=2e-6
                ,seed=None):
    
    # Load tidy set
    tidy_set=TidySet.read(tidy_set_path)
    tidy_set.desc()
    
    # Check if GPU is available and if not, fall back on CPU
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate ontonet
    uti_ontonet=ontonet(
        tidy_set
        ,device
        ,init_seed=seed
        ,init2_seed=seed
        ,l2_norm=l2_norm
        ,output_unit=1
        ,output_activation='sigmoid'
    )
    
    if save_dir is not None:
        # Create the directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save ontonet
        torch.save(uti_ontonet.state_dict(),f'{save_dir}null.pt')
    
    # Instantiate dataset
    train_indices=np.arange(notes(tidy_set.experimentData)['ontomap'].shape[0])
    train_dataset=OntoDataset(tidy_set,train_indices)
    
    # Define the size of validation set (20% of total data)
    val_size=int(0.2*len(train_dataset))
    train_size=len(train_dataset)-val_size
    
    # Split dataset
    train_dataset,val_dataset=random_split(train_dataset,[train_size,val_size])
    
    # Define DataLoader
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_dataloader=DataLoader(val_dataset,batch_size=batch_size,shuffle=True)
    
    # Determine outcome weights
    class_counts=pData(tidy_set)['outcome'].value_counts()  # Count the number of samples in each class
    total_samples=len(pData(tidy_set))  # Total number of samples
    
    weights={}
    
    for class_label,count in class_counts.items():
        p=count/total_samples  # Compute the proportion of samples in the class
        weight=1/(p*0.5)  # Compute the weight for the class
        weights[class_label]=weight

    weights=torch.Tensor([weights[class_label] for class_label in sorted(weights.keys())])
    
    # Define optimizer, criterion, and scheduler
    def criterions(input,target,weights):
        # Create a tensor for weights that is the same shape as target
        weight_map=torch.ones_like(target)
        for outcome,weight in enumerate(weights):
            weight_map[target==outcome]=weight
        # Compute weighted MSE loss
        return torch.sum(weight_map*(input-target)**2)/torch.sum(weight_map)

    optimizer=optim.SGD(
        uti_ontonet.parameters()
        ,lr=lr
        ,momentum=0.9
      )
    
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(
        optimizer
        ,factor=0.96
        ,patience=1
        ,verbose=False
        ,mode='max'
        ,threshold=0.01
        ,cooldown=0
        ,min_lr=lr/32
      )
    
    # Define custom batch scheduler for warm-up learning rate
    class WarmupScheduler(optim.lr_scheduler._LRScheduler):
      def __init__(self,optimizer,warm_up_steps,warm_up_factor=32,last_epoch=-1):
        self.warm_up_steps=warm_up_steps
        self.warm_up_factor=warm_up_factor
        super(WarmupScheduler,self).__init__(optimizer,last_epoch)
      
      def get_lr(self):
        if self.last_epoch<self.warm_up_steps:# Last epoch contextually means last batch
          return [base_lr/self.warm_up_factor*(self.last_epoch+1) for base_lr in self.base_lrs]
        else:
          return self.base_lrs

    # Define the number of warm up steps for each batch
    steps_per_epoch=len(train_dataloader)
    warm_up_steps=int(np.ceil(steps_per_epoch*warm_up))
    
    # Initialize early stopping parameters
    best_val_roc_auc=float('-inf')
    patience_counter=0
    min_delta=0.001
    
    num_bootstrap_iterations=30
    
    progress_bar=tqdm(
        total=(epochs*len(train_dataloader)+epochs*len(val_dataloader))
        ,desc='Modeling Progress'
        ,dynamic_ncols=True
        ,leave=False
    )
    
    print(f"epoch,subset,lr,total_loss,root_loss,roc_auc,tp,fn,fp,tn")
    history=pd.DataFrame(
        columns=[
            'epoch'
            ,'subset'
            ,'lr'
            ,'total_loss'
            ,'root_loss'
            ,'roc_auc'
            ,'tp'
            ,'fn'
            ,'fp'
            ,'tn'
          ]
      )
    
    for epoch in range(epochs):
        # Training
        uti_ontonet.train()  # Set the model to training mode
        train_true=[]
        train_pred=[]
        
        batch_scheduler=WarmupScheduler(optimizer,warm_up_steps)
        for batch, (inputs,targets) in enumerate(train_dataloader):
            optimizer.zero_grad()  # Reset gradients
            tensor_inputs=inputs['ontoarray'].to(device)  # Extract tensor from dictionary and move to GPU
            targets={key:value.to(device) for key,value in targets.items()}  # Move targets to GPU
            outputs_dict=uti_ontonet(tensor_inputs)    # Forward pass
            
            train_true.append(targets['root'].detach().cpu().numpy())
            train_pred.append(outputs_dict['root'].detach().cpu().numpy())
            
            w_total=0.3*(len(outputs_dict.keys())-1)+1
            w_nonroot=0.3/w_total
            w_root=1/w_total
            loss_weights={output_name: w_root if output_name=='root' else w_nonroot for output_name in outputs_dict.keys()}
            
            total_loss=0
            train_rloss=0
            for output_name in outputs_dict.keys():
                loss=criterions(outputs_dict[output_name],targets[output_name],weights)  # Compute loss
                total_loss+=loss_weights[output_name]*loss  # Weight the loss
                if output_name=='root':
                    train_rloss+=loss
            
            train_loss=total_loss/w_total
            
            train_loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            batch_scheduler.step()
            if batch<warm_up_steps:
                if (batch+1)==warm_up_steps:
                    print(f"\r{epoch+1},training,{optimizer.param_groups[0]['lr']:.5e},-,-,-,-,-,-,-",end='                               ')
                else:
                    print(f"\r{epoch+1},training,{optimizer.param_groups[0]['lr']:.5e},-,-,-,-,-,-,- <-- warm-up {batch+1}/{warm_up_steps}",end='')
            
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch: {epoch+1}, Training Batch")
        
        train_true=np.concatenate(train_true)
        train_pred=np.concatenate(train_pred)
        train_roc_aucs=[]
        for i in range(num_bootstrap_iterations):
            np.random.seed(i+1)  # Set the seed
            boot_true,boot_pred=resample(train_true,train_pred)
            train_roc_aucs.append(roc_auc_score(boot_true,boot_pred))
        
        train_roc_auc=np.mean(train_roc_aucs)
        
        train_true_bin=(train_true>0.5).astype(int)
        train_pred_bin=(train_pred>0.5).astype(int)
        train_tn,train_fp,train_fn,train_tp=confusion_matrix(train_true_bin,train_pred_bin).ravel()
        
        print(f"\r{epoch+1},training,{optimizer.param_groups[0]['lr']:.5e},{train_loss:.4f},{train_rloss:.4f},{train_roc_auc:.4f},{train_tp},{train_fn},{train_fp},{train_tn}")
        history=pd.concat([history,pd.DataFrame({
          'epoch':[epoch+1]
          ,'subset':['training']
          ,'lr':[optimizer.param_groups[0]['lr']]
          ,'total_loss':[train_loss.cpu().detach().numpy()]
          ,'root_loss':[train_rloss.cpu().detach().numpy()]
          ,'roc_auc':[train_roc_auc]
          ,'tp':[train_tp]
          ,'fn':[train_fn]
          ,'fp':[train_fp]
          ,'tn':[train_tn]
        })],ignore_index=True)
        
        # Validation
        uti_ontonet.eval()  # Set the model to evaluation mode
        val_true=[]
        val_pred=[]
        with torch.no_grad():  # No gradient calculation
            for inputs,targets in val_dataloader:
                tensor_inputs=inputs['ontoarray'].to(device)  # Extract tensor from dictionary and move to GPU
                targets={key:value.to(device) for key,value in targets.items()}  # Move targets to GPU
                outputs_dict=uti_ontonet(tensor_inputs)  # Forward pass
                
                val_true.append(targets['root'].detach().cpu().numpy())
                val_pred.append(outputs_dict['root'].detach().cpu().numpy())
                
                total_loss=0
                val_rloss=0
                for output_name in outputs_dict.keys():
                    loss=criterions(outputs_dict[output_name],targets[output_name],weights)  # Compute loss
                    total_loss+=loss_weights[output_name]*loss  # Weight the loss
                    if output_name=='root':
                        val_rloss+=loss
                
                val_loss=total_loss/w_total
                
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch: {epoch+1}, Validation Batch")
        
        val_true=np.concatenate(val_true)
        val_pred=np.concatenate(val_pred)
        val_roc_aucs=[]
        for i in range(num_bootstrap_iterations):
            np.random.seed(i+1)  # Set the seed
            boot_true,boot_pred=resample(val_true,val_pred)
            val_roc_aucs.append(roc_auc_score(boot_true,boot_pred))
        
        val_roc_auc=np.mean(val_roc_aucs)
        
        val_true_bin=(val_true>0.5).astype(int)
        val_pred_bin=(val_pred>0.5).astype(int)
        val_tn,val_fp,val_fn,val_tp=confusion_matrix(val_true_bin,val_pred_bin).ravel()
        
        print(f"{epoch+1},validation,{optimizer.param_groups[0]['lr']:.5e},{val_loss:.4f},{val_rloss:.4f},{val_roc_auc:.4f},{val_tp},{val_fn},{val_fp},{val_tn}")
        history=pd.concat([history,pd.DataFrame({
          'epoch':[epoch+1]
          ,'subset':['validation']
          ,'lr':[optimizer.param_groups[0]['lr']]
          ,'total_loss':[val_loss.cpu().detach().numpy()]
          ,'root_loss':[val_rloss.cpu().detach().numpy()]
          ,'roc_auc':[val_roc_auc]
          ,'tp':[val_tp]
          ,'fn':[val_fn]
          ,'fp':[val_fp]
          ,'tn':[val_tn]
        })],ignore_index=True)
        
        scheduler.step(val_roc_auc)

        history.to_csv(f'{save_dir}history.csv',index=False)
        
        # Check for early stopping
        if abs(val_roc_auc-best_val_roc_auc)<min_delta:
            patience_counter+=1
        else:
            patience_counter=0
        
        if patience_counter>=patience:
            print("Early stopping triggered.")
            break
        
        if val_roc_auc>best_val_roc_auc:
            best_val_roc_auc=val_roc_auc
            # Save model weights
            if save_dir is not None:
                torch.save(uti_ontonet.state_dict(),f'{save_dir}{epoch+1}.pt')

def best_configuration(training_auc_rocs,validation_auc_rocs,alpha=0.5):
    """
    Find the best configuration based on two criteria:
    1. Maximizing validation AUC-ROC
    2. Minimizing the absolute difference between training and validation AUC-ROC

    Parameters:
        training_auc_rocs (list): List of training AUC-ROC scores
        validation_auc_rocs (list): List of validation AUC-ROC scores
        alpha (float): Weight for the validation AUC-ROC in the combined score (default is 0.5)

    Returns:
        int: Index of the best configuration
    """
    pairs=[(train_auc,val_auc) for train_auc,val_auc in zip(training_auc_rocs,validation_auc_rocs)]
    
    # Calculate the differences and find the maximum difference
    differences=[abs(p[0]-p[1]) for p in pairs]
    max_diff=max(differences)
    
    # If all differences are zero, return the index of the maximum validation AUC-ROC
    if max_diff==0:
        return validation_auc_rocs.index(max(validation_auc_rocs))
    
    # Normalize the validation AUCs and differences
    normalized_val_aucs=[p[1]/max(p[1] for p in pairs) for p in pairs]
    normalized_diffs=[1-(diff/max_diff) for diff in differences]
    
    # Weights for the criteria
    weight_diff=1-alpha
    weight_val_auc=alpha
    
    # Calculate the weighted sum for each pair
    scores=[
        weight_diff*normalized_diff+weight_val_auc*normalized_val_auc
        for normalized_diff,normalized_val_auc in zip(normalized_diffs,normalized_val_aucs)
    ]
    
    return scores.index(max(scores))

def model_predict(tidy_set_path
                  ,model_weight_dir
                  ,l2_norm
                  ,batch_size
                  ,seed
                  ,new_tidy_set_path):
    
    # Load tidy set
    tidy_set=TidySet.read(tidy_set_path)
    print(f"Training file: {tidy_set_path}")
    tidy_set.desc()
    
    # Check if GPU is available and if not, fall back on CPU
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate ontonet
    uti_ontonet=ontonet(
        tidy_set
        ,device
        ,init_seed=seed
        ,init2_seed=seed
        ,l2_norm=l2_norm
        ,output_unit=1
        ,output_activation='sigmoid'
    )
    uti_ontonet.eval()  # Set the model to evaluation mode
    
    # Get list of all epoch.pt files in directory
    files=[f for f in os.listdir(model_weight_dir) if f.endswith('.pt') and f.split('.')[0].isdigit()]
    if not files:
        print("No .pt files found.")
        return
    
    # Get epoch numbers from file names and find the file with the highest epoch
    epochs=[int(f.split('.')[0]) for f in files]
    latest_epoch_file=files[epochs.index(max(epochs))]
    
    # Build complete file path
    model_path=os.path.join(model_weight_dir,latest_epoch_file)
    print(f'\nLoading model from: {model_path}\n')
    
    # load the model state dict
    uti_ontonet.load_state_dict(torch.load(model_path),strict=False)
    
    # Instantiate dataset
    new_tidy_set=TidySet.read(new_tidy_set_path)
    print(f"Prediction file: {new_tidy_set_path}")
    new_tidy_set.desc()
    
    val_indices=np.arange(notes(new_tidy_set.experimentData)['ontomap'].shape[0])
    val_dataset=OntoDataset(new_tidy_set,val_indices)
    
    # Define DataLoader
    val_dataloader=DataLoader(val_dataset,batch_size=batch_size,shuffle=True)
    
    # Determine outcome weights
    class_counts=pData(tidy_set)['outcome'].value_counts()  # Count the number of samples in each class
    total_samples=len(pData(tidy_set))  # Total number of samples
    
    weights={}
    
    for class_label,count in class_counts.items():
        p=count/total_samples  # Compute the proportion of samples in the class
        weight=1/(p*0.5)  # Compute the weight for the class
        weights[class_label]=weight

    weights=torch.Tensor([weights[class_label] for class_label in sorted(weights.keys())])
    
    # Define optimizer, criterion, and scheduler
    def criterions(input,target,weights):
        # Create a tensor for weights that is the same shape as target
        weight_map=torch.ones_like(target)
        for outcome,weight in enumerate(weights):
            weight_map[target==outcome]=weight
        # Compute weighted MSE loss
        return torch.sum(weight_map*(input-target)**2)/torch.sum(weight_map)
    
    progress_bar=tqdm(
        total=len(val_dataloader)
        ,desc='Predicting Progress'
        ,dynamic_ncols=True
        ,leave=False
    )
    
    val_true=[]
    val_pred=[]
    with torch.no_grad():  # No gradient calculation
        for inputs,targets in val_dataloader:
            tensor_inputs=inputs['ontoarray'].to(device)  # Extract tensor from dictionary and move to GPU
            targets={key:value.to(device) for key,value in targets.items()}  # Move targets to GPU
            outputs_dict=uti_ontonet(tensor_inputs)  # Forward pass

            val_true.append(targets['root'].detach().cpu().numpy())
            val_pred.append(outputs_dict['root'].detach().cpu().numpy())
            
            w_total=0.3*(len(outputs_dict.keys())-1)+1
            w_nonroot=0.3/w_total
            w_root=1/w_total
            loss_weights={output_name: w_root if output_name=='root' else w_nonroot for output_name in outputs_dict.keys()}
            
            total_loss=0
            val_rloss=0
            for output_name in outputs_dict.keys():
                loss=criterions(outputs_dict[output_name],targets[output_name],weights)  # Compute loss
                total_loss+=loss_weights[output_name]*loss  # Weight the loss
                if output_name=='root':
                    val_rloss+=loss

            val_loss=total_loss/w_total

            progress_bar.update(1)
            progress_bar.set_description(f"Validation Batch")
    
    val_true=np.concatenate(val_true).flatten()
    val_pred=np.concatenate(val_pred).flatten()
        
    # Xonvert val_true and val_pred to a DataFrame
    df=pd.DataFrame({'outcome':val_true,'prob':val_pred})
    
    # Write the DataFrame to a CSV file
    df.to_csv(f'{model_weight_dir}prob_'+os.path.basename(new_tidy_set_path).split('.')[0]+'.csv',index=False)

def model_grad_cam(tidy_set_path
                   ,model_weight_dir
                   ,l2_norm
                   ,batch_size
                   ,seed
                   ,new_tidy_set_path
                   ,ontology_name):
    
    # Load tidy set
    tidy_set=TidySet.read(tidy_set_path)
    print(f"Training file: {tidy_set_path}")
    tidy_set.desc()
    
    # Check if GPU is available and if not, fall back on CPU
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate ontonet
    uti_ontonet=ontonet(
        tidy_set
        ,device
        ,init_seed=seed
        ,init2_seed=seed
        ,l2_norm=l2_norm
        ,output_unit=1
        ,output_activation='sigmoid'
    )
    uti_ontonet.eval()  # Set the model to training mode
    
    # Get list of all epoch.pt files in directory
    files=[f for f in os.listdir(model_weight_dir) if f.endswith('.pt') and f.split('.')[0].isdigit()]
    if not files:
        print("No .pt files found.")
        return
    
    # Get epoch numbers from file names and find the file with the highest epoch
    epochs=[int(f.split('.')[0]) for f in files]
    latest_epoch_file=files[epochs.index(max(epochs))]
    
    # Build complete file path
    model_path=os.path.join(model_weight_dir,latest_epoch_file)
    print(f'\nLoading model from: {model_path}\n')
    
    # load the model state dict
    uti_ontonet.load_state_dict(torch.load(model_path),strict=False)
    
    # Instantiate dataset
    new_tidy_set=TidySet.read(new_tidy_set_path)
    print(f"Prediction file: {new_tidy_set_path}")
    new_tidy_set.desc()
    
    train_indices=np.arange(notes(new_tidy_set.experimentData)['ontomap'].shape[0])
    full_dataset=ontoarray(new_tidy_set,train_indices)
    
    def make_multiple_of_batch_size(index_array, batch_size):
        remainder=len(index_array)%batch_size
        if remainder!=0:
            extra_samples_needed=batch_size-remainder
            extra_samples=index_array[:extra_samples_needed]
            index_array=np.concatenate([index_array,extra_samples])
        return index_array
    
    train_indices=np.arange(len(full_dataset))
    train_dataset=Subset(full_dataset,make_multiple_of_batch_size(train_indices,batch_size))
    
    # Create a generator object with the random seed
    if seed is not None:
        generator=torch.Generator()
        generator.manual_seed(seed)
    
    # Define DataLoader
    if seed is not None:
        train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,generator=generator)
    else:
        train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    
    heatmap_accumulator={}
    
    progress_bar=tqdm(
        total=len(train_dataloader)
        ,desc='Grad-CAM Progress'
        ,dynamic_ncols=True
        ,leave=False
    )
    
    with torch.no_grad():  # No gradient calculation
        for inputs,targets in train_dataloader:
            tensor_inputs=inputs['ontoarray'].float().to(device)  # Extract tensor from dictionary and move to GPU
            targets={key:value.to(device) for key,value in targets.items()}  # Move targets to GPU
            outputs_dict=uti_ontonet(tensor_inputs)  # Forward pass
            
            # Compute Grad-CAM for a specific class and input
            grad_cam_maps=uti_ontonet.compute_grad_cam(tensor_inputs)
            
            for key in grad_cam_maps:
                if key not in heatmap_accumulator:
                    heatmap_accumulator[key]=grad_cam_maps[key].clone()
                else:
                    heatmap_accumulator[key]+=grad_cam_maps[key]
            
            progress_bar.update(1)
    
    num_batches=len(train_dataloader)
    for key in heatmap_accumulator:
        heatmap_accumulator[key]/=num_batches
    
    data=[]
    for key,heatmap in heatmap_accumulator.items():
        heatmap_np=heatmap.cpu().numpy()
        for z in range(heatmap_np.shape[1]):
            for x in range(heatmap_np.shape[2]):
                for y in range(heatmap_np.shape[3]):
                    value=heatmap_np[0,0,x,y,z]
                    data.append([key,x,y,z,value])
    
    # print('gradcam: ',data)
    
    df=pd.DataFrame(data,columns=['ontology','x','y','z','value'])
    df.to_csv(f'{model_weight_dir}gradcam_'+os.path.basename(new_tidy_set_path).split('.')[0]+'.csv',index=False)
    
    return heatmap_accumulator

