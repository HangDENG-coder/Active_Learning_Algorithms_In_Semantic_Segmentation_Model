import argparse
import os
import numpy as np
import math

from torchvision import datasets

from transformer_net import *
import torchvision
import torchvision.transforms.functional as TF
import albumentations as albu
import copy
from skimage import io, util, exposure, segmentation,measure, draw, morphology, restoration

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
import matplotlib as mpl
import datetime

import pickle as pkl

from PIL import Image,ExifTags,ImageFilter,ImageOps, ImageDraw, ImageFont

from model_architectures_for_semantic_segmentation import *
from semantic_postprocessing import *
from dilution_check import *

import shutil
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import Counter
import random

from matplotlib.colors import ListedColormap
from utils_semantic_segmentation import *


import parameters



def get_paths_to_training_and_validation_samples(path_to_data_sets):
    paths_train=[]
    paths_val=[]
    for folder in os.listdir(path_to_data_sets):
        if os.path.isdir(path_to_data_sets+folder) and folder!=".DS" and folder!= ".ipynb_checkpoints":
            if folder.split('.')[-1]!='pkl':
                path_to_folder = path_to_data_sets+folder+'/'
                for file in os.listdir(path_to_folder+'train/'):
                    paths_train.append(path_to_folder+'train/'+file)
                for file in os.listdir(path_to_folder+'test/'):
                    paths_val.append(path_to_folder+'test/'+file)
    return {'paths_train': paths_train, 'paths_val': paths_val,'paths_all': paths_train+paths_val}

paths_to_samples_dict = get_paths_to_training_and_validation_samples(parameters.path_to_data_sets)


#Next we load the dictionary that maps class indexes to the class names
with open(parameters.path_to_data_sets+'semantic_class_idx_to_semantic_class_name.pkl', 'rb') as handle:
    semantic_class_idx_to_semantic_class_name = pkl.load(handle)
    
class_name_to_class_idx_semantic = {class_name: idx for idx,class_name in semantic_class_idx_to_semantic_class_name.items()}
class_name_to_class_idx_object = {'background': 0,'object': 1, 'border': 2}
class_idx_to_class_name_object = {0: 'background', 1: 'object', 2: 'border'}
class_idx_to_incidence_object,class_idx_to_incidence_semantic = get_class_idx_to_incidence(paths_to_samples_dict['paths_val'])








cuda = True if torch.cuda.is_available() else False
loss_weights_tensor_object = get_loss_weights(class_idx_to_incidence_object,
                                              class_idx_to_prior=None,cuda=cuda)
loss_weights_tensor_semantic = get_loss_weights(class_idx_to_incidence_semantic,
                                                class_idx_to_prior=None,cuda=cuda)




data_generator_tot = DataGenerator(paths_to_samples_dict['paths_all'],
                                   class_name_to_class_idx_semantic,
                                   class_name_to_class_idx_object,
                            loss_weights_tensor_object,
                                   loss_weights_tensor_semantic,
                                   cuda=cuda,batch_size=20,
                 augmentation_probability=1,
                                   steps_per_epoch=int(len(paths_to_samples_dict['paths_all'])/20))





#The model stores a list with the class names, where the position of the list encodes for the class index
sorted_semantic_class_name_to_idx = {k: v for k, v in sorted(data_generator_tot.semantic_class_name_to_idx.items(), key=lambda item: item[1])}
sorted_object_class_name_to_idx = {k: v for k, v in sorted(data_generator_tot.object_class_name_to_idx.items(), key=lambda item: item[1])}
class_names_semantic = list(sorted_semantic_class_name_to_idx.keys())
class_names_binary = list(sorted_object_class_name_to_idx.keys())





#The model family is parameterized by a complexity parameter that controlls the number of parameters by atlernatingly 
#adding depth and width to the network.
complexity=6#the larger the more paramter the model has
n_models=5#the number of independently trained models in the ensemble
weight_decay=1e-3#the weight of the L2 regularization of the parameters

#model_type can be either 'pseudo_geometric' or 'pseudo_geometric'
#these are both ensemble architectures.
#pseudo_geometric: Is much faster as its individual model are not rotaion and reflexion invariant. But during training and testing
#                   they see an individually rotated or reflected data set
#'geometric': The individual models are all 90 degrees rotation and reflexion invariant (group invariance).
            #It is hence much slower (8 times more slow than pseudo_geometric, the cardinality of the group) but more accurate and 
            #needs also less regularization
model_type = parameters.model_type




print('Initializing model of type: ',model_type)
if model_type=='pseudo_geometric':
    model=PseudoGeometricSegNet(n_models=n_models,in_ch=1,out_ch_mask=3,
                out_ch_semantic=data_generator_tot.n_classes,complexity=complexity,
                n_low_res_layers=2,separate_decoder=False,class_names_semantic=class_names_semantic,
                class_names_binary=class_names_binary)
    
elif model_type=='geometric':
    model=EnsembleSegNet(n_models=n_models,in_ch=1,out_ch_mask=3,
                out_ch_semantic=data_generator_tot.n_classes,complexity=complexity,
                n_low_res_layers=2,separate_decoder=False,geometric_model=True,
                class_names_semantic=class_names_semantic,class_names_binary=class_names_binary)

else:
    print('This model type is not implemented: ',model_type)

n_params = count_parameters(model,trainable_only=True)
print('num total params: ',n_params)
if cuda:
    model.cuda()
    
    
folder = parameters.folder
now = datetime.datetime.now()
if folder[-1]!='/':
    folder+='/'
save_dir =folder +datetime_to_string(now)+'.pkl'

if not os.path.exists(folder):
    os.mkdir(folder)
model_save_dir = save_model(model,save_dir)    
    
if cuda:
    device='cuda'
else:
    device='cpu'
loaded_model = load_model(model_save_dir, device = device).eval()    


model_id = parameters.model_id


path_to_semantic_segmentation_model = model_id + '.pkl'
cuda = True if torch.cuda.is_available() else False
if cuda:
    device='cuda'
else:
    device='cpu'
print(device)
def load_model_semantic(load_dir,device='cpu'):
    state_dict=torch.load(load_dir,map_location=torch.device(device))
    
    class_names_semantic=state_dict['class_names_semantic']
    class_names_binary=state_dict['class_names_binary']
    complexity=state_dict['complexity']
    model_type=state_dict['model_type']
    n_low_res_layers=state_dict['n_low_res_layers']
    in_ch=state_dict['in_ch']
    out_ch_mask=state_dict['out_ch_mask']
    out_ch_semantic=state_dict['out_ch_semantic']
    separate_decoder=state_dict['separate_decoder']

    if model_type=='EnsembleSegNet':
        n_models=state_dict['n_models']
        geometric_model=state_dict['geometric_model']
        model=EnsembleSegNet(n_models=n_models,in_ch=in_ch,out_ch_mask=out_ch_mask,
                        out_ch_semantic=out_ch_semantic,complexity=complexity,
                        n_low_res_layers=n_low_res_layers,separate_decoder=separate_decoder,geometric_model=geometric_model,
                        class_names_semantic=class_names_semantic,class_names_binary=class_names_binary)
    elif model_type =='PseudoGeometricSegNet':
        n_models=state_dict['n_models']
        model=PseudoGeometricSegNet(n_models=n_models,in_ch=in_ch,out_ch_mask=out_ch_mask,
                        out_ch_semantic=out_ch_semantic,complexity=complexity,
                        n_low_res_layers=n_low_res_layers,separate_decoder=separate_decoder,
                        class_names_semantic=class_names_semantic,class_names_binary=class_names_binary)
    else:
        raise ValueError('Unknown model_architecture: ',model_architecture)

    
    model.load_state_dict(state_dict['model_state']) 
    
    if device == 'cpu':
        model.cpu()
    elif device == 'cuda':
        model.cuda()
    else:
        raise ValueError('device must be either cpu or cuda.')

    return model
loaded_model = load_model_semantic(path_to_semantic_segmentation_model,device=device).eval()
    
    


class_names_semantic = loaded_model.class_names_semantic
class_name_object = loaded_model.class_names_binary

model_class_idx_to_class_name_semantic = {idx: class_name for idx,class_name in enumerate(class_names_semantic)}
model_class_idx_to_class_name_object = {idx: class_name for idx,class_name in enumerate(class_name_object)}

model_class_name_to_class_idx_semantic = {class_name: idx for idx,class_name in model_class_idx_to_class_name_semantic.items()}
model_class_name_to_class_idx_object = {class_name: idx for idx,class_name in model_class_idx_to_class_name_object.items()}




def segmentation_out_idx(phase_img,out,idx,model_number):
    ### model_number is a string of a single model number of {0,1,2}
    ### "ensembled" is the average output of the models
    if model_number.isnumeric(): 
        pred = out[0][int(model_number)]
    elif model_number == "ensembled":
        pred = out[1]
    
    img_torch, bg_int = preprocess_image(np.array(phase_img[idx,0].cpu()),cuda=cuda)
    segmentation_output = {
                            'object_map_logits':pred['binary_mask'][idx].cpu(),
                          'semantic_map_logits':pred['semantic_mask'][idx].cpu(),
                            'phase_img': np.array(phase_img[idx,0].cpu()),
                            'background_intensity': bg_int,
                          }
    return segmentation_output


def segmentation_label_idx(phase_img,object_mask,semantic_mask,idx):
    ### model_number is a string of model number of {0,1,2}
    ### "ensembled" is the average output of the models
    
    img_torch, bg_int = preprocess_image(np.array(phase_img[idx,0]),cuda=cuda)
    segmentation_output = {
                            'object_map_logits':object_mask[idx].cpu(),
                          'semantic_map_logits':semantic_mask[idx].cpu(),
                            'phase_img': np.array(phase_img[idx,0]),
                            'background_intensity': bg_int,
                          }
    return segmentation_output

def get_processed_segmentation_output(segmentation_output,
                                      model_class_idx_to_class_name,
                       class_idx_to_min_max_feature_value, 
                                      region_mode,
                       cuda=False,
                       minimum_probability=0,
                       min_object_size = 2, remove_small_holes = 3,connectivity=1):
    '''
    region_mode: 'object_mask_pooling' is fast but does not take the semantic information in to account
                'semantic_thresholding' slow but does take the semantic information in to account
    '''

   

    print("line 14")
    processed_frame = SegmentationPostProcessing(segmentation_output['object_map_logits'],
                               segmentation_output['semantic_map_logits'], 
#                                segmentation_output['phase_img'], 
                               model_class_idx_to_class_name,
                                minimum_probability=0,min_object_size = 2, remove_small_holes = 3,connectivity=1)
    
    print("line 21")
    processed_frame.define_semantic_regions(mode=region_mode)

    processed_frame.post_process_semantic_regions(class_idx_to_min_max_feature_value=class_idx_to_min_max_feature_value)

    processed_frame.exclude_non_informative_objects(non_informative_classes = ['background'])

    processed_frame.original_phase_img = segmentation_output['phase_img']

    return processed_frame







