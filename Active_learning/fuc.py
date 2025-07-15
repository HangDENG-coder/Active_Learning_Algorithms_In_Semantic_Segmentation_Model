import pickle
import numpy as np
import torch
import torch.nn as nn
import datetime
from pynvml import *
from utils_semantic_segmentation import *


path_to_data_sets = '/scratch/hangdeng/ActiveLearning/data_06-09-22/'
paths_to_samples_dict=get_paths_to_training_and_validation_samples(path_to_data_sets)

def cuda_memory():
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')

def selected_path_list(selected_index):
    a = np.array(paths_to_samples_dict['paths_train'])
    return list(a[selected_index])

def prediction_unlabeled(model,phase_img_unlabeled):
    model.eval()#make sure that the model is in eval mode (applying learned batch-norm parameters)
    with torch.no_grad():
        out=model(phase_img_unlabeled)
    return out


##########################################################################################################################
############# get single data uncertainty entropy on on model ############################################################
def get_entropy_tensor(probability_tensor,axis):
    #computes the entropy of a probability_tensor along axis, while it is assumed that the vectors along axis are probabilites.
    # log_p = torch.log(probability_tensor.cpu())
    # return - torch.sum(probability_tensor.cpu()*log_p.cpu(),dim=axis)
    log_p = torch.log(probability_tensor)
    return - torch.sum(probability_tensor*log_p,dim=axis)

def softmax_probibility(torch_raw_output_all_class,axis):
    return nn.functional.softmax(torch_raw_output_all_class, dim=axis)

def metric_entropy(segmentation_output_logits,y_one):
#### segmentation_output_logits is the   segmentation_output['object_map_logits']/['semantic_map_logits'] on multiple input images  
#### return the summed entropy among each input images
    probability_tensor = softmax_probibility(segmentation_output_logits,1)
    # data_uncertainty_tensor = get_entropy_tensor(probability_tensor,axis=1).numpy()
    # data_uncertainty_tensor = data_uncertainty_tensor * np.array(y_one.cpu())
    # sumed_entropy_output = np.apply_over_axes(np.sum, data_uncertainty_tensor, [1,2])
    # return sumed_entropy_output.reshape((sumed_entropy_output.shape[0]))
    data_uncertainty_tensor = get_entropy_tensor(probability_tensor,axis=1)
    data_uncertainty_tensor = data_uncertainty_tensor * y_one
    sumed_entropy_output = torch.sum(data_uncertainty_tensor, dim = [1,2])
    return sumed_entropy_output
    
    



##########################################################################################################################
######################################## get model uncertainty ############################################################
def entropy_raw_output(output):
    probibility= softmax_probibility(output,1)
    entropy_uncertainty = get_entropy_tensor(probibility,1)
    return entropy_uncertainty

def BALD(out,mask_key):  
    if mask_key in list(out[1].keys()):
        entropy_ensemble = entropy_raw_output(out[1][mask_key])
        entropy_member = 0
        for i in range(len(out[0])):
            entropy_member += entropy_raw_output(out[0][i][mask_key])
        return entropy_ensemble - entropy_member/len(out[0])
    else:
        print('error: no existed ',mask_key)
        
        
def metric_bald(out,mask_key,y_one):
    model_uncertainty_tensor = BALD(out,mask_key)
    # model_uncertainty_tensor = (model_uncertainty_tensor*(y_one)).cpu()
    # sumed_bald_output = np.apply_over_axes(np.sum, model_uncertainty_tensor, [1,2])
    # return sumed_bald_output.reshape((sumed_bald_output.shape[0]))
    model_uncertainty_tensor = model_uncertainty_tensor*y_one
    sumed_bald_output = torch.sum( model_uncertainty_tensor, dim = [1,2])
    return sumed_bald_output




##########################################################################################################################
######################################## get variation ratios ##########################################################
def model_member_decision(out,mask_key):
    '''
    returns the predicted class of each ensembel model member
    '''
    size = (out[0][0][mask_key].shape)
    tot_decision = torch.argmax(out[0][0][mask_key],axis = 1,keepdim=True)
    for model_member in range(1,len(out[0])):
        member_decision = torch.argmax(out[0][model_member][mask_key],axis = 1,keepdim=True)
        tot_decision = torch.cat((tot_decision,member_decision),axis = 1)
    return tot_decision


def variation_ratios_calculation(tot_decision):
    '''
    most_decision: the predicted class of majority decision
    diffence_decision: find the prediction that disagree
    variation_ratios: should be (0,1) for each pixel
    return the ratio of disagreement among all the ensemble model member
    '''
    most_decision = torch.mode(tot_decision,1)[0]
    diffence_decision = (tot_decision - most_decision[:,None])
    diffence_decision[diffence_decision!=0] = 1
    num_diff = torch.sum(diffence_decision,axis = 1)
    variation_ratios = num_diff/tot_decision.shape[1]
    return variation_ratios


def variation_ratios_uncertainty(out,mask_key):
    '''
    calculate multiple images variation ratios
    '''
    tot_decision = model_member_decision(out,mask_key)
    variation_ratios = variation_ratios_calculation(tot_decision)
    return variation_ratios


def metric_variation_ratios(out,mask_key,y_one):
    variation_uncertainty = variation_ratios_uncertainty(out,mask_key)
    variation_uncertainty = variation_uncertainty * y_one
    sumed_variation = torch.sum( variation_uncertainty, dim = [1,2])
    return sumed_variation

##########################################################################################################################
######################################## get standard_deviation from the final ensemble model ############################
def standard_deviation_ensemble(out_ensemble,mask_key):
    '''
    it is a kind of how well the final prediction is based on the assumption: 
        (1) model coverged
        (2) shape well predicted
    target on the ensemble model's final prediction from the out[1] to obtain the standard_deviation 
    '''
    std = torch.std(out_ensemble[mask_key],dim = 1)
    return std 


def metric_sumed_std(out_ensemble,mask_key,y_one):
    std_ensmeble = standard_deviation_ensemble(out_ensemble,mask_key)
    std_ensmeble = std_ensmeble * y_one
    sumed_std_ensmeble = torch.sum( std_ensmeble, dim = [1,2])
    return sumed_std_ensmeble

##########################################################################################################################
######################################## get uncertainty metric ##########################################################

def one_like_labeled(mask_labeled):
    ind = mask_labeled>0
    y_one = torch.zeros(mask_labeled.shape)
    y_one[ind] = 1
    return y_one.cuda()

def pred_data_generator(model,data_generator,mask_key):
    sample_selected = next(data_generator)
    out = prediction_unlabeled(model,sample_selected['phase_img'])
    y_one = one_like_labeled(sample_selected[mask_key])
    
    key_list = list(out[1].keys())
    key_list.remove(mask_key)
    removed_key = key_list[0]
    out[1].pop(removed_key)
    for i in range(len(out[0])):
        out[0][i].pop(removed_key)
    return out,y_one


def metric_dic(out,mask_key,y_one):
    metric_dic = {}
    # metric_dic.update({'data_uncertainty_entropy': metric_entropy(out[1][mask_key],y_one)})
    metric_dic.update({'model_uncertainty_bald': metric_bald(out,mask_key,y_one)})
    # metric_dic.update({'model_varaiation_ratios': metric_variation_ratios(out,mask_key,y_one)})
    # metric_dic.update({'data_standard_deviation': metric_sumed_std(out[1],mask_key,y_one)})
    return metric_dic


def selected_training_list(uncertainty,selected_number,untrained_sample_list):
    ind = np.argpartition(np.array(uncertainty.cpu()), -selected_number)[-selected_number:]
    selected_training_sample = list(np.array(untrained_sample_list)[ind])
    return selected_training_sample


def uncertainty_tot(model,data_generator_untrained,mask_key,uncertainty_key,untrained_sample_list):
    # uncertainty = np.array([])
    uncertainty = torch.zeros(0).cuda()
    for _ in range(int(len(untrained_sample_list)/data_generator_untrained.batch_size)+1):
        uncertainty = uncertainty_batch(model,data_generator_untrained,mask_key,uncertainty_key,uncertainty)
    uncertainty = uncertainty[0:len(untrained_sample_list)]
    uncertainty = torch.nan_to_num(uncertainty)
    return uncertainty


def uncertainty_batch(model,data_generator_untrained,mask_key,uncertainty_key,uncertainty):
    out,y_one = pred_data_generator(model,data_generator_untrained,mask_key)
    uncertainty = torch.cat((metric_dic(out,mask_key,y_one)[uncertainty_key],uncertainty),0)
    return uncertainty


def selected_training_sample(model,data_generator_untrained,selected_number,untrained_sample_list,mask_key,uncertainty_key):
    ############  selelct 'selected_number' of samples to train 
    ############  mask_key:'semantic_mask' or 'object_mask' ############
    ############  uncertainty_key: 'data_uncertainty_entropy' or 'model_uncertainty_bald'############
    # out = pred_data_generator(model,data_generator_untrained)
    # uncertainty = metric_dic(out,mask_key)[uncertainty_key]
    uncertainty = uncertainty_tot(model,data_generator_untrained,mask_key,uncertainty_key,untrained_sample_list)
    selected_training_sample = selected_training_list(uncertainty,selected_number,untrained_sample_list)
    return selected_training_sample

def save_dic_pkl(pkl_filename,performance_over_epochs):
    # create a binary pickle file 
    f = open(pkl_filename,"wb")
    # write the python object (dict) to pickle file
    pickle.dump(performance_over_epochs,f)
    # close file
    f.close()

def save_trained_model(folder,model):
    if folder[-1]!='/':
        folder+='/'
    now = datetime.datetime.now()
    save_dir =folder +datetime_to_string(now)+'.pkl'
    if not os.path.exists(folder):
        os.mkdir(folder)
    model_save_dir = save_model(model,save_dir)