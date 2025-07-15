import pickle
import numpy as np
import torch
import torch.nn as nn
import datetime
from scipy.stats import mode
from pynvml import *
from utils_semantic_segmentation_active import *
import os
import pandas as pd
cuda = True if torch.cuda.is_available() else False
path_to_data_sets = 'data_06-09-22/'
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
    return list(a[np.array(selected_index)])

def prediction_unlabeled(model,phase_img_unlabeled):
    model.eval()#make sure that the model is in eval mode (applying learned batch-norm parameters)
    with torch.no_grad():
        out=model(phase_img_unlabeled)
    return out


def new_batch_root(root):
    dirlist = [ int(item) for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
    if dirlist:
        new_batch = max(dirlist) - 1
    else:
        new_batch = 0
    return new_batch


def pkl_file_path(pathname):

    if os.path.exists(pathname):
        read_file = pd.read_pickle(pathname)
    else:
        read_file = {}
    return read_file

##########################################################################################################################
############# get single data uncertainty entropy on on model ############################################################

def get_entropy_tensor(probability_tensor,axis):
    #computes the entropy of a probability_tensor along axis, while it is assumed that the vectors along axis are probabilites.
   
    log_p = torch.log2(probability_tensor)
    entropy = - torch.sum(probability_tensor*log_p,dim=axis)
    '''
    Here notice: the probability_tensor.shape(1) is the total number of classes
    '''
    entropy_normalized = entropy/math.log2(probability_tensor.shape[1])
    return entropy_normalized

def softmax_probibility(torch_raw_output_all_class,axis):
    return nn.functional.softmax(torch_raw_output_all_class, dim=axis)


def entropy_images(out,mask_key,y_one):
    probability_tensor = softmax_probibility(out[1][mask_key],1)
    data_uncertainty_tensor = get_entropy_tensor(probability_tensor,axis=1)
    data_uncertainty_tensor = data_uncertainty_tensor * y_one  * get_binary_filter(out)
    return (data_uncertainty_tensor)


def metric_entropy(out,mask_key,y_one):
#### segmentation_output_logits is the   segmentation_output['object_map_logits']/['semantic_map_logits'] on multiple input images 
#### return the summed entropy among each input images
    data_uncertainty_tensor = entropy_images(out,mask_key,y_one)
    sumed_entropy_output = torch.sum(data_uncertainty_tensor, dim = [1,2])
    # return sumed_entropy_output/torch.sum(data_uncertainty_tensor>0, dim = [1,2])
    return sumed_entropy_output


    
    
##########################################################################################################################
######################################## get model uncertainty ############################################################
def entropy_raw_output(output):
    probibility= softmax_probibility(output,1)
    entropy_uncertainty = get_entropy_tensor(probibility,1)
    return entropy_uncertainty




def BALD(out,mask_key):  
    if mask_key in list(out[1].keys()):
        
        entropy_member = 0
        entropy_member_softmax = 0
        for i in range(len(out[0])):
            entropy_member += entropy_raw_output(out[0][i][mask_key])
            entropy_member_softmax += softmax_probibility(out[0][i][mask_key],1)
        
        entropy_member_softmax = entropy_member_softmax/len(out[0])
        entropy_ensemble = entropy_raw_output(entropy_member_softmax)    
            
        return entropy_ensemble - entropy_member/len(out[0])
        
        
    else:
        print('error: no existed ',mask_key)
        
def bald_images(out,mask_key,y_one):
    '''
    !!! In the early beggining of the training, the mutual information could be negative?
    '''
    model_uncertainty_tensor = BALD(out,mask_key)
    model_uncertainty_tensor = model_uncertainty_tensor * y_one * get_binary_filter(out)
    return model_uncertainty_tensor

        
    
def metric_bald(out,mask_key,y_one):
    model_uncertainty_tensor = bald_images(out,mask_key,y_one)
    sumed_bald_output = torch.sum( model_uncertainty_tensor, dim = [1,2])
    
    return sumed_bald_output


    
##########################################################################################################################
############################### get mean standard deviation (Jensen'sinequality) #########################################

def mean_standard_deviation(out,mask_key):
    p_sqaure = 0
    p_ave = 0
    for i in range(len(out[0])):
        probibility = softmax_probibility(out[0][i][mask_key],1)
        p_sqaure += probibility*probibility/len(out[0])
        p_ave += probibility/len(out[0])

    p_sqaure_ave = p_sqaure
    p_ave_square = p_ave**2
    sigma = torch.sqrt(p_sqaure_ave -  p_ave_square)
    return torch.sum(sigma,1)




def mean_standard_deviation_images(out,mask_key,y_one):
    sigma_tensor = mean_standard_deviation(out,mask_key) 
    sigma_tensor = sigma_tensor * y_one * get_binary_filter(out)
    return sigma_tensor

def metric_mean_standard_deviation(out,mask_key,y_one):
    sigma_tensor = mean_standard_deviation_images(out,mask_key,y_one)
    sumed_sigma_tensor = torch.sum(sigma_tensor,dim=[1,2])
      
    return sumed_sigma_tensor
    



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


def variation_ratios_images(out,mask_key,y_one):
    variation_uncertainty = variation_ratios_uncertainty(out,mask_key)
    variation_uncertainty = variation_uncertainty * y_one * get_binary_filter(out)
    return variation_uncertainty


def metric_variation_ratios(out,mask_key,y_one):
    variation_uncertainty = variation_ratios_images(out,mask_key,y_one)
    sumed_variation = torch.sum( variation_uncertainty, dim = [1,2])
    return sumed_variation

##########################################################################################################################
######################################## get standard_deviation from the final ensemble model ############################
def standard_deviation_ensemble(out,mask_key):
    '''
    it is a kind of how well the final prediction is based on the assumption: 
        (1) model coverged
        (2) shape well predicted
    target on the ensemble model's final prediction from the out[1] to obtain the standard_deviation 
    '''
    std = torch.std(out[1][mask_key],dim = 1)
    return std 


def std_images(out,mask_key,y_one):
    std_ensmeble = standard_deviation_ensemble(out,mask_key)
    std_ensmeble = std_ensmeble * y_one * get_binary_filter(out)
    return std_ensmeble 


def metric_sumed_std(out,mask_key,y_one):
    std_ensmeble = std_images(out,mask_key,y_one)
    sumed_std_ensmeble = torch.sum( std_ensmeble, dim = [1,2])
    
    return sumed_std_ensmeble
    
##########################################################################################################################
######################################## get least_confidencefrom the final ensemble model ############################
def least_confidence(out,mask_key):
    '''
    it is a kind of how well the final prediction is based on the assumption: 
        (1) model coverged
        (2) shape well predicted
    target on the ensemble model's final prediction from the out[1] to obtain the standard_deviation 
    '''
    probability_tensor = softmax_probibility(out[1][mask_key],1)
    least_confidence = 1 - torch.max(probability_tensor,dim = 1).values
    return least_confidence

def leastconfidence_images(out,mask_key,y_one):
    least_con = least_confidence(out,mask_key)
    least_con = least_con * y_one * get_binary_filter(out)
    return least_con 


def metric_sumed_leastconfidence(out,mask_key,y_one):
    least_con = leastconfidence_images(out,mask_key,y_one)
    sumed_least_con = torch.sum( least_con, dim = [1,2])
    
    return sumed_least_con


##########################################################################################################################
######################################## get uncertainty metric ##########################################################

def one_like_tensor(mask_labeled):
    
    y_one = torch.ones(mask_labeled.shape)
    '''
    !!!
    For old camera,remove those unknow area without labeling
    In new camera, remove the edge! remove the following lines!!!
    !!!
    '''
    ind = mask_labeled<0
    y_one[ind] = 0
    return y_one.cuda()

def get_binary_filter(out):

    pred_bianry = torch.argmax(out[1]['binary_mask'],1)
    binary_filter = torch.zeros(pred_bianry.shape)
    binary_filter[pred_bianry==1] = 1
    return binary_filter.cuda()

def pred_data_generator(model,data_generator,mask_key):
    sample_selected = next(data_generator)
    sample_selected['binary_mask'] = sample_selected['object_mask'] 

    
    out = prediction_unlabeled(model,sample_selected['phase_img'])  
    '''
    if using the old camera data, here remove those unlabeled area.
    Else using the new camera data, here only remove the edge area.
    '''
    y_one = one_like_tensor(sample_selected[mask_key])

    '''
    '''

    
    return out,y_one


def out_ensemble(out,mask_key):
    entropy_member_softmax = 0
    for i in range(len(out[0])):           
        entropy_member_softmax += softmax_probibility(out[0][i][mask_key],1)
    
    return entropy_member_softmax/len(out[0])


def metric_dic(out,mask_key,y_one,uncertainty_key):
    if uncertainty_key == 'data_uncertainty_entropy':
        
        return metric_entropy(out,mask_key,y_one)
        
    elif uncertainty_key == 'model_uncertainty_bald':
        return metric_bald(out,mask_key,y_one)
    
    elif uncertainty_key == 'model_varaiation_ratios':
        return metric_variation_ratios(out,mask_key,y_one)
    
    elif uncertainty_key == 'data_standard_deviation':
        return metric_sumed_std(out,mask_key,y_one)
    
    elif uncertainty_key == 'least_confidence':
        return metric_sumed_leastconfidence(out,mask_key,y_one)
    
    elif uncertainty_key == 'mean_standard_deviation':
        return metric_mean_standard_deviation(out,mask_key,y_one)
        
        
def uncertainty_image(out,mask_key,y_one,uncertainty_key):
    if uncertainty_key == 'data_uncertainty_entropy':
        # return entropy_images(out_ensemble(out,mask_key),y_one)
        return entropy_images(out,mask_key,y_one)
        
    elif uncertainty_key == 'model_uncertainty_bald':
        return bald_images(out,mask_key,y_one)
    
    elif uncertainty_key == 'model_varaiation_ratios':
        return variation_ratios_images(out,mask_key,y_one)
    
    elif uncertainty_key == 'data_standard_deviation':
        return std_images(out,mask_key,y_one)
    
    elif uncertainty_key == 'least_confidence':
        return leastconfidence_images(out,mask_key,y_one)
    
    elif uncertainty_key == 'mean_standard_deviation':
        return mean_standard_deviation_images(out,mask_key,y_one)


def uncertainty_batch(model,data_generator_untrained,mask_key,uncertainty_key,uncertainty):
    out,y_one = pred_data_generator(model,data_generator_untrained,mask_key)
    # uncertainty = torch.cat((metric_dic(out,mask_key,y_one,uncertainty_key)[uncertainty_key],uncertainty),0)
    uncertainty = torch.cat((uncertainty, metric_dic(out, mask_key, y_one, uncertainty_key)), 0)
    return uncertainty


def selected_training_list(uncertainty,selected_number):
    ind = np.argpartition(np.array(uncertainty.cpu()), -selected_number)[-selected_number:]
    return ind


def test_selected_training_sample(model,data_generator_untrained,selected_number,mask_key,uncertainty_key):
    ############  selelct 'selected_number' of samples to train 
    ############  mask_key:'semantic_mask' or 'object_mask' ############
    ############  uncertainty_key: 'data_uncertainty_entropy' or 'model_uncertainty_bald'############
    uncertainty = uncertainty_tot(model,data_generator_untrained,mask_key,uncertainty_key)
    selected_training_sample_ind = selected_training_list(uncertainty,selected_number)
    return selected_training_sample_ind


def uncertainty_tot(model,data_generator_untrained,mask_key,uncertainty_key):
    uncertainty = torch.zeros(0).cuda()
    for _ in range(int(len(data_generator_untrained.data_path_list)/data_generator_untrained.batch_size)):
        uncertainty = uncertainty_batch(model,data_generator_untrained,mask_key,uncertainty_key,uncertainty)
    uncertainty = uncertainty[0:len(data_generator_untrained.data_path_list)]
    uncertainty = torch.nan_to_num(uncertainty)
    return uncertainty


def save_dic_pkl(pkl_filename,performance_over_epochs):
    # create a binary pickle file 
    dirname = os.path.dirname(pkl_filename) 
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    f = open(pkl_filename,"wb")    
    pickle.dump(performance_over_epochs,f)
    f.close()


def save_trained_model(folder,model,optimizer):
    if folder[-1]!='/':
        folder+='/'
    now = datetime.datetime.now()
    save_dir =folder +datetime_to_string(now)+'.pkl'
    if not os.path.exists(folder):
        os.mkdir(folder)
    model_save_dir = save_model(model,save_dir,optimizer)
    
    
##########################################################################################################################
######################################## get best uncertainty key from the validation_data ###############################
def uncertainty_images(uncertainty_key,out,mask_key,y_one):
    if uncertainty_key == 'data_uncertainty_entropy':
        # y_uncertainty =  entropy_images(out_ensemble(out,mask),y_one)
        y_uncertainty =  entropy_images(out[1][mask_key],y_one)
        
    elif uncertainty_key == 'model_uncertainty_bald':
        y_uncertainty =  bald_images(out,mask_key,y_one)
        
    elif uncertainty_key == 'model_varaiation_ratios':
        y_uncertainty = variation_ratios_images(out,mask_key,y_one)
        
    elif uncertainty_key == 'data_standard_deviation':
        y_uncertainty = std_images(out[1],mask_key,y_one)
        
    elif uncertainty_key == 'least_confidence':
        return leastconfidence_images(out[1],mask_key,y_one)
    
    elif uncertainty_key == 'mean_standard_deviation':
        return mean_standard_deviation_images(out,mask_key,y_one)
        
    else:
        print("no such uncertainty_key", uncertainty_key)
    return y_uncertainty





def score_intersection(one_intersection,y_uncertainty,thr):
    threshold_uncertainty = copy.deepcopy(y_uncertainty)
    threshold_uncertainty[threshold_uncertainty<thr] = 0
    score_intersection = torch.sum(one_intersection * threshold_uncertainty,dim=(1,2))/torch.sum(threshold_uncertainty,dim=(1,2))
    return score_intersection


def intersection_uncertainty(y_pred,y_one,uncertainty_key,out,mask_key):
    one_pred = one_like_tensor(y_pred)
    one_error = one_like_tensor(one_pred - y_one)
    y_uncertainty = uncertainty_images(uncertainty_key,out,mask_key,y_one)
    one_uncertainy = one_like_tensor(y_uncertainty)
    one_intersection = one_uncertainy * one_error
    return one_intersection,y_uncertainty


def thr_score(y_pred,y_one,uncertainty_key,out,mask_key):
    one_intersection,y_uncertainty = intersection_uncertainty(y_pred,y_one,uncertainty_key,out,mask_key)
    num_thr = 10
    thr_score = np.zeros((num_thr,2))
    for i, thr in enumerate(np.linspace(torch.min(y_uncertainty[y_uncertainty!=0]).cpu(), torch.max(y_uncertainty[y_uncertainty!=0]).cpu(), num=num_thr)):
        # score_intersection = score_intersection(one_intersection,y_uncertainty,thr)
        threshold_uncertainty = copy.deepcopy(y_uncertainty)
        threshold_uncertainty[threshold_uncertainty<thr] = 0
        score_intersection = torch.sum(one_intersection * threshold_uncertainty,dim=(1,2))/torch.sum(threshold_uncertainty,dim=(1,2))
        sumed_score_intersection = torch.sum(score_intersection).cpu()
        thr_score[i] = np.array([thr,sumed_score_intersection])
    thr_score = np.nan_to_num(thr_score)
    idx = np.argmax(thr_score, axis=1)[0]
    return thr_score[idx]



def select_uncertainty_key_batch(model,data_generator,mask_key):
    uncertainty_key_list = ['data_uncertainty_entropy','model_uncertainty_bald','model_varaiation_ratios','data_standard_deviation']
    y_pred,y_one,out = eval_data_generator(model,data_generator,mask_key)
    score = []
    for uncertainty_key in uncertainty_key_list:
        score.append(thr_score(y_pred,y_one,uncertainty_key,out,mask_key))
    score = np.array(score)
    return np.argmax(score[:,1],axis = 0)


def select_uncertainty_key_all(model,data_generator,mask_key):
    idx_list = []
    uncertainty_key_list = ['data_uncertainty_entropy','model_uncertainty_bald','model_varaiation_ratios','data_standard_deviation']
    for _ in range(int(len(data_generator.data_path_list)/data_generator.batch_size)):
        idx_list.append(select_uncertainty_key_batch(model,data_generator,mask_key))

    idx_frequent = max(set(idx_list), key=idx_list.count)
    return uncertainty_key_list[idx_frequent]



##########################################################################################################################
######################################## get shape uncertainty ###########################################################

def extract_object(original):
    original[original<0] = 0
    original = np.uint8(original)
    _, label_region_id_semantic_mask_all = cv2.connectedComponents(original,4, cv2.CV_32S)
    return label_region_id_semantic_mask_all 


def intersection_object_semantic_shape(label_region_id_object_mask_all,label_region_id_semantic_mask_all):
    object_idx_list = []
    for object_idx  in range(1,int(np.max(label_region_id_semantic_mask_all))+1):
      
        ind = label_region_id_semantic_mask_all == object_idx 

        new_object_idx = mode(label_region_id_object_mask_all[ind], keepdims = False)[0]
        ind2 = label_region_id_object_mask_all == new_object_idx 
    
        if (np.sum(ind*ind2)/(np.sum(ind2))) <0.95:
            object_idx_list.append(object_idx)
    return object_idx_list


def shape_uncertainty_each_img(sample,model):
    phase_img= sample['phase_img']
    out = prediction_unlabeled(model,phase_img)
    object_mask, semantic_mask = torch.argmax(out[1]['binary_mask'],axis = 1),torch.argmax(out[1]['semantic_mask'],axis = 1)

    num_shape_uncertainty = []
    for sample_idx in range(phase_img.shape[0]):
        original = semantic_mask[sample_idx].cpu()
        label_region_id_semantic_mask_all = extract_object(original)

        original = object_mask[sample_idx].cpu()
        original[original==2] = 0
        label_region_id_object_mask_all = extract_object(original)

        object_idx_list = intersection_object_semantic_shape(label_region_id_object_mask_all,label_region_id_semantic_mask_all)
        num_shape_uncertainty.append(len(object_idx_list))
    return num_shape_uncertainty


def idx_shape_uncertainty(data_generator,model):
    '''
    return the index that the shape is not the same in the predicted object_mask and sematic_mask
    '''
    shape_uncertainty = []
    for _ in range(int(len(data_generator.data_path_list)/data_generator.batch_size)):
        sample = next(data_generator)
        sample['binary_mask'] = sample['object_mask'] 
        num_shape_uncertainty = shape_uncertainty_each_img(sample,model)
        shape_uncertainty.append(num_shape_uncertainty)
     
    ind = np.array(shape_uncertainty)>0
    if np.sum(ind)/len(data_generator.data_path_list) > 0.2 :
        return list(np.array(list(range(len(data_generator.data_path_list))))[ind]),"continue"
    else:
        return list(range(len(data_generator.data_path_list))),"converged_in_shape"


    
    
    
    
##############################################################################################################################
######## The following select the uncertainty key that can most overlap with the false positive ##############################

def one_like_torch(mask_labeled):
    '''
    return the one_like in torch
    '''
    ind = mask_labeled<0
    y_one = torch.ones(mask_labeled.shape)
    y_one[ind] = 0
    return y_one

def remove_edge_torch(pred_region_id_object_mask_all):
    '''
    remove the edge effect, here edge is 10 pixel
    '''
    pred_region_id_object_mask_all_edged = torch.zeros(pred_region_id_object_mask_all.shape)
    pred_region_id_object_mask_all_edged[:,10:-10,10:-10] = pred_region_id_object_mask_all[:,10:-10,10:-10]
    return pred_region_id_object_mask_all_edged.cuda()

def adaptive_uncertainty_batch(model,data_generator,mask_key,adaptive_uncertainty,uncertainty_key_list):
    '''
    in one batch of the images, find out the uncertainty overlap with the FP and the total uncertainty values in each uncertainty key
    '''
    out,y_one,sample_selected = test_data_generator(model,data_generator,mask_key)
    pred_semantic_mask = torch.argmax(out[1][mask_key],axis = 1)
    label_semantic_mask = sample_selected[mask_key]
    diff_semantic_mask = remove_edge_torch(one_like_torch(pred_semantic_mask - label_semantic_mask)).cuda()
    for i in range(len(uncertainty_key_list)):
        uncertainty_key = uncertainty_key_list[i]
        uncertainty_map = uncertainty_images(uncertainty_key,out,mask_key,y_one).cuda()
        FP_uncertainty_sum = (torch.multiply(diff_semantic_mask,abs(uncertainty_map)))
        adaptive_uncertainty[i,0] += torch.sum(FP_uncertainty_sum).cpu()
        adaptive_uncertainty[i,1] += torch.sum(abs(uncertainty_map)).cpu()
    return adaptive_uncertainty


def adaptive_uncertainty_key_tot(model,data_generator,mask_key):
    '''
    mask_key: {'binary_mask','semantic_mask'}
    return the uncertainty_key name
    '''
    if data_generator.data_path_list:
        uncertainty_key_list = ['data_uncertainty_entropy','model_uncertainty_bald','model_varaiation_ratios','data_standard_deviation']
        adaptive_uncertainty = np.zeros((len(uncertainty_key_list),2))
        for iteration in range(int(len(data_generator.data_path_list)/data_generator.batch_size)):
            adaptive_uncertainty = adaptive_uncertainty_batch(model,data_generator,
                                                              mask_key,adaptive_uncertainty,uncertainty_key_list)
        ind = np.argmax(adaptive_uncertainty[:,0]/adaptive_uncertainty[:,1])
        return uncertainty_key_list[ind]
    
    else:
        return 'data_uncertainty_entropy'
    
    
    
def data_generator_parameters(path_list):
    path_to_data_sets = '/scratch/hangdeng/ActiveLearning/data_06-09-22/'
    paths_to_samples_dict = get_paths_to_training_and_validation_samples(path_to_data_sets)
    
    with open(path_to_data_sets+'semantic_class_idx_to_semantic_class_name.pkl', 'rb') as handle:
        semantic_class_idx_to_semantic_class_name = pkl.load(handle)
    class_name_to_class_idx_semantic = {class_name: idx for idx,class_name in semantic_class_idx_to_semantic_class_name.items()}
    class_name_to_class_idx_object = {'background': 0,'object': 1, 'border': 2}
    class_idx_to_class_name_object = {0: 'background', 1: 'object', 2: 'border'}
    class_idx_to_incidence_object,class_idx_to_incidence_semantic = get_class_idx_to_incidence(path_list)
    cuda = True if torch.cuda.is_available() else False
    loss_weights_tensor_object = get_loss_weights(class_idx_to_incidence_object,class_idx_to_prior=None,cuda=cuda)
    loss_weights_tensor_semantic = get_loss_weights(class_idx_to_incidence_semantic,class_idx_to_prior=None,cuda=cuda)

    class_name_to_loss_weights_semantic = {
        semantic_class_idx_to_semantic_class_name[idx]: weight for idx,weight in enumerate(loss_weights_tensor_semantic.cpu())}
    class_name_to_loss_weights_object = {
        class_idx_to_class_name_object[idx]: weight for idx,weight in enumerate(loss_weights_tensor_object.cpu())}

    return class_name_to_class_idx_semantic,class_name_to_class_idx_object,loss_weights_tensor_object,loss_weights_tensor_semantic


def build_model(model_type,complexity,n_models,weight_decay,data_generator):
    sorted_semantic_class_name_to_idx = {k: v for k, v in sorted(data_generator.semantic_class_name_to_idx.items(), key=lambda item: item[1])}
    sorted_object_class_name_to_idx = {k: v for k, v in sorted(data_generator.object_class_name_to_idx.items(), key=lambda item: item[1])}
    class_names_semantic = list(sorted_semantic_class_name_to_idx.keys())
    class_names_binary = list(sorted_object_class_name_to_idx.keys())

    if model_type=='pseudo_geometric':
        model=PseudoGeometricSegNet(n_models=n_models,in_ch=1,out_ch_mask=3,
                    out_ch_semantic=data_generator.n_classes,complexity=complexity,
                    n_low_res_layers=2,separate_decoder=False,class_names_semantic=class_names_semantic,
                    class_names_binary=class_names_binary)

    elif model_type=='geometric':
        model=EnsembleSegNet(n_models=n_models,in_ch=1,out_ch_mask=3,
                    out_ch_semantic=data_generator.n_classes,complexity=complexity,
                    n_low_res_layers=2,separate_decoder=False,geometric_model=True,
                    class_names_semantic=class_names_semantic,class_names_binary=class_names_binary)

    else:
        print('This model type is not implemented: ',model_type)

    n_params=count_parameters(model,trainable_only=True)
    print('num total params: ',n_params)
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        model.cuda()
    return model

