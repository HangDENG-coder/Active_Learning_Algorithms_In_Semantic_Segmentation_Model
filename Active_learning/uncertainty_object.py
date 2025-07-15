

from utils_semantic_segmentation import *
from acquisition_fuc import *
import cv2
import scipy
from scipy.stats import mode


def test_data_generator(model,data_generator,mask_key):
    sample_selected = next(data_generator)    
    y_one = one_like_tensor(sample_selected[mask_key])
    out = prediction_unlabeled(model,sample_selected['phase_img'])
    return out,y_one,sample_selected

def test_extract_object(original_semantic_img):
    label_region_id_mask_all = np.zeros(original_semantic_img.shape)
    for class_idx in list(np.array(np.unique(original_semantic_img)))[1:None] :
        original_img = copy.deepcopy(original_semantic_img)
        original_img[original_img!=class_idx] = 0
        img = np.uint8(original_img)
        _, label_region_id_mask = cv2.connectedComponents(img,4, cv2.CV_32S)
        ind = label_region_id_mask>0
        label_region_id_mask_all[ind] = label_region_id_mask[ind] + np.max(label_region_id_mask_all)
    return label_region_id_mask_all

def remove_edge(pred_region_id_object_mask_all):
    pred_region_id_object_mask_all_edged = np.zeros(pred_region_id_object_mask_all.shape)
    pred_region_id_object_mask_all_edged[10:-10,10:-10] = pred_region_id_object_mask_all[10:-10,10:-10]
    return pred_region_id_object_mask_all_edged

def one_positive_array(mask_labeled):
    ind = mask_labeled>0
    y_one = torch.zeros(mask_labeled.shape)
    y_one[ind] = 1
    ind2 = mask_labeled<0
    y_one[ind2] = 0
    return y_one

def one_like_array(mask_labeled):
    ind = mask_labeled!=0
    y_one = np.zeros(mask_labeled.shape)
    y_one[ind] = 1
    return y_one



def each_object_uncertainty_dic(uncertainty_map,sample_idx,pred_region_id_object_mask_all,label_region_id_object_mask_all,iteration,batch_size,selected_obj_sample):
    '''
    the dictionary of each object uncertainty in each image 
    return the dictionary as {{‘img_num’: {‘obj_num’:{‘uncertainty’:uncertainty,‘labeled_obj_num’: number }}}
    '''

    uncertainty_map_img = np.array(uncertainty_map[sample_idx].cpu())
    uncertainty_pred_obj = {}
    a = list(np.unique(pred_region_id_object_mask_all))
    a.remove(0)

    img_idx = sample_idx+iteration*batch_size
    
    if img_idx in list(selected_obj_sample.keys()):
        a = list(set(a) - set(list(selected_obj_sample[img_idx]['pred_obj'])))   
        obj_filter = np.isin(label_region_id_object_mask_all,list(selected_obj_sample[img_idx]['label_obj']))
        label_region_id_object_mask_all = label_region_id_object_mask_all*(1-obj_filter)
    for i in a:
        ind = pred_region_id_object_mask_all == i
        
        uncertainty_obj = np.sum(uncertainty_map_img[ind])/np.sum(ind)
        
         
        label_obj = np.unique(label_region_id_object_mask_all[ind])
        
        label_obj = label_obj[label_obj >0 ]  #### for label_obj<=0, they are selected randomly cuase they are background
        
        if not list(label_obj):
            label_obj = 0
            uncertainty_obj = 0
            
            
        uncertainty_pred_obj.update({i:{'uncertainty':uncertainty_obj, 
                                   'label_obj': label_obj }})

    
    if len(uncertainty_pred_obj)==0:
        uncertainty_pred_obj = {0:{'uncertainty':0, 'label_obj': np.array([1])}}
    return {img_idx:{'uncertainty':uncertainty_pred_obj,
                       'num_label_obj':int(np.max(label_region_id_object_mask_all))}}




def each_object_uncertainty_dic_batch(out,mask_key,y_one,uncertainty_key,sample_selected,iteration,batch_size,selected_obj_sample):
    '''
    the dictionary of each object uncertainty in batch of images 
    '''
    uncertainty_map = uncertainty_image(out,mask_key,y_one,uncertainty_key)
    label_semantic_mask = sample_selected[mask_key]
    pred_semantic_mask = torch.argmax(out[1][mask_key],axis = 1)
    
    uncertainty_pred_batch = {}
    for sample_idx in range(batch_size):
        pred_region_id_object_mask_all = test_extract_object(remove_edge(pred_semantic_mask[sample_idx].cpu()))
        label_region_id_object_mask_all = test_extract_object(remove_edge(label_semantic_mask[sample_idx].cpu()))
            
        uncertainty_pred_obj = each_object_uncertainty_dic(uncertainty_map,sample_idx,pred_region_id_object_mask_all,
                                                           label_region_id_object_mask_all,iteration,batch_size,selected_obj_sample)
        uncertainty_pred_batch.update(uncertainty_pred_obj)
    return uncertainty_pred_batch



def uncertainty_ind_df_batch(uncertainty_pred_batch,selected_number):
    '''
    !!! select double size of selected_number among the uncertainty_pred_batch with highest uncertainty in the object
    return the index of df_array where the uncertainty is high 
    return the row list name to show object_id list for each image
    '''
    key_list = list(uncertainty_pred_batch.keys())
    df = pd.DataFrame.from_dict(uncertainty_pred_batch[key_list[0]]['uncertainty']).loc['uncertainty']

    for i in key_list[1:None]:
        a = pd.DataFrame.from_dict(uncertainty_pred_batch[i]['uncertainty'])
        df = pd.concat([df,a.loc['uncertainty']], axis=1)
  
    df_array = (df.fillna(0).to_numpy())

    num_largest = 2* selected_number + 1
    ind_positive = df_array>0
    if np.sum(ind_positive) > num_largest:
        value_largest = np.sort(df_array[df_array>0], axis=None)[-num_largest:][0]
        ind = np.where((df_array>value_largest))
    else:
        ind = np.where((df_array>0))
    return ind,df.axes[0].tolist()


def get_uncertainty_pred_batch(uncertainty_pred_batch,img_index,object_id_list):
    '''
    given the dictionary(uncertainty_pred_batch), which image(img_index), the selected object_id list in this image(object_id_list)
    return the selected uncertainty dictionary in the img_index
    '''
    a = {}
    for key in list(object_id_list):
        a.update( {key:uncertainty_pred_batch[img_index]['uncertainty'][key]} )
    return a


def get_selected_uncertainty_pred_batch(ind,row_name_list,uncertainty_pred_batch):
    '''
    given the index and the object_id list 
    return the selected uncertainty in one batch of predicted images
    '''
    object_id = np.array(row_name_list)[ind[0]] ### object_id list
    img_id = np.array(list(uncertainty_pred_batch.keys()))[ind[1]] ### image_id list

    selected_uncertainty_pred_batch = {}
    for img_index in list(np.unique(img_id)):
        object_id_list = list(object_id[img_id==img_index])
        selected_uncertainty_pred_batch.update( {img_index:{'uncertainty': 
                                                        get_uncertainty_pred_batch(uncertainty_pred_batch,img_index,object_id_list) ,
                                             'num_label_obj': uncertainty_pred_batch[img_index]['num_label_obj'] }} )
    return selected_uncertainty_pred_batch



def key_transfer(selected_uncertainty_pred_all,data_generator_sample_list):
    '''
    selected_uncertainty_pred_all has the image order (keys) assumed the data_generator's image list is natural number list
    while in the data_generator_sample_list, they are untrained image list with discontinuous number
    transfer the old keys into the new keys
    '''
    ind = np.array(list(selected_uncertainty_pred_all.keys()))
    key_new_list = list(np.array(data_generator_sample_list)[ind])
    key_old_list = list(selected_uncertainty_pred_all.keys())
    new_selected_uncertainty_pred_all = {}
    for i in range(len(key_new_list)):
        new_selected_uncertainty_pred_all[key_new_list[i]] = selected_uncertainty_pred_all[key_old_list[i]]
    return new_selected_uncertainty_pred_all




def obtain_selected_uncertainty_pred_all(model,data_generator,mask_key,uncertainty_key,selected_number,data_generator_sample_list,selected_obj_sample):
    '''
    !!! note: here the data_generator cannot shuffle the sample size
    mask key: 'semantic_mask'(mainly) or 'binary_mask'
    uncertainty_key: here are several uncertainty method
    uncertainty_key_list = ['data_uncertainty_entropy','model_uncertainty_bald','model_varaiation_ratios','data_standard_deviation']
    return selected_number of object with highest uncertainty among all the samples in the data_generator
    '''
    batch_size = data_generator.batch_size
    selected_uncertainty_pred_batch = {}
    for iteration in range(int(len(data_generator.data_path_list)/data_generator.batch_size)):
        out,y_one,sample_selected = test_data_generator(model,data_generator,mask_key)
        uncertainty_pred_batch = each_object_uncertainty_dic_batch(out,mask_key,y_one,uncertainty_key,sample_selected,iteration,batch_size,selected_obj_sample)
        uncertainty_pred_batch = dict(list(selected_uncertainty_pred_batch.items()) + list(uncertainty_pred_batch.items()) )
        ind,row_name_list = uncertainty_ind_df_batch(uncertainty_pred_batch,selected_number)
        selected_uncertainty_pred_batch = get_selected_uncertainty_pred_batch(ind,row_name_list,uncertainty_pred_batch)

    ind,row_name_list = uncertainty_ind_df_batch(selected_uncertainty_pred_batch,int(selected_number/2))
    selected_uncertainty_pred_all = get_selected_uncertainty_pred_batch(ind,row_name_list,selected_uncertainty_pred_batch)
    
    new_selected_uncertainty_pred_all = key_transfer(selected_uncertainty_pred_all,data_generator_sample_list)
    return new_selected_uncertainty_pred_all




def get_selected_obj_sample(selected_uncertainty_pred_all):
    selected_obj_sample = {}
    for img_id in list(selected_uncertainty_pred_all.keys()):

        obj_id_list = []
        pred_obj_list = list(selected_uncertainty_pred_all[img_id]['uncertainty'].keys())
        for obj_id in list(selected_uncertainty_pred_all[img_id]['uncertainty'].keys()):
            obj_id_list+=(list(selected_uncertainty_pred_all[img_id]['uncertainty'][obj_id]['label_obj']))
        
        selected_obj_sample.update({img_id:
                                    {'label_obj': np.unique(np.array(obj_id_list).flatten()),
                                    'pred_obj': np.array(pred_obj_list)} 
                                   })
    return dict(sorted(selected_obj_sample.items()))


def merge_obj_dic(new_selected_uncertainty_pred_all,selected_uncertainty_pred_all):
    '''
    merge the new selected object into the old selected object
    '''
    for new_key in list(new_selected_uncertainty_pred_all.keys()):
        if new_key in list(selected_uncertainty_pred_all.keys()):
            selected_uncertainty_pred_all[new_key]['uncertainty'].update(new_selected_uncertainty_pred_all[new_key]['uncertainty'])
        else:
            selected_uncertainty_pred_all[new_key] = new_selected_uncertainty_pred_all[new_key]
    return selected_uncertainty_pred_all


































