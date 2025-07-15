


import numpy as np
import cv2
import matplotlib.pyplot as plt


def region_id_mask(img_np,semantic_regions):
	#### mark each object with region id

	region_id_mask = np.zeros((img_np.shape))
	for ind,region_dict in semantic_regions.items():
		x = region_dict['coords'][:,0]
		y = region_dict['coords'][:,1]
		region_id_mask[x,y] = region_dict['region_id']
	region_id_mask_cut_edge = np.zeros((img_np.shape))
	region_id_mask_cut_edge[10:region_id_mask.shape[0]-10,10:region_id_mask.shape[0]-10] = region_id_mask[10:region_id_mask.shape[0]-10,10:region_id_mask.shape[0]-10]
    
	return region_id_mask_cut_edge

def pred_object(pred_region_id_mask,pred_id):
    #### find the expected object area with region_id(pred_id)
    pred_object = np.zeros(pred_region_id_mask.shape)    
    for i in pred_id:
        ind = np.where(pred_region_id_mask == i)
        pred_object[ind] = pred_region_id_mask[ind]
    return pred_object
    
    
def object_class(region_id_mask,label_ind):
    object_class = np.unique(region_id_mask[label_ind])
    object_class = object_class[object_class != 0]
    return object_class    
    

def label_object_image(img):
    '''    
    img must unit8
    find each object in the labeled images
    num_labels: number of objects
    label_region_id_mask: region_id_mask
    '''
    ind = img < 0
    img[ind] = 0
    num_labels, label_region_id_mask = cv2.connectedComponents(img)
    return num_labels, label_region_id_mask

def class_id_mask(img_np,semantic_regions):
    '''
    remove the edge in each image   
    mark each object with class id
    '''
    
    class_id_mask = np.zeros(img_np.shape)
    for ind,region_dict in semantic_regions.items():
        x = region_dict['coords'][:,0]
        y = region_dict['coords'][:,1]
        class_id_mask[x,y] = region_dict['class_idx']
    class_id_mask_cut_edge = np.zeros((img_np.shape))
    class_id_mask_cut_edge[10:class_id_mask.shape[0]-10,10:class_id_mask.shape[0]-10] = class_id_mask[10:class_id_mask.shape[0]-10,10:class_id_mask.shape[0]-10]

    return class_id_mask_cut_edge
    

    
######