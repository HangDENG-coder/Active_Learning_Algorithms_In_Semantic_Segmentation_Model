import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

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
    img = np.uint8(img)
    num_labels, label_region_id_mask = cv2.connectedComponents(img,4,cv2.CV_32S)
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
    
    
##### visualizaiton###########
def plot_each_object_phase_img(img,pred_region_id_mask):
    '''
    choose the sample_idx from the phase_img to plot
    plot each labeled object seperatly 
    '''
    num_labels, label_region_id_mask = label_object_image(img.astype('int8'))
    for i in range(1,num_labels):
        test = np.zeros(pred_region_id_mask.shape)
        ind = (label_region_id_mask== i)
        test[ind] = np.array(img)[ind]
        plt.imshow(test,'gray')
        plt.title("object " + str(i))
        plt.show()
        
        
def plot_each_object_label(img,object_ind,label_region_id_mask,pred_region_id_mask):
    '''
    comparet each object between the phase_img, true_labeled, predicted
    '''
    label_ind = (label_region_id_mask==object_ind)
    pred_id = np.unique(pred_region_id_mask[label_ind])
    pred_ob = pred_object(pred_region_id_mask,pred_id)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax3.imshow(pred_ob)
    ax3.set_title("pred")


    test = np.zeros(pred_ob.shape)
    test[label_ind] = 1
    ax2.imshow(test)
    ax2.set_title("label")


    test = np.zeros(pred_ob.shape)
    test[label_ind] = img[label_ind]
    ax1.imshow(test,'gray')
    ax1.set_title("original image")
    plt.show()
    
    
    
def plot_all_objects_phase_label_pred(img,pred_semantic_mask,semantic_mask,pred_region_id_mask):
    '''
    To plot each object in the original phase_img, semantic_mask, predicted_semantic_mask
    img: one image labeled as semantic_mask
    pred_semantic_mask: 
    
    '''
    num_labels, label_region_id_mask = label_object_image(img.astype('int8'))
    for object_ind in range(1,num_labels):
        label_ind = (label_region_id_mask==object_ind)
        pred_id = np.unique(pred_semantic_mask[label_ind])
        pred_id = pred_id[pred_id != 0]
        print("object", object_ind , ": labeled class", np.unique(semantic_mask[label_ind].cpu()),", predicted class",pred_id)
        plot_each_object_label(img,object_ind,label_region_id_mask,pred_region_id_mask)

        
#######################################################################################################################
#################################### Confusion Matrix in the object level after the post processing####################

def out_each_batch(model,phase_img):
    model.eval()#make sure that the model is in eval mode (applying learned batch-norm parameters)
    with torch.no_grad():#no gradient calculation -> speed-up for inference
        out = model(phase_img)
    return out

def processed_frame_img(phase_img,out,sample_idx):
    '''
    post_process the sample_idx phase_img
    '''
    pred_segmentation_output =  model_prediction.segmentation_out_idx(phase_img,out,sample_idx,"ensembled" )

    pred_processed_frame = model_prediction.get_processed_segmentation_output( pred_segmentation_output,
                                                                         model_prediction.model_class_idx_to_class_name_semantic,
                                                                         class_idx_to_min_max_feature_value = None,
                                            region_mode = 'object_mask_pooling',
                                                                         cuda = model_prediction.cuda,
                                                                              minimum_probability=0,
                                                                         min_object_size = 2, 
                                                                         remove_small_holes = 3,
                                                                         connectivity=1)
    ### get the dictionary type of each object
    pred_processed_frame.convert_semantic_regions_to_object_and_semantic_mask()
    return pred_processed_frame

def confusion_each_object(img,label_region_id_mask,object_ind,pred_semantic_mask):
    label_ind = (label_region_id_mask==object_ind)
    values, counts = np.unique(pred_semantic_mask[label_ind], return_counts=True)
    class_idx = np.unique(img[label_ind])[0]
    return class_idx,values, counts

def update_confusion_matrix_each_obejct(confusion_matrix,class_idx,values, counts):
    confusion_matrix[class_idx][values.astype(int)] += counts/np.sum(counts)
    ### count how many times the class_idx should be predicted
    confusion_matrix[class_idx][-1] += 1
    return confusion_matrix

def update_confusion_matrix_each_img(sample_idx,confusion_matrix,img,pred_processed_frame):
    pred_semantic_mask = class_id_mask(img,pred_processed_frame.semantic_regions)
    pred_region_id_mask = region_id_mask(img,pred_processed_frame.semantic_regions)
    num_labels, label_region_id_mask = label_object_image(img)

    for object_ind in range(1,num_labels):
        class_idx,values, counts = confusion_each_object(img,label_region_id_mask,object_ind,pred_semantic_mask)
        confusion_matrix = update_confusion_matrix_each_obejct(confusion_matrix,class_idx,values, counts)
    return confusion_matrix


def update_confusion_matrix_batch(confusion_matrix,sample,model):
    phase_img, object_mask, semantic_mask = sample['phase_img'], sample['object_mask'], sample['semantic_mask']
    out = out_each_batch(model,phase_img)
    for sample_idx in range(len(semantic_mask)):
        img = np.array(semantic_mask[sample_idx].cpu())
        pred_processed_frame = processed_frame_img(phase_img,out,sample_idx)
        confusion_matrix = update_confusion_matrix_each_img(sample_idx,confusion_matrix,img,pred_processed_frame)
    return confusion_matrix

### initialize the confusion_matrix
def confusion_matrix(data_generator,model):
    n_class = data_generator.n_classes + 1
    confusion_matrix = np.zeros((n_class, n_class+1))
   
    
    tot_samples = len(data_generator.data_path_list)
    # tot_samples = 20
    t0 = time.time()
    for num_batch in range(tot_samples//data_generator.batch_size):
        sample = next(data_generator)
        confusion_matrix = update_confusion_matrix_batch(confusion_matrix,sample,model)
    t1 = time.time()
    print("time",t1-t0)
    confusion_matrix_class = np.nan_to_num(confusion_matrix / confusion_matrix[:,-1][:,None])[:,0:-1]
    return confusion_matrix_class


def Plot_confusion_matrix_object(confusion_matrix_class):
    im = plt.imshow(confusion_matrix_class, cmap='hot', interpolation='nearest')
    plt.xticks(list(range(16)))
    plt.yticks(list(range(16)))
    plt.xlabel("Predicted class")
    plt.ylabel("Manual Label",loc = "center")
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    plt.title("confusion matrix in object level")
    plt.colorbar(im)
    plt.show()

