import pandas as pd
import os
import matplotlib.pyplot as plt
from utils_semantic_segmentation_active import *
from acquisition_fuc import *

class_name_to_class_idx_semantic = pkl_file_path('class_weights/'+ 'class_name_to_class_idx_semantic.pkl')
class_name_to_class_idx_object = pkl_file_path('class_weights/'+ 'class_name_to_class_idx_object.pkl')
loss_weights_tensor_object = pkl_file_path('class_weights/'+ 'loss_weights_tensor_object.pkl')
loss_weights_tensor_semantic = pkl_file_path('class_weights/'+ 'loss_weights_tensor_semantic.pkl')
data_generator_val = DataGenerator(paths_to_samples_dict['paths_val'],
                                   class_name_to_class_idx_semantic,class_name_to_class_idx_object,
                        loss_weights_tensor_object,loss_weights_tensor_semantic,cuda=cuda,batch_size = 20,
             augmentation_probability=0,steps_per_epoch=int(len(paths_to_samples_dict['paths_val'])/20))

def train_history_pklfile(pklfile):
    return pd.read_pickle(pklfile)
                          
                          
def model_loss_history(train_history,model_type,loss_type,single_model_nuber = False):
    '''
    model type: 'single', 'ensemble'
    single_model_nuber: the each single model member number
    loss_type: 'train_binary_losses', 'train_semantic_losses', 'val_binary_losses', 'val_semantic_losses'
    '''
    loss = []
    for i in range(len(train_history.keys())):
        if model_type == 'single':
            loss.append(train_history[i][single_model_nuber][model_type][loss_type][-1])
        elif model_type == 'ensemble':
            # ensemble_all = list(train_history[0].keys())[-1]
            # loss.append(train_history[i][ensemble_all][model_type][loss_type][0])
            loss.append(train_history[i][single_model_nuber][model_type][loss_type][-1])
    return loss


def file_list_dir_extension(dir_path, extension_type):
    file_list = []
    for file in os.listdir(dir_path):
        if file.endswith(extension_type):
            file_list.append(os.path.join(dir_path, file))
    return file_list


def plot_loss_model(file_list,model_type,loss_type,single_model_nuber):
    # legend_list = [os.path.splitext(os.path.split(file)[1])[0] for file in file_list]
    if not file_list:
        print("the file list is empty")
    else:
#         legend_list = [os.path.split(file)[1].split('./')[0] for file in file_list]
        legend_list = [os.path.split(os.path.split(file)[0])[1] for file in file_list]
        x_lim_list = []
        for i in range(len(file_list)):
            train_history = train_history_pklfile(file_list[i])
            loss = model_loss_history(train_history,model_type,loss_type,single_model_nuber = single_model_nuber)
            plt.plot(loss,label=legend_list[i])    
            x_lim_list.append(len(loss))

        plt.xlabel("batch")
        plt.xlim((0, min(x_lim_list)))
        # plt.ylim((0, 15))
        plt.ylabel(loss_type)
        # plt.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(loc='upper right')
        if single_model_nuber == False:
            plt.title(model_type + " model active learning training history")
        else:
            plt.title(model_type + " model "+str(single_model_nuber)+ " active learning training history")
        plt.show()

        
################################################################################################################################ 
############################################# compute the accuracy history per epoch ###########################################        
def acc_list(save_folder,epoch,data_generator,pixel_acc_history,obj_acc_history ):
    '''
    compute the accuracy with the model at epoch 
    '''
    load_dir = save_folder + str(epoch + 1) + '/' + os.listdir(save_folder + str(epoch + 1))[0]
    model,optimizers_old = load_model(load_dir,device='cuda')

    confusion_matrix_pixel_val,confusion_matrix_object_val,n_pixels_tot_val,n_pixels_correct_val,labels_in_val,probabilities_in_val=get_performance_statistics(
        data_generator_val,model,n_batches=int(len(paths_to_samples_dict['paths_val'])/20))
    
    pixel_acc_history.append(n_pixels_correct_val/n_pixels_tot_val)

    obj_acc = []
    '''
    here there are 16 classes, it may change
    '''
    for i in range(16):
        obj_acc.append(get_top_k_accuracy(probabilities_in_val,labels_in_val,K=i))
    obj_acc_history.append(obj_acc)
    
    return pixel_acc_history,obj_acc_history 


def acc_list_folder(save_folder):
    '''
    compute the accuracy history training with the epoch growing
    '''
    obj_acc_history = []
    pixel_acc_history = []
    epoch_tot = new_batch_root(save_folder)
    for epoch in range(epoch_tot):
        data_generator = DataGenerator(paths_to_samples_dict['paths_val'],
                                           class_name_to_class_idx_semantic,class_name_to_class_idx_object,
                                loss_weights_tensor_object,loss_weights_tensor_semantic,cuda=cuda,batch_size = 20,
                     augmentation_probability=0,steps_per_epoch=int(len(paths_to_samples_dict['paths_val'])/20))

        pixel_acc_history,obj_acc_history = acc_list(save_folder,epoch,data_generator,pixel_acc_history,obj_acc_history)
    return pixel_acc_history,obj_acc_history




















