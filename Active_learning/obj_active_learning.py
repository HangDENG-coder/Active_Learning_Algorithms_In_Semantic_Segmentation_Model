from utils_semantic_segmentation import *
from acquisition_fuc import *
from uncertainty_object import *
from pynvml import *
import random
import sys
import os
cuda = True if torch.cuda.is_available() else False

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
    if cuda:
        model.cuda()
    return model






complexity=6#the larger the more paramter the model has
n_models=5#the number of independently trained models in the ensemble
weight_decay=1e-3#the weight of the L2 regularization of the parameters
batch_size = 20
selected_number = 40 ### the selected_number must smaller or equal to batch_size
mask_key = 'semantic_mask'
# uncertainty_key = 'model_uncertainty_bald'
uncertainty_key = 'data_uncertainty_entropy'
# uncertainty_key = 'model_varaiation_ratios'
path_to_data_sets = '/scratch/hangdeng/ActiveLearning/data_06-09-22/'
# path_to_data_sets = '/work/venkatachalamlab/Hang/active_learning/data_06-09-22'
save_folder = '/work/venkatachalamlab/Hang/active_learning/obj_models/data_entropy/'





paths_to_samples_dict = get_paths_to_training_and_validation_samples(path_to_data_sets)
class_name_to_class_idx_semantic,class_name_to_class_idx_object,loss_weights_tensor_object,loss_weights_tensor_semantic = data_generator_parameters(paths_to_samples_dict['paths_all'])
print('--- Data generator - Validation ---')
sys.stdout.flush()

data_generator_train = DataGenerator(paths_to_samples_dict['paths_train'],
                                       class_name_to_class_idx_semantic,
                                       class_name_to_class_idx_object,
                                       loss_weights_tensor_object,
                                       loss_weights_tensor_semantic,
                                       cuda=cuda,batch_size=20,
                                       augmentation_probability=0,
                                       steps_per_epoch=int(len(paths_to_samples_dict['paths_train'])/20),shuffle = False)

new_batch = new_batch_root(save_folder)
training_history = pkl_file_path(save_folder+'entropy_history.pkl')
selected_obj_sample = pkl_file_path(save_folder+'selected_obj_sample.pkl')
selected_uncertainty_pred_all = pkl_file_path(save_folder+'selected_uncertainty_pred_all.pkl')

criteria = "stop"
print("new_batch: ", new_batch)
sys.stdout.flush()
if new_batch == 0:
    model = build_model('pseudo_geometric',complexity,n_models,weight_decay,data_generator_train)
else:
    load_dir = save_folder + str(new_batch + 1) + '/' + os.listdir(save_folder + str(new_batch + 1))[0]
    model = load_model(load_dir,device='cuda')

    
    
    

while len(selected_obj_sample) < len(paths_to_samples_dict['paths_train']):
    data_generator_train = DataGenerator(paths_to_samples_dict['paths_train'],
                                       class_name_to_class_idx_semantic,
                                       class_name_to_class_idx_object,
                                       loss_weights_tensor_object,
                                       loss_weights_tensor_semantic,
                                       cuda=cuda,batch_size=20,
                                       augmentation_probability=0,
                                       steps_per_epoch=int(len(paths_to_samples_dict['paths_train'])/20),shuffle = False)

    new_selected_uncertainty_pred_all = obtain_selected_uncertainty_pred_all(model,data_generator_train,mask_key,uncertainty_key,selected_number,                                                                      list(range(len(paths_to_samples_dict['paths_train']))),selected_obj_sample )
    
    selected_uncertainty_pred_all = merge_obj_dic(new_selected_uncertainty_pred_all,selected_uncertainty_pred_all)
    selected_obj_sample = get_selected_obj_sample(selected_uncertainty_pred_all)
    print("untrained_sample_list",selected_obj_sample.keys(),selected_obj_sample)
    sys.stdout.flush()

    data_generator_selected = selected_DataGenerator(paths_to_samples_dict['paths_train'],
                                       class_name_to_class_idx_semantic,
                                       class_name_to_class_idx_object,
                                loss_weights_tensor_object,
                                       loss_weights_tensor_semantic,
                                       cuda=cuda,batch_size=20,
                                         augmentation_probability=0,
                                       steps_per_epoch=10,shuffle = False,selected_obj_sample = selected_obj_sample)

    data_generator_val = DataGenerator(paths_to_samples_dict['paths_val'],
                                       class_name_to_class_idx_semantic,
                                       class_name_to_class_idx_object,
                                       loss_weights_tensor_object,
                                       loss_weights_tensor_semantic,
                                       cuda=cuda,batch_size=20,
                                       augmentation_probability=0,
                                       steps_per_epoch=int(len(paths_to_samples_dict['paths_val'])/20),shuffle = True)

    performance_over_epochs = train_segmentation_ensemble_model(model,
                                                        data_generator_selected,loss_weights_tensor_object,
                                                                    loss_weights_tensor_semantic,
                                                                  data_generator_val=data_generator_val,
                                                                    num_epochs=10,print_every=1,
                                                                 lr=5e-4, weight_decay=weight_decay,
                                                                    with_gradient_clipping=False,transformations=None)
    


    training_history.update({new_batch:performance_over_epochs})
    save_dic_pkl(save_folder+'entropy_history.pkl',training_history)
    save_dic_pkl(save_folder+'selected_obj_sample.pkl',selected_obj_sample)
    save_dic_pkl(save_folder+'selected_uncertainty_pred_all.pkl',selected_uncertainty_pred_all)
    save_dic_pkl(save_folder+str(new_batch)+'/'+'new_selected_uncertainty_pred_all.pkl',new_selected_uncertainty_pred_all)

    new_batch += 1
    save_trained_model(save_folder+str(new_batch),model)


  
