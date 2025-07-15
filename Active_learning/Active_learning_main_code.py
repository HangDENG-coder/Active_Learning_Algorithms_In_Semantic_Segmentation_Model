from utils_semantic_segmentation_active import *
from acquisition_fuc import *
from pynvml import *
import random
import sys
import os
cuda = True if torch.cuda.is_available() else False




complexity=6#the larger the more paramter the model has
n_models=5#the number of independently trained models in the ensemble
weight_decay=1e-3#the weight of the L2 regularization of the parameters
batch_size = 20
selected_number = 100 ### the selected_number must smaller or equal to batch_size

### here mainly use the semantic_mask to compute the uncertainty
mask_key = 'semantic_mask'
# mask_key = 'binary_mask'


#### select the uncertainty method from 'model_uncertainty_bald','data_uncertainty_entropy','model_varaiation_ratios',random_select','data_standard_deviation','least_confidence','mean_standard_deviation'

uncertainty_key = 'mean_standard_deviation'

path_to_data_sets = '/data_06-09-22/'
### chose the folder path to save the training results
save_folder = '/work/Hang/active_learning/adaptive_tensor/' + uncertainty_key +'/'





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


def selected_data_generator_parameters(selected_ind):
    path_to_data_sets = '/data_06-09-22/'
    paths_to_samples_dict = get_paths_to_training_and_validation_samples(path_to_data_sets)
    total_path_list = list(np.array(paths_to_samples_dict['paths_train'])[np.array(selected_ind)]) +paths_to_samples_dict['paths_val']
    
    with open(path_to_data_sets+'semantic_class_idx_to_semantic_class_name.pkl', 'rb') as handle:
        semantic_class_idx_to_semantic_class_name = pkl.load(handle)
    class_name_to_class_idx_semantic = {class_name: idx for idx,class_name in semantic_class_idx_to_semantic_class_name.items()}
    class_name_to_class_idx_object = {'background': 0,'object': 1, 'border': 2}
    class_idx_to_class_name_object = {0: 'background', 1: 'object', 2: 'border'}
    class_idx_to_incidence_object,class_idx_to_incidence_semantic = get_class_idx_to_incidence(total_path_list)
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

def all_uncertainty(model,data_generator_untrained,selected_number,mask_key,uncertainty_key):
    uncertainty = uncertainty_tot(model,data_generator_untrained,mask_key,uncertainty_key)
    return uncertainty

def combine_two_uncertainty(model,data_generator,selected_number,uncertainty_key):
    semantic_uncertainty = all_uncertainty(model,data_generator,selected_number,'semantic_mask',uncertainty_key)
    binary_uncertainty = all_uncertainty(model,data_generator,selected_number,'binary_mask',uncertainty_key)
    tot_uncertainty = semantic_uncertainty + binary_uncertainty 
    return tot_uncertainty









paths_to_samples_dict = get_paths_to_training_and_validation_samples(path_to_data_sets)
# class_name_to_class_idx_semantic,class_name_to_class_idx_object,loss_weights_tensor_object,loss_weights_tensor_semantic = data_generator_parameters(paths_to_samples_dict['paths_all'])
class_name_to_class_idx_semantic = pkl_file_path('/class_weights/'+ 'class_name_to_class_idx_semantic.pkl')
class_name_to_class_idx_object = pkl_file_path('/class_weights/'+ 'class_name_to_class_idx_object.pkl')
loss_weights_tensor_object = pkl_file_path('/class_weights/'+ 'loss_weights_tensor_object.pkl')
loss_weights_tensor_semantic = pkl_file_path('/class_weights/'+ 'loss_weights_tensor_semantic.pkl')

print('--- Data generator - Validation ---')
sys.stdout.flush()

data_generator_train = DataGenerator(paths_to_samples_dict['paths_train'],
                                     class_name_to_class_idx_semantic,class_name_to_class_idx_object,
                            loss_weights_tensor_object,loss_weights_tensor_semantic,cuda=cuda,batch_size=20,
                 augmentation_probability=0,steps_per_epoch=int(len(paths_to_samples_dict['paths_train'])/20))





new_batch = new_batch_root(save_folder)
training_history = pkl_file_path(save_folder + 'entropy_history.pkl')
selected_history = pkl_file_path(save_folder + 'selected_history.pkl')
uncertainty_key_history = pkl_file_path(save_folder + 'uncertainty_history.pkl')
selected_training_sample_list  = list(np.array(list(pkl_file_path(save_folder + 'selected_history.pkl').values())).flatten())
all_sample_list = list(range(len(paths_to_samples_dict['paths_train'])))
untrained_sample_list = list(set(all_sample_list) - set(selected_training_sample_list ))
untrained_sample_list = list(set(untrained_sample_list) )


if selected_training_sample_list:
    data_generator_selected = DataGenerator(selected_path_list(selected_training_sample_list),
                                            class_name_to_class_idx_semantic,
                                         class_name_to_class_idx_object,
                                loss_weights_tensor_object,loss_weights_tensor_semantic,cuda=cuda,
                                            batch_size = 20,
                                         augmentation_probability=0,
                                            steps_per_epoch=int(len(selected_training_sample_list)/20),shuffle = True )


print("new_batch: ", new_batch)
sys.stdout.flush()
if new_batch == 0:
    model = build_model('pseudo_geometric',complexity,n_models,weight_decay,data_generator_train)
    optimizers_old = {}
else:
    load_dir = save_folder + str(new_batch + 1) + '/' + os.listdir(save_folder + str(new_batch + 1))[0]
    model,optimizers_old = load_model(load_dir,device='cuda')

while len(untrained_sample_list) > selected_number:
    
    if len(untrained_sample_list) > 400:
        batch_size = 400
    else: 
        batch_size = 20 
        
    data_generator_untrained = DataGenerator(selected_path_list(untrained_sample_list),class_name_to_class_idx_semantic,
                                         class_name_to_class_idx_object,
                                loss_weights_tensor_object,loss_weights_tensor_semantic,cuda=cuda,batch_size = batch_size,
                     augmentation_probability=0,steps_per_epoch=int(len(untrained_sample_list)/batch_size),shuffle = False)
    

    shape_uncertainty_list = copy.deepcopy(untrained_sample_list)
        
        
        
    data_generator_val = DataGenerator(paths_to_samples_dict['paths_val'],
                                       class_name_to_class_idx_semantic,class_name_to_class_idx_object,
                            loss_weights_tensor_object,loss_weights_tensor_semantic,cuda=cuda,batch_size = 20,
                 augmentation_probability=0,steps_per_epoch=int(len(paths_to_samples_dict['paths_val'])/20))
        
       
    
    
    print("uncertainty_key: ", uncertainty_key)
    sys.stdout.flush()
    
    data_generator_unshaped = DataGenerator(selected_path_list(shape_uncertainty_list),class_name_to_class_idx_semantic,
                                         class_name_to_class_idx_object,
                                loss_weights_tensor_object,loss_weights_tensor_semantic,cuda=cuda,batch_size = batch_size,
                     augmentation_probability=0,steps_per_epoch=int(len(shape_uncertainty_list)/batch_size),shuffle = False)

    if uncertainty_key!= "random_select":
        
        data_generator_unshaped = DataGenerator(selected_path_list(shape_uncertainty_list),class_name_to_class_idx_semantic,
                                         class_name_to_class_idx_object,
                                loss_weights_tensor_object,loss_weights_tensor_semantic,cuda=cuda,batch_size = batch_size,
                     augmentation_probability=0,steps_per_epoch=int(len(shape_uncertainty_list)/batch_size),shuffle = False)
        semantic_uncertainty = all_uncertainty(model,data_generator_unshaped,selected_number,'semantic_mask',uncertainty_key)
       
        z_score = (semantic_uncertainty-torch.mean(semantic_uncertainty))/(torch.std(semantic_uncertainty))
        ind = torch.where(z_score > 3)
        semantic_uncertainty[ind] = 0
        tot_uncertainty = semantic_uncertainty
        
        
        selected_idx = np.argpartition(np.array(tot_uncertainty.cpu()), -selected_number)[-selected_number:]
        selected_training_sample = list(np.array(shape_uncertainty_list)[selected_idx])
        selected_training_sample_list += selected_training_sample
        
        
    else:
        selected_training_sample = random.choices(untrained_sample_list, k = selected_number)
        selected_training_sample_list += selected_training_sample
    
    print("selected_training_sample_list",(selected_training_sample))
    sys.stdout.flush()

    untrained_sample_list = list(set(list(range(len(paths_to_samples_dict['paths_train'])))) - set(selected_training_sample_list))   
   
    #### train the selected_training_sample for one epoch
    
    class_name_to_class_idx_semantic,class_name_to_class_idx_object,loss_weights_tensor_object,loss_weights_tensor_semantic = selected_data_generator_parameters(selected_training_sample_list)
    data_generator_selected = DataGenerator(selected_path_list(selected_training_sample_list),
                                            class_name_to_class_idx_semantic,
                                         class_name_to_class_idx_object,
                                loss_weights_tensor_object,loss_weights_tensor_semantic,cuda=cuda,
                                            batch_size = 20,
                                         augmentation_probability=1,
                                            steps_per_epoch=int(len(selected_training_sample_list)/20),shuffle = True )

    performance_over_epochs,optimizers_new = train_segmentation_ensemble_model_active(model,optimizers_old,
                                                    data_generator_selected,loss_weights_tensor_object,
                                                                loss_weights_tensor_semantic,
                                                              data_generator_val=data_generator_val,
                                                                num_epochs=10,print_every=1,
                                                             lr=5e-4, weight_decay=weight_decay,
                                                                with_gradient_clipping=False,transformations=None)
    
    #### remove the trained samples from untrainned
    
    print("untrained_sample_list",len(untrained_sample_list))
    sys.stdout.flush()
    training_history.update({new_batch:performance_over_epochs})
    selected_history.update({new_batch:selected_training_sample})
    uncertainty_key_history.update({new_batch:uncertainty_key})
    
    save_dic_pkl(save_folder+'entropy_history.pkl',training_history)
    save_dic_pkl(save_folder+'selected_history.pkl',selected_history)
    save_dic_pkl(save_folder+'uncertainty_history.pkl',uncertainty_key_history)
    
    new_batch += 1
    save_trained_model(save_folder+str(new_batch),model,optimizers_new)
    optimizers_old = optimizers_new
    
    
