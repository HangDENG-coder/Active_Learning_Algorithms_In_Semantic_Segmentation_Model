
path_to_data_sets = '/scratch/hangdeng/ActiveLearning/data_06-09-22/'

model_type = 'pseudo_geometric'

folder = './trained_models/'

model_id = './best_train/22-06-26-13-03-50'

complexity=6#the larger the more paramter the model has
n_models=5#the number of independently trained models in the ensemble
weight_decay=1e-3#the weight of the L2 regularization of the parameters
batch_size = 20
selected_number = batch_size ### the selected_number must smaller or equal to batch_size

mask_key = 'semantic_mask'
uncertainty_key = 'model_uncertainty_bald'