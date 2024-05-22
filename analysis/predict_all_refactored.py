# this is a copy of the predict_all.ipynb

import pandas as pd
import numpy as np
import glob
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.model_selection import KFold
import argparse
from sklearn.model_selection import train_test_split
import tqdm 
import torch
from torch import nn
from skorch import NeuralNetRegressor

parser = argparse.ArgumentParser(description="Model Type Argument Parser")

parser.add_argument("--folder", type=str, required=True)
parser.add_argument("--subsample_rate", type=float, default=1.0)

args = parser.parse_args()



class Regressor(nn.Module):
    def __init__(self, input_dim):
        super(Regressor, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)





def predict(X, epochs=100):

    nr_samples, dim = X.shape
    
    
    scores_ridge = []
    scores_nnet = []
    
    all_dims = np.arange(dim)
    subsample_dims = np.random.choice(all_dims, int(args.subsample_rate * dim), replace=False)
    for i in (pbar := tqdm.tqdm(subsample_dims)):
        
        # now we build a train and val using a random split
        X_train, X_val = train_test_split(X, test_size=0.25, random_state=0)
        
        X_train = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-100)
        X_val = (X_val - X_val.mean(axis=0)) / (X_val.std(axis=0) + 1e-100)
        
        
        y_i_train = X_train[:,i]
        X_i_train = np.delete(X_train, i, 1)
    
        
        y_i_val = X_val[:,i]
        X_i_val = np.delete(X_val, i, 1)

        
        ridge_clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1,10])
            
        ridge_clf.fit(X_i_train, y_i_train)
        
        pred_ridge = ridge_clf.predict(X_i_val)
        
        ridge_error = mean_squared_error(y_i_val, pred_ridge)

        
        scores_ridge.append(ridge_error)
        
        
        nnet_clf = NeuralNetRegressor(
            Regressor(input_dim=X_i_train.shape[1]),
            max_epochs=epochs,
            lr=0.01,
            batch_size=512,
            device='cuda',
            train_split=None,
            optimizer=torch.optim.SGD,
            verbose=0,
        )
        # transform to torch tensor 
        X_i_train = torch.tensor(X_i_train, dtype=torch.float32).cuda()
        y_i_train = torch.tensor(y_i_train, dtype=torch.float32).cuda()
        # transform from [dim] to [dim, 1]
        y_i_train = y_i_train.unsqueeze(1)
        X_i_val = torch.tensor(X_i_val, dtype=torch.float32).cuda()
    
    

            
        nnet_clf.fit(X_i_train, y_i_train)
        
        pred_nnet = nnet_clf.predict(X_i_val)
        neural_error = mean_squared_error(y_i_val, pred_nnet)
        scores_nnet.append(neural_error)
        
        pbar.set_description(f"Ridge: {ridge_error:.4f}, Neural: {neural_error:.4f}")        
        
        
    scores_diff = np.array(scores_ridge) - np.array(scores_nnet)

    # mean_ridge, std_ridge, mean_nnet, std_nnet, mean_diff, std_diff 
    return np.mean(scores_ridge), np.std(scores_ridge), np.mean(scores_nnet), np.std(scores_nnet), np.mean(scores_diff), np.std(scores_diff)
        
# find all folders in the 'correlation_analysis' folder
# for every folder we extract the path, the model name, the wandb run id and the number and end up creating a dataframe
all_folders = glob.glob(args.folder+'/*')

folder_name = args.folder.replace('/','-')
print("new folder name: ", folder_name,flush=True)

print("all folders: ", all_folders,flush=True)

all_data = pd.DataFrame(columns=['path', 'model_name', 'dataset_name', 'data_split', 'wandb_run_id'])

for folder in all_folders:
    path = Path(folder)
    print(path)
    model_name = path.name 
    model_name = model_name.split('-')[0]
    dataset_name = path.name.split('-')[1].split('_')[0]
    data_split = path.name.split('_')[-1]
    wandb_run_id = path.name.split('_')[-2]
    all_data.loc[len(all_data)] = [path, model_name, dataset_name, data_split, wandb_run_id]
    






results = []
# Sequential processing of each row
for index, row in all_data.iterrows():
    print("Currently running: ", row['path'],flush=True)
    
    final_path = row['path']/Path('data_standardized.csv')
    X = pd.read_csv(final_path)
    X = X.to_numpy()
    # if X contains Nan values record this in the results file
    if np.isnan(X).any():
        row['nan_in_X'] = True
        # set them to 0
        X = np.nan_to_num(X)
    else:
        row['nan_in_X'] = False
        
        
    # first we calculate the sum of the covariance matrix
    correlation_path = row['path']/Path('correlation_no_NaN.csv')
    
    corr = (np.transpose(X) @ X) / X.shape[0]
    
    # save corr
    np.savetxt(correlation_path, corr, delimiter=",")
    
    # take mean and std of all absolute values of non-diagonal values
    mean_abs = np.mean(np.abs(corr[np.triu_indices_from(corr, k=1)]))
    std_abs = np.std(np.abs(corr[np.triu_indices_from(corr, k=1)]))
    
    # now we do the same but not absolute
    mean = np.mean(corr[np.triu_indices_from(corr, k=1)])
    std = np.std(corr[np.triu_indices_from(corr, k=1)])
    
    # store in row
    row['mean_abs_corr'] = mean_abs
    row['std_abs_corr'] = std_abs
    row['mean_corr'] = mean
    row['std_corr'] = std
    


    mean_ridge, std_ridge, mean_nnet, std_nnet, mean_diff, std_diff = predict(X, epochs=100)
    row[f'mean_ridge'] = mean_ridge
    row[f'std_ridge'] = std_ridge
    row[f'mean_nnet'] = mean_nnet
    row[f'std_nnet'] = std_nnet
    row[f'mean_diff'] = mean_diff
    row[f'std_diff'] = std_diff
    
    results.append(row)
    
    #store results in as a json file
    temp = pd.DataFrame(results)
    time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    temp.to_csv(f'TEMP_all_results_temp_{folder_name}_{time}.csv', index=False)



# Create a new DataFrame containing all the results
updated_dataframe = pd.DataFrame(results)

# also include timestamp
timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
updated_dataframe.to_csv(f'all_results_{timestamp}_{folder_name}.csv', index=False)
