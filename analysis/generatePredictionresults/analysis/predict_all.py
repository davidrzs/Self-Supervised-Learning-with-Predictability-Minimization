# this is a copy of the predict_all.ipynb

import pandas as pd
import numpy as np
import glob
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.model_selection import KFold
from multiprocessing import Pool, cpu_count
import argparse


parser = argparse.ArgumentParser(description="Model Type Argument Parser")

parser.add_argument("--model_type", type=str, required=True, help="Specify the model type as a string", choices=['linear', 'xgb', 'mlp'])
parser.add_argument("--folder", type=str, required=True)

args = parser.parse_args()

regress_model = args.model_type

def predict(X_train, X_val, model):

    nr_train_embeddings, dim = X_train.shape
    
    # normalize the embeddings in case they weren't already
    X_train = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-100)
    X_val = (X_val - X_val.mean(axis=0)) / (X_val.std(axis=0) + 1e-100)
    
    scores = []
    
    for i in range(dim):
        print(i)
        
        y_i_train = X_train[:,i]
        X_i_train = np.delete(X_train, i, 1)
    
        
        y_i_val = X_val[:,i]
        X_i_val = np.delete(X_val, i, 1)
        
        
        if model == 'linear':
            clf = SGDRegressor()
        elif model == 'xgb':
            clf = xgb.XGBRegressor(n_jobs=-1)  
        elif model == 'mlp':
            clf = MLPRegressor()
            
        clf.fit(X_i_train, y_i_train)
    
        scores.append(mean_squared_error(y_i_val, clf.predict(X_i_val)))
        
    return np.mean(scores), np.std(scores)
        

all_data = pd.DataFrame(columns=['path', 'model_name', 'dataset_name', 'data_split', 'wandb_run_id'])

# for folder in all_folders:
path = Path(args.folder)
print("PATH IS", path)
print("PATH NAME IS", path.name)
model_name = path.name 
model_name = model_name.split('-')[0]
dataset_name = path.name.split('-')[1].split('_')[0]
data_split = path.name.split('_')[-1]
wandb_run_id = path.name.split('_')[-2]
all_data.loc[len(all_data)] = [path, model_name, dataset_name, data_split, wandb_run_id]

f_name = (args.folder).replace('/','_')



def worker_function(args):

    index, row = args
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

    kf = KFold(n_splits=5)

    
    for i, (train, test) in enumerate(kf.split(X)):
            mean, std = predict(X[train], X[test], model=regress_model)
            row[f'mean_{regress_model}_{i}'] = mean
            # print with timestamp
            print(f"mean_{regress_model}_{i}: {mean} at {pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}",flush=True)
            row[f'std_{regress_model}_{i}'] = std
            # print with timestamp
            print(f"std_{regress_model}_{i}: {std} at {pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}",flush=True)
        
    return index, row


print("about to run now",flush=True)

# Initialize an empty list to store the updated rows
updated_rows = []

# Iterate over the DataFrame rows using iterrows()
for index, row in all_data.iterrows():
    # Apply the worker function to each row
    result = worker_function((index, row))
    # Collect the updated rows
    updated_rows.append(result[1])

# Create a new DataFrame containing all the results
updated_dataframe = pd.DataFrame(updated_rows)

# Include timestamp
timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
updated_dataframe.to_csv(f'all_results_{regress_model}_{timestamp}_{f_name}.csv', index=False)

