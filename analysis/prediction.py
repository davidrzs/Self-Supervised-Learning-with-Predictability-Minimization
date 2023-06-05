import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from tqdm import tqdm




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Redundancy Evaluator')
    # we need the following arguments
    # path of the embeddings
    # path of the labels
    # model: choices=['linear','svm']
    parser.add_argument('--path_train', type=str, help='path of the train embeddings')
    parser.add_argument('--path_val', type=str, help='path of the val embeddings')
    parser.add_argument('--model', type=str, choices=['linear','svm'], help='model to use')

    args = parser.parse_args()
    
    # print args in case we are on a cluster and want to inspect the output file
    print(args)
    
    X_train = np.loadtxt(args.path_train, delimiter=',')
    
    X_val = np.loadtxt(args.path_val, delimiter=',')
    
    nr_train_embeddings, dim = X_train.shape
    
    # normalize the embeddings in case they weren't already
    # X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    # X_val = (X_val - X_val.mean(axis=0)) / X_val.std(axis=0)
    

    if args.model == 'linear':
        clf = SGDRegressor()
    elif args.model == 'svm':
        clf = SVR()
    
    scores = []
    
    pb = tqdm(range(dim))
        
    for i in pb:
        
        y_i_train = X_train[:,i]
        X_i_train = np.delete(X_train, i, 1)

        
        y_i_val = X_val[:,i]
        X_i_val = np.delete(X_val, i, 1)
        
        print(X_val.shape)
        
        
        clf = SGDRegressor()

        clf.fit(X_i_train, y_i_train)

        scores.append(mean_squared_error(y_i_val, clf.predict(X_i_val)))
        
        pb.set_description(f'Mean: {np.mean(scores)}, Std: {np.std(scores)}')

        
    # print mean and std of the scores
    print(f'Mean: {np.mean(scores)}, Std: {np.std(scores)}')
