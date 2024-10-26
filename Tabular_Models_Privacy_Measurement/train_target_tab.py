import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import random
import os
import re
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from models.MLP_classifier import *
import joblib

def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--continuous_col', nargs='+', default=['age', 'capital_gain', 'capital_loss'], help='List of continuous columns')
    parser.add_argument('--target', type=str, default='over_50k', help='Target variable')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data_pth', type=str, default='/home/hadrien/data/adult_new/adult_clean.csv', help='Path to the  dataset')
    parser.add_argument('--save_data_folder', type=str, default='./data', help='Path to save datasets')
    parser.add_argument('--random_seed', type=str, default=42, help='random seed for consistancy')
    parser.add_argument('--mia_train_size', type=int, default=13024, help='Size of MIA training set')
    parser.add_argument('--mia_test_size', type=int, default=13024, help='Size of MIA test set')
    parser.add_argument('--gen_train_size', type=int, default=13024, help='Size of GAN training set')
    parser.add_argument('--name', type=str, default='default', help='Name of the experiment')
    return parser.parse_args()

def load_data(pth, save_pth, continuous_col, mia_test, mia_train, gen_train, seed=42, save=True):
    """
    Load and preprocess data for training and testing.

    Parameters:
    pth (str): Path to the directory containing the dataset file 'adult_clean.csv'.
    continuous_col (list of str): List of column names that contain continuous data.
    mia_test (int): Number of samples to be used for MIA testing.
    mia_train (int): Number of samples to be used for MIA training.
    gen_train (int): Number of samples to be used for GAN training.

    Returns:
    tuple: A tuple containing:
        - df_train (pd.DataFrame): The training split of the dataset.
        - df_test (pd.DataFrame): The test split of the dataset.
        - df_gen (pd.DataFrame): The subset of the dataset for generative model training.
        - df_mia_tr (pd.DataFrame): The subset of the dataset for MIA training.
        - df_mia_te (pd.DataFrame): The subset of the dataset for MIA testing.

    Raises:
    AssertionError: If the sum of mia_test, mia_train, and gen_train does not equal the total number of samples in the dataset.
    """
    try:
        df_train= pd.read_csv(os.path.join(save_pth, 'train.csv'))
        df_test= pd.read_csv(os.path.join(save_pth, 'test.csv'))
        df_gen= pd.read_csv(os.path.join(save_pth, 'gen_train.csv'))
        df_mia_tr= pd.read_csv(os.path.join(save_pth, 'mia_train.csv'))
        df_mia_te= pd.read_csv(os.path.join(save_pth, 'mia_test.csv'))
    except:
        df= pd.read_csv(pth)
        print('df', df.shape)
        df_train, df_test= train_test_split(df, test_size=.2, random_state=seed)
        kbin= KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
        kbin.fit(df_train[continuous_col])  
        df_train= pd.concat([df_train.drop(columns=continuous_col).reset_index(drop=True),
                        pd.DataFrame(kbin.transform(df_train[continuous_col]), columns= continuous_col).reset_index(drop=True)], axis=1)
        df_test= pd.concat([df_test.drop(columns=continuous_col).reset_index(drop=True),
                        pd.DataFrame(kbin.transform(df_test[continuous_col]), columns= continuous_col).reset_index(drop=True)], axis=1)
        dataset_len= df_train.shape[0]
        assert mia_test + mia_train+ gen_train == dataset_len, f'problem with dataset sizes {mia_test + mia_train+ gen_train} !={dataset_len}'
        df_gen= df.iloc[:gen_train,:]
        df_mia_tr= df.iloc[gen_train:gen_train+mia_train,:]
        df_mia_te= df.iloc[-mia_test:,:]
        if save:
            df_train.to_csv(os.path.join(save_pth, 'train.csv'), index=False)
            df_test.to_csv(os.path.join(save_pth, 'test.csv'), index=False)
            df_gen.to_csv(os.path.join(save_pth, 'gen_train.csv'), index=False)
            df_mia_tr.to_csv(os.path.join(save_pth, 'mia_train.csv'), index=False)
            df_mia_te.to_csv(os.path.join(save_pth, 'mia_test.csv'), index=False)
    return df_train, df_test, df_gen, df_mia_tr, df_mia_te

def scale_col(train_x, test_x, sc_col):
    """
    Scales specified columns in the training and testing datasets using StandardScaler.

    Parameters:
    train_x (pd.DataFrame): The training dataset.
    test_x (pd.DataFrame): The testing dataset.
    sc_col (list): List of column names to be scaled.

    Returns:
    tuple: A tuple containing:
        - train_sc (pd.DataFrame): The training dataset with specified columns scaled.
        - test_sc (pd.DataFrame): The testing dataset with specified columns scaled.
        - sc (StandardScaler): The fitted StandardScaler object.
    """
    sc= StandardScaler()
    train_sc= train_x.drop(columns=[col for col in train_x.columns if col not in sc_col])
    test_sc= test_x.drop(columns=[col for col in test_x.columns if col not in sc_col])
    train_sc[sc_col]= sc.fit_transform(train_x[sc_col])
    test_sc[sc_col]= sc.transform(test_x[sc_col])
    return train_sc, test_sc, sc

def ohe_transform(train_x, test_x, ohe_col):
    """
    Perform one-hot encoding on specified columns of training and testing datasets.

    Parameters:
    train_x (pd.DataFrame): The training dataset.
    test_x (pd.DataFrame): The testing dataset.
    ohe_col (list): List of column names to be one-hot encoded.

    Returns:
    tuple: A tuple containing:
        - pd.DataFrame: One-hot encoded training dataset.
        - pd.DataFrame: One-hot encoded testing dataset.
        - OneHotEncoder: The fitted OneHotEncoder instance.
    """
    ohe= OneHotEncoder(sparse=False, handle_unknown='ignore')
    ohe= ohe.fit(pd.concat([train_x, test_x], axis=0)[ohe_col])
    train_ohe= ohe.transform(train_x[ohe_col])
    test_ohe= ohe.transform(test_x[ohe_col])
    return pd.DataFrame(train_ohe, columns= ohe.get_feature_names()), pd.DataFrame(test_ohe, columns= ohe.get_feature_names()), ohe


def ml_preprocess(train_x, test_x, sc_col, ohe_col, return_transformers=False):
    """
    Preprocesses the training and testing data by scaling and one-hot encoding specified columns.

    Parameters:
    train_x (pd.DataFrame): The training data.
    test_x (pd.DataFrame): The testing data.
    sc_col (list): List of columns to be scaled.
    ohe_col (list): List of columns to be one-hot encoded.
    return_transformers (bool, optional): If True, returns the transformers used for scaling and encoding. Defaults to False.

    Returns:
    tuple: 
        - train_x (pd.DataFrame): The preprocessed training data.
        - test_x (pd.DataFrame): The preprocessed testing data.
        - sc (optional): The scaler used for scaling (if return_transformers is True).
        - ohe (optional): The one-hot encoder used for encoding (if return_transformers is True).
    """
    train_sc, test_sc, sc= scale_col(train_x, test_x, sc_col)
    train_ohe, test_ohe, ohe= ohe_transform(train_x, test_x, ohe_col)
    train_x= pd.concat([train_sc.reset_index(drop=True), train_ohe.reset_index(drop=True)], axis=1)
    test_x= pd.concat([test_sc.reset_index(drop=True), test_ohe.reset_index(drop=True)], axis=1)
    if return_transformers:
        return train_x, test_x, sc, ohe
    else:
        return train_x, test_x

def ml_prepare(df, y_col, num_col,cat_col, ohe, sc):
    """
    Prepares the data for machine learning when we already have fitted scaler and ohe object.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the data.
    y_col (str): The name of the target column.
    num_col (list of str): List of numerical column names to be scaled.
    cat_col (list of str): List of categorical column names to be one-hot encoded.
    ohe (OneHotEncoder): The fitted OneHotEncoder instance for transforming categorical features.
    sc (StandardScaler): The fitted StandardScaler instance for scaling numerical features.

    Returns:
    pd.DataFrame: A dataframe with processed features and the target column.
    """
    # TBD merge with ml_process
    df_x= df.drop(columns=[y_col])
    df_y= df[y_col]
    df_x_ohe= pd.DataFrame(ohe.transform(df_x[cat_col]), columns= ohe.get_feature_names())
    df_x_sc= df_x[num_col]
    df_x_sc[num_col]= sc.transform(df_x_sc)
    df_concat= pd.concat([df_x_sc.reset_index(drop=True), df_x_ohe.reset_index(drop=True)], axis=1)
    return pd.concat([df_concat.reset_index(drop=True), df_y.reset_index(drop=True)], axis=1)


class TabDataset(Dataset):
    """
    A wrapper dataset class for tabular data.
    Args:
        X (pd.DataFrame): The input features as a pandas DataFrame.
        y (pd.Series): The target values as a pandas Series.
    """
    def __init__(self, X, y):
        super().__init__()
        self.X= X.values
        self.y= y.values
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return torch.Tensor(self.X[idx]), torch.Tensor([self.y[idx]])
    

def main():
    params= parse_args()
    random.seed(params.random_seed)
    df_train, df_test, df_gen, df_mia_tr, df_mia_te= load_data(params.data_pth, params.save_data_folder,params.continuous_col, params.mia_test_size, params.mia_train_size, params.gen_train_size)
    helper_synth= pd.read_csv(os.path.join(params.save_data_folder, 'synth.csv'))
    train_x, train_y= df_train.drop(columns=[params.target]), df_train[params.target]
    test_x, test_y= df_test.drop(columns=[params.target]), df_test[params.target]
    num_col=[col for col in train_x.columns if train_x[col].dtypes !='object']
    cat_col= [col for col in train_x.columns if col not in num_col]
    train_x, test_x, sc, ohe= ml_preprocess(train_x, test_x, num_col, cat_col, return_transformers=True)
    joblib.dump((sc, ohe), os.path.join(params.save_data_folder,'transforms', 'sc_ohe_transforms.pkl'))
    train_dataset= TabDataset(train_x, train_y)
    test_dataset= TabDataset(test_x, test_y)
    helper_synth= ml_prepare(helper_synth, params.target, num_col, cat_col, ohe, sc)
    helper_dataset= TabDataset(helper_synth.drop(columns=[params.target]), helper_synth[params.target])
    train_dataloader= DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataloader= DataLoader(test_dataset, batch_size=32, shuffle=False)
    helper_dataloader= DataLoader(helper_dataset, batch_size=32, shuffle=True)

    nb_var=train_x.shape[1]
    #train f and h
    criterion = nn.BCELoss()
    f= DNN(nb_var)
    optim_f= torch.optim.Adam(f.parameters(), lr= params.lr)
    train_network(f, 100, train_dataloader, valid_dataloader, optim_f, criterion, pth=f'models/f_{params.name}.pth')
    torch.save(f, f'models/f_overfit_{params.name}.pth')

    h= DNN(nb_var)
    optim_h= torch.optim.Adam(h.parameters(), lr= params.lr)
    train_network(h, 30, helper_dataloader, valid_dataloader, optim_h, criterion, pth=f'models/helper_{params.name}.pth')

if __name__ == '__main__':
    main()
