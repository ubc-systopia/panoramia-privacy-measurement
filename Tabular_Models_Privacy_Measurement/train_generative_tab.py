import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from reprosyn.methods import MST
import json
import reprosyn
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--continuous_col', nargs='+', default=['age', 'capital_gain', 'capital_loss'], help='List of continuous columns')
    parser.add_argument('--target', type=str, default='over_50k', help='Target variable')
    parser.add_argument('--data_pth', type=str, default='/home/hadrien/data/adult_uci/adult_clean.csv', help='Path to the  dataset')
    parser.add_argument('--domain_pth', type=str, default='/home/hadrien/data/adult/domain_adult.json', help='Path to the  dataset')
    parser.add_argument('--save_data_folder', type=str, default='./data', help='Path to save datasets')
    parser.add_argument('--random_seed', type=str, default=42, help='random seed for consistancy')
    parser.add_argument('--mia_train_size', type=int, default=13024, help='Size of MIA training set')
    parser.add_argument('--mia_test_size', type=int, default=13024, help='Size of MIA test set')
    parser.add_argument('--gen_train_size', type=int, default=13024, help='Size of GAN training set')
    return parser.parse_args()

# data can be download here: curl https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -o adult.csv

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

def generate_synth(df, domain_pth):
    # from reprosyn
    with open(domain_pth, 'r') as js_file:#reposyn read json with json.loads instead of json.load. this  workaround fix the metadata loading issues
            domaine= json.load(js_file)


    gen = MST(dataset=df.copy(), size=df.shape[0] *2, epsilon = 10000, metadata= domaine['columns'])
    gen.run()
    synth= gen.output
    synth= synth[df.columns]
    return synth


def main():
    args= parse_args()
    df_train, df_test, df_gen, df_mia_tr, df_mia_te= load_data(args.data_pth, args.save_data_folder, args.continuous_col, args.mia_test_size, args.mia_train_size, args.gen_train_size)
    synth= generate_synth(df_gen, args.domain_pth)
    synth_helper= generate_synth(df_gen, args.domain_pth)
    synth.to_csv(os.path.join(args.save_data_folder, 'synth.csv'), index=False)
    synth_helper.to_csv(os.path.join(args.save_data_folder, 'synth_helper.csv'), index=False)

if __name__ == '__main__':
    main()