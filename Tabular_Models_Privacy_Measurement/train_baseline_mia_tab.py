import sys
import os
# Add the root folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from models.classifier import *
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from models.classifier import *
import argparse
import os
from train_target_tab import ml_prepare
import joblib
from O1_Steinke_Code import o1_audit
from scipy.stats import bernoulli

def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--target', type=str, default='over_50k', help='Target variable')
    parser.add_argument('--save_data_folder', type=str, default='./data', help='Path to save datasets')
    parser.add_argument('--random_seed', type=str, default=42, help='random seed for consistancy')
    parser.add_argument('--nb_run', type=int, default=1, help='Number of time to run experiment for CI')
    parser.add_argument('--name', type=str, default='default', help='Name of the experiment')
    parser.add_argument('--audit_test_size', type=int, default=22000, help='Size of audit test set')
    return parser.parse_args()


def load_data(path):
    df_mia_tr= pd.read_csv(os.path.join(path, 'mia_train.csv'))
    df_mia_te= pd.read_csv(os.path.join(path, 'mia_test.csv'))
    df_gen= pd.read_csv(os.path.join(path, 'gen_train.csv'))
    df_synth_helper= pd.read_csv(os.path.join(path, 'synth_helper.csv'))
    df_synth= pd.read_csv(os.path.join(path, 'synth.csv'))
    return df_mia_tr, df_mia_te, df_gen, df_synth, df_synth_helper

def add_loss(df, net, prefix,to_drop=['over_50k']):
    if to_drop is not None:
        with torch.no_grad():
            pred=net(torch.Tensor(df.drop(columns=to_drop).values))
            truth= torch.Tensor(df['over_50k'].values).reshape(-1,1)
            loss_fn= nn.BCELoss(reduction='none')
            loss= loss_fn(pred, truth)
            #loss= nn.BCELoss(net(torch.Tensor(df.drop(columns=to_drop).values)), torch.Tensor(df['over_50k'].values))
        df[prefix+'_loss']= loss.cpu().numpy()
    return df

def remoove_obvious_synth(quali_x, quali_y,train_x, train_y, valid_x, valid_y):
    label= train_y.columns
    disc= GradientBoosting(quali_x, quali_y, valid_x)
    pred= disc.train_predict(30)
    valid = pd.concat([valid_x.reset_index(drop=True), valid_y.reset_index(drop=True), pd.DataFrame({'pred':pred})], axis=1)
    valid_keep= valid.loc[~((valid.member==0) & (valid.pred < .1)), valid.drop(columns=['pred']).columns]
    pred= disc.best_gb.predict_proba(train_x)[:,1]
    train = pd.concat([train_x.reset_index(drop=True), train_y.reset_index(drop=True), pd.DataFrame({'pred':pred})],axis=1)
    train_keep= train.loc[~((train.member==0)& (train.pred < .1)), valid.drop(columns=['pred']).columns]
    return train_keep.drop(columns=label), train_keep[label], valid_keep.drop(columns=label), valid_keep[label]

def balance_dataset(df_x, df_y):
    df = pd.concat([df_x, df_y], axis=1)
    real_df = df.loc[df.member == 0]
    synthetic_df = df.loc[df.member == 1]
    num_synthetic = len(synthetic_df)
    num_real = len(real_df)
    if num_real > num_synthetic:
        real_df = real_df.sample(n=num_synthetic, random_state=42)
    elif num_synthetic > num_real:
        synthetic_df = synthetic_df.sample(n=num_real, random_state=42)
    balanced_df = pd.concat([real_df, synthetic_df], axis=0).reset_index(drop=True)
    # Split the balanced dataframe back into X and y
    balanced_x = balanced_df.drop(columns=['member'])
    balanced_y = balanced_df[['member']]
    return balanced_x, balanced_y

def mia_baseline_discriminateur(train_x, train_y, test_x, test_y):
    #disc= LogReg(train_x, train_y, test_x)
    #disc= DecisionTree(train_x, train_y, test_x)
    #disc= RandomForest(train_x, train_y, test_x, 100)
    disc= GradientBoosting(train_x, train_y, test_x, acc=True)
    pred= disc.train_predict(max_evals=200)
    return pred

def main():
    args= parse_args()
    df_mia_tr, df_mia_te, df_gen, df_synth, df_synth2= load_data(args.save_data_folder)
    sc, ohe= joblib.load(os.path.join(args.save_data_folder,'transforms', 'sc_ohe_transforms.pkl'))
    num_col=[col for col in df_mia_tr.columns if df_mia_tr[col].dtypes !='object' and col != args.target]
    cat_col= [col for col in df_mia_tr.columns if col not in num_col+[args.target]]
    print('num_col', num_col)
    print('cat_col', cat_col)

    # prepare data for panoramia
    df_mia_tr= ml_prepare(df_mia_tr, 'over_50k', num_col, cat_col, ohe, sc)
    df_mia_te= ml_prepare(df_mia_te, 'over_50k', num_col, cat_col, ohe, sc)
    synth= ml_prepare(df_synth,'over_50k', num_col, cat_col, ohe, sc)
    synth2= ml_prepare(df_synth2,'over_50k', num_col, cat_col, ohe, sc)
    df_gen= ml_prepare(df_gen, 'over_50k', num_col, cat_col, ohe, sc)

    df_synth['over_50k']=df_synth['over_50k'].astype('int64')
    df_synth2['over_50k']=df_synth2['over_50k'].astype('int64')

    synth_tr= synth.iloc[:len(df_mia_tr),:]
    synth_te= synth.iloc[len(df_mia_tr):len(df_mia_tr)+len(df_mia_te),:]

    real_synth_x= pd.concat([df_mia_te.reset_index(drop=True), synth_te.reset_index(drop=True)], axis=0)
    real_synth_y= pd.DataFrame({'member':[1 for i in range(len(df_mia_te))]+[0 for i in range(len(synth_te))]})

    train_x= pd.concat([df_mia_tr.reset_index(drop=True), synth_tr.reset_index(drop=True)], axis=0)
    train_y= pd.DataFrame({'member':[1 for i in range(len(df_mia_tr))]+[0 for i in range(len(synth_tr))]})#

    df_real_syth_quali_x= pd.concat([synth2.reset_index(drop=True), df_gen.reset_index(drop=True)], axis=0).reset_index(drop=True)
    df_real_syth_quali_y= pd.DataFrame({'member':[0 for i in range(len(synth2))]+[1 for i in range(len(df_gen))]})
    
    #train_x, train_y, real_synth_x, real_synth_y= remoove_obvious_synth(df_real_syth_quali_x,df_real_syth_quali_y,train_x, train_y, real_synth_x, real_synth_y)
    #train_x, train_y= balance_dataset(train_x.reset_index(drop=True), train_y.reset_index(drop=True))
    #real_synth_x, real_synth_y= balance_dataset(real_synth_x, real_synth_y)

    f= torch.load(f'models/f_{args.name}.pth')
    f_overfit= torch.load(f'models/f_overfit_{args.name}.pth')
    h= torch.load(f'models/helper_{args.name}.pth')


    df_mia_full_x= pd.concat([train_x, real_synth_x], axis=0)
    df_mia_full_y= pd.concat([train_y, real_synth_y], axis=0)
    df_mia= pd.concat([df_mia_full_x.reset_index(drop=True), df_mia_full_y.reset_index(drop=True)], axis=1)

    # ADD loss from helper and target model

    df_mia= add_loss(df_mia, f, 'f', ['member', 'over_50k'])
    df_mia= add_loss(df_mia, f_overfit, 'f_overfit', ['f_loss','member', 'over_50k'])
    df_mia= add_loss(df_mia, h, 'h', ['f_loss','f_overfit_loss','member', 'over_50k'])

    print(df_mia.shape)

    for i in range(args.nb_run):
        train, test= train_test_split(df_mia, test_size= args.audit_test_size, random_state=args.random_seed+i)
        print(f'run {i}')
        coin_flip = bernoulli.rvs(.5, size=len(test), random_state=args.random_seed+i)
        this_test= test[np.array(coin_flip).astype(bool)]
        print('MIA train size:', len(train))
        print('MIA test size:', len(this_test))
        baseline_pred= mia_baseline_discriminateur(train.drop(columns=['member', 'f_loss', 'f_overfit_loss']), train['member'], this_test.drop(columns=['member', 'f_loss', 'f_overfit_loss']), this_test['member'])
        f_pred= mia_baseline_discriminateur(train.drop(columns=['member', 'h_loss', 'f_overfit_loss']), train['member'], this_test.drop(columns=['member', 'h_loss', 'f_overfit_loss']), this_test['member'])
        f_overfit_pred= mia_baseline_discriminateur(train.drop(columns=['member', 'f_loss', 'h_loss']), train['member'], this_test.drop(columns=['member', 'f_loss', 'h_loss']), this_test['member'])
        df= pd.DataFrame({'member': this_test['member'],'baseline_pred':baseline_pred, 'mia_mlp10_pred':f_pred, 'mia_mlp100_pred':f_overfit_pred})
        df.to_csv(os.path.join(args.save_data_folder, 'results', f'mia_baseline_{args.name}_{i}.csv'), index=False)

if __name__ == '__main__':
    main()