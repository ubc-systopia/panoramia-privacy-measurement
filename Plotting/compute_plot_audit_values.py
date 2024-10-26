import sys
import os
# Add the root folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from sklearn.metrics import confusion_matrix
import math
import argparse
import matplotlib.pyplot as plt
import scipy.stats
from O1_Steinke_Code.o1_audit import get_eps_audit
import numpy as np
import pandas as pd
plt.rcParams['text.usetex'] = True
from matplotlib.lines import Line2D

def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--results_data_folder', type=str, default='./data/results', help='Path to results folder')
    return parser.parse_args()

def privacy_recall(preds, labels):
    thresholds = np.linspace(0, 1, 100)
    recalls = []
    eps_lbs = []
    for th in thresholds:
        hard_preds = (preds > th).astype('float')
        tn, fp, fn, tp = confusion_matrix(labels, hard_preds).ravel()
        #if tp+fp == 0:
        #    continue
        eps_lb = get_eps_audit(len(preds), fp+tp, tp, 0.05/(2*100), 0.)
        recall = (tp) / (tp+fn)
        recalls.append(recall)
        eps_lbs.append(eps_lb)
    
    recalls = np.array(recalls)
    eps_lbs = np.array(eps_lbs)
    eps_max= np.max(eps_lbs)

    return thresholds, recalls, eps_lbs, eps_max

def privacy_do_plot(recall, privacy, max_privacy, **plot_kwargs):
    plt.plot(recall, privacy , plot_kwargs['color'], linewidth=plot_kwargs['linewidth'], label=plot_kwargs['legend'])
    plt.axhline(y=max_privacy, color=plot_kwargs['color'], linestyle='--')


def preision_recall_extract(preds, labels, eps_max):
    thresholds = np.linspace(0, 1, 1000) 
    recall_curve = []
    precision_curve = []
    theoretical_precision_curve = []
    for th in thresholds:
        hard_preds = (preds > th).astype('float')
        tn, fp, fn, tp = confusion_matrix(labels, hard_preds).ravel()
        if tp+fp == 0:
            continue
        recall = (tp) / (tp+fn)
        precision = (tp) / (tp+fp)
        recall_curve.append(recall)
        precision_curve.append(precision)
        r = tp + fp
        q = 1/(1+math.exp(-eps_max))
        theoretical_precision_curve.append(scipy.stats.binom.isf(0.05/2, r, q) / r)        
        #precision, recall, thresholds = precision_recall_curve(y_true=labels, probas_pred=preds)

    recall_curve, precision_curve, theoretical_precision_curve = np.array(recall_curve), np.array(precision_curve), np.array(theoretical_precision_curve)    
    return recall_curve, precision_curve, theoretical_precision_curve

def precision_do_plot(recall_mean, precision_mean, theoretical_precision_max, **plot_kwargs):
    confidence = 1.96
    plt.plot(recall_mean, precision_mean , plot_kwargs['color'], linewidth=plot_kwargs['linewidth'], label=plot_kwargs['legend'])
    plt.plot(recall_mean, theoretical_precision_max, plot_kwargs['color'], linewidth=plot_kwargs['linewidth'], linestyle='--')

def nb_pred_recall(preds, labels):
    thresholds = np.linspace(0, 1, 1000) 
    recall_curve = []
    nb_pred_curve = []
    tps=[]
    for th in thresholds:
        hard_preds = (preds > th).astype('float')
        tn, fp, fn, tp = confusion_matrix(labels, hard_preds).ravel()
        if tp+fp == 0:
            continue
        recall = (tp) / (tp+fn)
        nb_pred =  (tp+fp)
        recall_curve.append(recall)
        nb_pred_curve.append(nb_pred)
        tps.append(tp)
    return np.array(recall_curve), np.array(nb_pred_curve), np.array(tps)

def nb_pred_do_plot(recall, nb_pred, tps, **plot_kwargs):
    plt.plot(recall, nb_pred,plot_kwargs['color'], linewidth=plot_kwargs['linewidth'], label=plot_kwargs['legend'])
    if tps is not None:
        plt.plot(recall, tps,color='black', linewidth=plot_kwargs['linewidth'], label='Number of true positive',linestyle='--')


def main():
    args= parse_args()
    mia_baseline_files = [f for f in os.listdir(args.results_data_folder) if 'mia_baseline' in f]
    #for f in mia_baseline_files:
    df_results = pd.read_csv(os.path.join(args.results_data_folder, mia_baseline_files[0])) # for now only plot results for one file, to be change to include CI
    models= [col for col in df_results.columns if 'pred' in col]
    privacy_values={}
    for model_col in models:
        print(model_col)
        th, recall, eps, eps_max= privacy_recall(df_results[model_col],df_results['member'])
        privacy_values[model_col]= [recall, eps, eps_max]
    
    c_lb= privacy_values['baseline_pred'][2]
    print('c_lb:', c_lb)
    for k in models:
        if k != 'baseline_pred':
            print(f'\n eps tilde {k}:', c_lb- privacy_values[k][2], '\n')

    colors= ['r', 'm', 'p']
    i=0
    plt.figure(figsize=(10,8))
    for model in models:
        if model=='baseline_pred':
            privacy_do_plot(privacy_values[model][0], privacy_values[model][1], privacy_values[model][2], color='b',legend=r'$c_\textrm{lb}$',linewidth=2.5)
        else:
            privacy_do_plot(privacy_values[model][0], privacy_values[model][1], privacy_values[model][2], color=colors[i],legend=r"$\{{c+\epsilon}\}_{\textrm{lb}}$"+model.replace('mia', 'PANORAMIA').replace('_', ' ').replace['pred', ''],linewidth=2.5)
            i+=1
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tick_params(axis='both', which='both', length=5)
    plt.xlabel('Recall', fontsize=28)
    plt.ylabel(r'Measurement for $c_\textrm{lb}$ or $\{{c+\epsilon}\}_{\textrm{lb}}$', fontsize=28)
    plt.tight_layout()
    # Get handles and labels from the existing legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # Add custom legend entry
    custom_line = Line2D([0], [0], color='black', linestyle='--', label='empirical maximum value')
    handles.append(custom_line)
    labels.append(custom_line.get_label())
    # Create a single legend with all entries
    plt.legend(handles=handles, labels=labels, fontsize=28, loc='lower right')
    plt.savefig(os.path.join(args.results_data_folder,'privacy_recall.png'), format='pdf')
    plt.show()

    precision_values={}
    for model_col in models:
        recall, precision, th_precision= preision_recall_extract(df_results[model_col],df_results['member'],privacy_values[model_col][2])
        precision_values[model_col]= [recall, precision, th_precision]
    i=0
    plt.figure(figsize=(10,8))
    for model in models:
        if model=='baseline_pred':
            precision_do_plot(precision_values[model][0], precision_values[model][1], precision_values[model][2], color='b',egend=f"baseline model",alpha=0.09,linewidth=2.5)
        else:
            precision_do_plot(precision_values[model][0], precision_values[model][1], precision_values[model][2], color=colors[i],legend=model.replace('mia', 'PANORAMIA').replace('_', ' ').replace['pred', ''],linewidth=2.5, alpha=0.09)
            i+=1
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tick_params(axis='both', which='both', length=5)
    plt.xlabel('Recall', fontsize=28)
    plt.ylabel('Precision', fontsize=28)
    plt.tight_layout()
    # Get handles and labels from the existing legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # Add custom legend entry
    custom_line = Line2D([0], [0], color='black', linestyle='--', label='Theoretical maximum precision')
    handles.append(custom_line)
    labels.append(custom_line.get_label())
    # Create a single legend with all entries
    plt.legend(handles=handles, labels=labels, fontsize=28, loc='lower left')
    plt.savefig(os.path.join(args.results_data_folder,'precision_recall.png'), format='pdf')
    plt.show()

if __name__ == '__main__':
    main()