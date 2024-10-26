import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, plot_precision_recall_curve, f1_score, roc_auc_score, fbeta_score, make_scorer, precision_score, auc
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
make_scorer
import warnings
import pickle
from O1_Steinke_Code.o1_audit import *

#Ajouter sample weight

class LogReg():
    def __init__(self, train_x, train_y, test_x):
        self.lr= LogisticRegression(random_state=42)
        self.train_x= train_x
        self.train_y= train_y
        self.test_x= test_x
    
    def train_predict(self):
        self.lr.fit(self.train_x, self.train_y)
        return self.lr.predict_proba(self.test_x)[:,1]

def reg_log_train_predict(train_x, train_y, test_x):
    lr= LogisticRegression(random_state=42)
    lr.fit(train_x, train_y)
    return lr.predict_proba(test_x)[:,1], lr

def max_precision(y_true, y_pred):
    for th in reversed(np.arrange(np.min(y_pred), np.max(y_pred)+0.01, 0.01)):
        preds= np.where(y_pred>th,1,0)
        return precision_score(y_true, y_pred)

def auc_pr_score(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

def privacy_recall(preds, labels):
    thresholds = np.linspace(0, 1, 100)
    recalls = []
    eps_lbs = []
    for th in thresholds:
        hard_preds = (preds > th).astype('float')
        tn, fp, fn, tp = confusion_matrix(labels, hard_preds).ravel()
        #if tp+fp == 0:
        #    continue
        eps_lb = get_eps_audit(len(preds), fp+tp, tp, 0., 0.05/(2))
        recall = (tp) / (tp+fn)
        recalls.append(recall)
        eps_lbs.append(eps_lb)
    
    recalls = np.array(recalls)
    eps_lbs = np.array(eps_lbs)
    eps_max= np.max(eps_lbs)
    #print(f"Privacy value score: {eps_max}")
    return thresholds, recalls, eps_lbs, eps_max

def privacy_value_score(y_true, y_pred):
    _,__,___, max_val= privacy_recall(y_pred, y_true)
    return max_val

auc_pr_scorer= make_scorer(auc_pr_score)
privacy_value_scorer= make_scorer(privacy_value_score)

class DecisionTree():
    def __init__(self, train_x, train_y, test_x):
        self.train_x= train_x
        self.train_y= train_y
        self.test_x= test_x
        self.init_hyp_param()
    
    def init_hyp_param(self):
        self.hyp_param= {
            'max_depth': hp.choice('max_depth', range(1,7)),
            #"criterion": hp.choice("criterion", ["gini", "entropy"])
            'class_weight': hp.choice('class_weight', [None, 'balanced'])

        }
    
    def train_test(self, params):
        tree= DecisionTreeClassifier(**params, random_state=42)
        return {'loss':-cross_val_score(tree, self.train_x, self.train_y, cv= 3, scoring='f1', n_jobs=-1).mean(), 'status': STATUS_OK}
    
    def find_best_params(self):
        trials = Trials()
        print('starting hyperopt')
        self.best_hp = fmin(self.train_test, self.hyp_param, algo=tpe.suggest, max_evals=50, trials=trials, return_argmin=False)
        print('best hyper-parameters found:', self.best_hp)
    
    def train_predict(self):
        self.find_best_params()
        print('final training')
        self.best_tree= DecisionTreeClassifier(**self.best_hp, random_state=42)
        self.best_tree.fit(self.train_x, self.train_y)
        return self.best_tree.predict_proba(self.test_x)[:,1]

class Tree1():
    def __init__(self, train_x, train_y, test_x, model_path):
        self.train_x= train_x
        self.train_y= train_y
        self.test_x= test_x
        self.model_path= model_path
        
    def train_predict(self):
        print('final training')
        self.best_tree= DecisionTreeClassifier(max_depth=1, random_state=42)
        self.best_tree.fit(self.train_x, self.train_y)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.best_tree, f)
        return self.best_tree.predict_proba(self.test_x)[:,1]
    
    def predict(self):
        # Load the best model
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        return model.predict_proba(self.test_x)[:, 1]

class RandomForest():
    def __init__(self, train_x, train_y, test_x, max_evals=20):
        self.train_x= train_x
        self.train_y= train_y
        self.test_x= test_x
        self.init_hyp_param()
        self.max_evals= max_evals
    
    def init_hyp_param(self):
        self.hyp_param= {
            'max_depth': hp.choice('max_depth', range(1,7)),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', 4,5,7]),
            #'n_estimators': hp.choice('n_estimators', range(1,200)),
            'n_estimators': hp.choice('n_estimators', [10,20,30,40,50,75,100,125,150,175,200]),
            'class_weight': hp.choice('class_weight', [None, 'balanced'])
        }
    
    def train_test(self, params):
        tree= RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        return {'loss':-cross_val_score(tree, self.train_x, self.train_y, cv= 3, scoring='f1', n_jobs=-1).mean(), 'status': STATUS_OK}
    
    def find_best_params(self):
        trials = Trials()
        print('starting hyperopt')
        self.best_hp = fmin(self.train_test, self.hyp_param, algo=tpe.suggest, max_evals=self.max_evals, trials=trials, return_argmin=False)
        print('best hyper-parameters found:', self.best_hp)
    
    def train_predict(self):
        self.find_best_params()
        print('final training')
        self.best_rf= RandomForestClassifier(**self.best_hp, random_state=42)
        self.best_rf.fit(self.train_x, self.train_y)
        return self.best_rf.predict_proba(self.test_x)[:,1]

class GradientBoosting(): #use light GBM with balanced option instead
    def __init__(self, train_x, train_y, test_x, acc=True, model_path=None):
        self.train_x= train_x
        self.train_y= train_y.squeeze()
        self.test_x= test_x
        self.model_path= model_path
        self.init_hyp_param()
        self.acc=acc
    
    def init_hyp_param(self):
        self.hyp_param= {
            'max_depth': hp.choice('max_depth', range(1,6)),
            #'max_features': hp.choice('max_features', ['sqrt', 'log2', 4,5,7]),
            'num_leaves': hp.choice('num_leaves', range(20, 100, 10)),
            'n_estimators': hp.choice('n_estimators', [10,20,30,40,50,75,100,125,150,175,200]),
            'learning_rate': hp.choice('learning_rate', [0.3,0.1, 0.05,0.01, 0.005, 0.001, 0.0005, 0.0001]),
            #'feature_fraction': hp.uniform('feature_fraction', 0.5, 1.0),
            #'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1.0),
            #'bagging_freq': hp.choice('bagging_freq', range(1, 10)),
            'class_weight': hp.choice('class_weight', [None, 'balanced'])
        }
    
    def train_test(self, params):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gb = LGBMClassifier(**params, random_state=42, verbose=-1)
            f05_scorer = make_scorer(fbeta_score, beta=0.5)
            if self.acc:
                return {'loss': -cross_val_score(gb, self.train_x, self.train_y, cv=3, scoring='accuracy', n_jobs=-1).mean(), 'status': STATUS_OK}
            else:
                return {'loss':-cross_val_score(gb, self.train_x, self.train_y, cv= 3, scoring=privacy_value_scorer, n_jobs=-1).mean(), 'status': STATUS_OK}
    
    def find_best_params(self, max_evals):
        trials = Trials()
        print('starting hyperopt')
        self.best_hp = fmin(self.train_test, self.hyp_param, algo=tpe.suggest, max_evals=max_evals, trials=trials, return_argmin=False)
        print('best hyper-parameters found:', self.best_hp)
    
    def train_predict(self, max_evals=100):
        self.find_best_params(max_evals)
        print('final training')
        self.best_gb= LGBMClassifier(**self.best_hp, random_state=42, importance_type='gain')
        self.best_gb.fit(self.train_x, self.train_y)
        if self.model_path:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.best_gb, f)
        return self.best_gb.predict_proba(self.test_x)[:,1]
    
    def predict(self):
        # Load the best model
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        return model.predict_proba(self.test_x)[:, 1]

