# Loss module for the image privacy measurement for panorama
# This module gets predictions on member and non-member samples based on loss values of the given model (target or helper model) and trains an MLP 
# to distinguish between these images based on these loss values.
# Loss module for Membership Inference Attack (MIA) and Baseline classifier
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch import nn
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc, roc_curve
from skimage.util import random_noise
import random
import scipy.stats
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from O1_Steinke_Code.o1_audit import *

# Free up GPU memory
torch.cuda.empty_cache()

# Command-line arguments parser
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Seed for data sampling")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloaders")
parser.add_argument("--n_cpu", type=int, default=4, help="Number of CPU threads during batch generation")
parser.add_argument("--img_size", type=int, default=32, help="Size of each image dimension")
opt = parser.parse_args()

# Debugging to show command-line arguments
print(opt)


def find_O1_pred(member_loss_values, non_member_loss_values, delta = 0.):
    """
    Args:
        member_loss_values: NumPy array containing member loss values
        non_member_loss_values: NumPy array containing non_member loss values
    Returns:
     best_eps: largest audit (epsilon) value that can be returned for a particular p-value
    """
    
    # Create labels for real and generated loss values
    member_labels = np.ones_like(member_loss_values)
    non_member_labels = np.zeros_like(non_member_loss_values)

    # Concatenate loss values and labels
    if all_losses.size > 0:
        all_losses = np.concatenate((member_loss_values, non_member_loss_values))
        all_labels = np.concatenate((member_labels, non_member_labels))
    # print(f"Shape of all_losses: {all_losses.shape}")
    

    # Step 1: Find t_pos that maximizes precision for positive predictions
    best_precision = 0
    best_t_pos = 0
    threshold_range = np.arange(np.min(all_losses), np.max(all_losses) + 0.01, 0.01)
    results, recall = [], []
    best_accuracy = 0
    best_t_neg = 0
    total_predictions = 0
    correct_predictions = 0
    best_eps = 0
    p = 0.05
    for t_pos in threshold_range:
        positive_predictions = all_losses[all_losses <= t_pos]
        if len(positive_predictions) == 0:
            continue

        true_positives = np.sum(all_labels[all_losses <= t_pos] == 1)
        
        eps = get_eps_audit(len(all_labels), len(positive_predictions), true_positives, p, delta)
        precision = true_positives / len(positive_predictions)
        if eps > best_eps:
            print("EPSILON UPDATE:", eps)
            best_eps = eps
            best_t_pos = t_pos
        recalls = true_positives / np.sum(all_labels == 1)
        recall.append(recalls)
        
        # Step 2: With t_pos fixed, find t_neg that maximizes overall accuracy
        for t_neg in reversed(threshold_range):
            if t_neg <= best_t_pos:
                break
            confident_predictions = all_losses[(all_losses <= best_t_pos) | (all_losses >= t_neg)]
            r = len(confident_predictions)
            mask_pos = (confident_predictions <= best_t_pos) & (all_labels[(all_losses <= best_t_pos) | (all_losses >= t_neg)] == 1)
            mask_neg = (confident_predictions >= t_neg) & (all_labels[(all_losses <= best_t_pos) | (all_losses >= t_neg)] == 0)
    
            v = np.sum(np.logical_or(mask_pos, mask_neg))
    
            if r > 0:
                accuracy = v / r
                eps = get_eps_audit(len(all_labels), r, v, p, delta)
                if eps > best_eps:
                    best_eps = eps
                    best_t_neg = t_neg
                    total_predictions = r
                    correct_predictions = v
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
        
    return best_eps
    

def eps_lb_metric(member_loss_values, non_member_loss_values):
    """
    Custom scoring metric for calculating epsilon lower bound in the audit.

    Args:
        member_loss_values: Loss values for members (real data).
        non_member_loss_values: Loss values for non-members (generated data).

    Returns:
        float: Largest epsilon value found using find_O1_pred function.
    """
    # Call the find_O1_pred function with the provided loss values
    best_eps = find_O1_pred(member_loss_values, non_member_loss_values)
    return best_eps


# Setting up model and optimizer parameters for the classifier
param_grid = {
    'hidden_layer_sizes': [10],
    'alpha': [0.01],
    'learning_rate_init': [0.01],
    'max_iter': [5]
}

# Setup device for PyTorch
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
loss_fn = nn.CrossEntropyLoss()

def weights_init_normal(m):
    """
    Custom weight initialization for Conv and BatchNorm layers.
    
    Args:
        m: The layer to initialize.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def logit_trainer(dataloader_in_train, net, logit_classifier, dataloader_in_gen_train, noise=False):
    """
    Train a logistic classifier on real (member) and fake (non-member) data using
    the epsilon lower bound metric as the scoring function.
    
    Args:
        dataloader_in_train: Dataloader for real member data (train set).
        net: Neural network used for feature extraction/loss values.
        logit_classifier : classifier to train on losses of the net to distinguish member vs non-member
        dataloader_in_gen_train: Dataloader for fake non-member data (train set).
        noise (bool): Whether to add Gaussian noise to data (default is False).

    Returns:
        best_model: Best model after training is complete and optimized.
        results: prediction returned by best_model.
    """
    real_loss_values, gen_loss_values = [], []

    # Iterator for generated fake data
    dataloader_gen_iterator_train = iter(dataloader_in_gen_train)

    for _, (real_images, real_clabels) in enumerate(dataloader_in_train):
        try:
            gen_imgs, gen_clabels = next(dataloader_gen_iterator_train)
        except StopIteration:
            dataloader_gen_iterator_train = iter(dataloader_in_gen_train)
            gen_imgs, gen_clabels = next(dataloader_gen_iterator_train)

        if noise:
            real_images = torch.tensor(random_noise(real_images, mode='gaussian', mean=0, var=0.0001, clip=False))
            gen_imgs = torch.tensor(random_noise(gen_imgs.detach().cpu().numpy(), mode='gaussian', mean=0, var=0.0001, clip=False))

        # Process real images and calculate losses
        real_imgs = real_images.to(device)
        real_clabels = real_clabels.to(device)
        for i in range(real_imgs.size(0)):
            image = real_imgs[i].unsqueeze(0)
            label = real_clabels[i].unsqueeze(0)

            if image.ndimension() == 3:
                image = image.unsqueeze(0)
            if image.shape[1] == 1:
                image = image.repeat(1, 3, 1, 1)

            with torch.no_grad():
                predictions = net(image)
                loss = loss_fn(predictions, label)
                real_loss_values.append(loss.item())  # Append scalar loss value

        # Process generated images and calculate losses
        gen_imgs = gen_imgs.to(device)
        gen_clabels = torch.tensor(gen_clabels, dtype=torch.long).to(device)
        for i in range(gen_imgs.size(0)):
            image = gen_imgs[i].unsqueeze(0)
            label = gen_clabels[i].unsqueeze(0)

            if image.ndimension() == 3:
                image = image.unsqueeze(0)
            if image.shape[1] == 1:
                image = image.repeat(1, 3, 1, 1)

            with torch.no_grad():
                predictions = net(image)
                loss = loss_fn(predictions, label)
                gen_loss_values.append(loss.item())  # Append scalar loss value

    # Convert loss values to NumPy arrays for classifier training
    real_loss_values = np.array(real_loss_values).reshape(-1, 1)  # Shape [num_real_images, 1]
    gen_loss_values = np.array(gen_loss_values).reshape(-1, 1)  # Shape [num_gen_images, 1]

    # Concatenate loss values
    X = np.concatenate((real_loss_values, gen_loss_values))
    y = np.concatenate((np.ones(len(real_loss_values)), np.zeros(len(gen_loss_values))))

    # Grid search
    eps_lb_score = make_scorer(eps_lb_metric, greater_is_better=True, needs_proba=False)
    grid_search = GridSearchCV(logit_classifier, param_grid, scoring=eps_lb_score, refit=True, cv=2)
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    results = best_model.predict(X)

    return best_model, results



def logit_tester(dataloader_in_test, net, dataloader_in_gen_test, best_model, noise=False):
    """
    Test the trained logistic classifier on real (member) and fake (non-member) data.

    Args:
        dataloader_in_test: Dataloader for real member data (test set).
        net: Neural network used for feature extraction/loss values.
        dataloader_in_gen_test: Dataloader for fake non-member data (test set).
        best_model: Trained logistic classifier.
        noise (bool): Whether to add Gaussian noise to data (default is False).

    Returns:
        y_pred_fold: Model predictions for the test set.
        fprs: False positive rates for ROC curve.
        tprs: True positive rates for ROC curve.
        aucs: Area Under the Curve (AUC) scores for ROC curve.
        accuracy: Test accuracy.
        prec: Test precision.
        recall: Test recall.
    """
    results = []
    y_pred = []
    y_true = []
    fprs, tprs, aucs = [], [], []
    real_loss_values, gen_loss_values = [], []
    dataloader_iterator_test = iter(dataloader_in_gen_test)

    for _, (real_images, real_clabels) in enumerate(dataloader_in_test):
        try:
            gen_imgs, gen_clabels = next(dataloader_iterator_test)
        except StopIteration:
            dataloader_iterator_test = iter(dataloader_in_gen_test)
            gen_imgs, gen_clabels = next(dataloader_iterator_test)

        if noise:
            # Add Gaussian noise to both real and fake images
            real_images = torch.tensor(random_noise(real_images, mode='gaussian', mean=0, var=0.0001, clip=False))
            gen_imgs = torch.tensor(random_noise(gen_imgs.detach().cpu().numpy(), mode='gaussian', mean=0, var=0.0001, clip=False))

        # Convert images to appropriate tensor type for the device
        gen_imgs = gen_imgs.type(torch.FloatTensor).to(device)
        real_imgs = real_images.type(torch.FloatTensor).to(device)
        

        # Calculate loss for real member images
        for i in range(real_imgs.size(0)):
            image = real_imgs[i].unsqueeze(0)  # Add batch dimension
            label = real_clabels[i].unsqueeze(0).to(device)

            if image.ndimension() == 3:
                image = image.unsqueeze(0)  # Add batch dimension, becomes [1, 1, 32, 32] if grayscale

            # Check if the image has only 1 channel (grayscale) and needs to be converted to RGB
            if image.shape[1] == 1:
                image = image.repeat(1, 3, 1, 1)  # Convert to 3 channels (RGB), shape becomes [1, 3, 32, 32]

            with torch.no_grad():
                predictions = net(image)
                loss = loss_fn(predictions, label)
                real_loss_values.append(loss.item())  # Append scalar loss value
        
        # Calculate loss for fake non-member images
        gen_clabels = torch.tensor(gen_clabels, dtype=torch.long).to(device)
        for i in range(gen_imgs.size(0)):
            image = gen_imgs[i].unsqueeze(0)
            label = gen_clabels[i].unsqueeze(0).to(device)

            if image.ndimension() == 3:
                image = image.unsqueeze(0)  # Add batch dimension, becomes [1, 1, 32, 32] if grayscale

            # Check if the image has only 1 channel (grayscale) and needs to be converted to RGB
            if image.shape[1] == 1:
                image = image.repeat(1, 3, 1, 1)  # Convert to 3 channels (RGB), shape becomes [1, 3, 32, 32]

            with torch.no_grad():
                predictions = net(image)
                loss = loss_fn(predictions, label)
                gen_loss_values.append(loss.item())  # Append scalar loss value
        
    # Convert loss values to NumPy arrays for classifier testing
    real_loss_values = np.array(real_loss_values).reshape(-1, 1)
    gen_loss_values = np.array(gen_loss_values).reshape(-1, 1)

    X_test = np.concatenate((real_loss_values, gen_loss_values))
    y_test = np.concatenate((np.ones(len(real_loss_values)), np.zeros(len(gen_loss_values))))

    # Evaluate the model on the test set
    y_pred_fold = best_model.predict(X_test)
    y_pred.extend(y_pred_fold)
    y_true.extend(y_test)
    # print("y-pred", len(y_pred_fold))
    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred_fold)
    prec = precision_score(y_test, y_pred_fold)
    recall = recall_score(y_test, y_pred_fold)
    auc_roc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

    # ROC curve
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    fprs.append(fpr)
    tprs.append(tpr)
    aucs.append(roc_auc)

    results.append([accuracy, prec, recall, auc_roc])

    print("loss_module auc_roc:", auc_roc)

    # Return the evaluation results
    return y_pred_fold, fprs, tprs, aucs, accuracy, prec, recall
