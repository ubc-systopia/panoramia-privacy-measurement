# Image module for the image privacy measurement for panorama
# This module gets predictions on member and non-member samples based on raw image data 
# to distinguish between these images based on these images.
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
import torchvision
from skimage.util import random_noise
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from O1_Steinke_Code.o1_audit import *
from torch import FloatTensor
# Set CUDA if available
cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_O1_pred(predictions, labels, delta = 0.):
    """
    Args:
        predictions: NumPy array containing predictions from either MIA or baseline classifiers
        labels: NumPy array containing labels for member or non-member data
    Returns:
     best_eps: largest privacy audit (epsilon) value that can be returned for a particular p value
    """
    all_losses = predictions
    all_labels = labels
    
    if isinstance(all_losses, torch.Tensor):
        all_losses = all_losses.detach().cpu().numpy()
        
    elif isinstance(all_losses, list):
        # Convert list of tensors to NumPy array if needed
        all_losses = np.array([loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in all_losses])
    if isinstance(all_labels, torch.Tensor):
        all_labels = all_labels.detach().cpu().numpy()
    all_losses = np.array(all_losses)
    # print("all-losses", all_losses.shape)

    # If all_labels is a list of tensors, convert each to a NumPy array
    if isinstance(all_labels, list):
        all_labels = np.array([label.detach().cpu().numpy() if isinstance(label, torch.Tensor) else label for label in all_labels])
    elif isinstance(all_labels, torch.Tensor):
        # If all_labels is a tensor, convert it to a NumPy array
        all_labels = all_labels.detach().cpu().numpy()
    # print("entring o1 pred finder")
    # print("dims:", all_losses.ndim)
    # print("all-losses", all_losses.shape)
    if all_losses.ndim > 1:
        all_losses = all_losses.flatten()
    if all_labels.ndim > 1:
        all_labels = all_labels.flatten()
        
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
        mask = all_losses <= t_pos
        positive_predictions = all_losses[all_losses <= t_pos]
        if len(positive_predictions) == 0:
            continue

        true_positives = np.sum(all_labels[mask] == 1)
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
    
# Custom scoring metric for epsilon lower bound
def eps_lb_metric(y_true, y_pred):
    best_eps = find_O1_pred(y_pred, y_true)
    return best_eps

import itertools

def image_trainer(epochs, resnet, dataloader_in_train, dataloader_in_gen_train, criterion, optimizer, scheduler, noise=False, early_stopping_patience=5, dropout_rate=0.5):
    """
    Train the model on both real and generated images, using best eps_lb_metric for tuning, learning rate scheduling, dropout, and early stopping.
    
    Parameters:
    ----------
    epochs: int
        number of epochs to train the image module
    resnet : torch.nn.Module
        Model to be trained (should have dropout layers).
    dataloader_in_train : DataLoader
        DataLoader for real images.
    dataloader_in_gen_train : DataLoader
        DataLoader for generated (fake) images.
    criterion : loss function
        Loss function, e.g., CrossEntropyLoss.
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler for dynamic adjustment during training.
    noise : bool, optional
        Add Gaussian noise to images (default: False).
    early_stopping_patience : int
        Number of epochs to wait for improvement in epsilon metric before stopping early.
    dropout_rate : float
        The rate of dropout to be applied during training (typically between 0.1 and 0.5).

    Returns:
    --------
    all_outs : list of torch.Tensors
        The predicted outputs for all images.
    y_true : list of torch.Tensors
        The true labels corresponding to the predictions.
    """

    
    best_epsilon = -float('inf')
    epochs_no_improvement = 0  # To track epochs without improvement for early stopping
    best_model_state = None
    dataloader_iterator_train = iter(dataloader_in_gen_train)

    for epoch in range(epochs):
        y_true, all_outs = [], []
        resnet.train()  # Set the model to training mode
        for _, (real_images, _) in enumerate(dataloader_in_train):
            try:
                gen_imgs, _ = next(dataloader_iterator_train)
            except StopIteration:
                dataloader_iterator_train = iter(dataloader_in_gen_train)
                gen_imgs, _ = next(dataloader_iterator_train)

            if noise:
                # Add Gaussian noise to real and generated images
                real_images = torch.tensor(random_noise(real_images, mode='gaussian', mean=0, var=0.0001, clip=False))
                gen_imgs = torch.tensor(random_noise(gen_imgs.detach().cpu().numpy(), mode='gaussian', mean=0, var=0.0001, clip=False))

            real_images = real_images.type(torch.FloatTensor)
            gen_imgs = gen_imgs.type(torch.FloatTensor)

            if torch.cuda.is_available():
                real_images = real_images.cuda()
                gen_imgs = gen_imgs.cuda()

            # Concatenate real and generated images
            X = torch.cat((real_images, gen_imgs), dim=0)
            y = torch.cat((torch.ones(len(real_images)), torch.zeros(len(gen_imgs))), dim=0).long().cuda()

            optimizer.zero_grad()
            outputs = resnet(X)  # Forward pass

            # Calculate loss
            loss = criterion(outputs, y)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Append true labels
            y_true.extend(y.cpu().numpy())  # Extend ensures per-image labels are stored

            # Get per-image predictions
            _, predicted = torch.max(outputs.data, 1)  # Get class predictions
            all_outs.extend(predicted.cpu().numpy())  # Extend ensures per-image predictions are stored

        # Calculate the epsilon metric after each epoch
        current_epsilon = eps_lb_metric(all_outs, y_true)
        print(f"Epoch {epoch+1}/{epochs}, Current Epsilon Metric: {current_epsilon:.4f}")

        # Check if this is the best epsilon so far and save the model state if it is
        if current_epsilon > best_epsilon:
            best_epsilon = current_epsilon
            best_model_state = resnet.state_dict()  # Save the best model state
            print(f"New best epsilon: {best_epsilon:.4f} - saving model state.")
            epochs_no_improvement = 0  # Reset the counter for early stopping
        else:
            epochs_no_improvement += 1
            print(f"No improvement in epsilon metric. Patience count: {epochs_no_improvement}/{early_stopping_patience}")

        # Early stopping check
        if epochs_no_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

        # Update the learning rate scheduler
        scheduler.step()

    # Load the best model parameters before returning
    if best_model_state is not None:
        resnet.load_state_dict(best_model_state)
        print(f"Training completed. Best epsilon: {best_epsilon:.4f}")
    else:
        print(f"Training completed without improvement.")

    return all_outs, y_true

def image_tester(resnet, dataloader_in_test, dataloader_in_gen_test, criterion, noise=False):
    """
    Test the model on real and generated images, calculating accuracy and loss. 
    Return predictions, and  labels accuracy of the model on the real and generated image samples
    """
    # Initialize lists to store the true labels, predicted outputs, and other statistics
    y_true, all_outs, losses, av_accs = [], [], [], []

    # Create an iterator for the generated image DataLoader
    dataloader_iterator_test = iter(dataloader_in_gen_test)

    # Turn off gradients for evaluation (testing)
    with torch.no_grad():
        # Iterate over batches of real images from the real image DataLoader
        for _, (real_images, _) in enumerate(dataloader_in_test):
            
            # Try to get the next batch of generated images
            try:
                gen_imgs, _ = next(dataloader_iterator_test)
            except StopIteration:
                # If we reach the end of the generated DataLoader, restart the iterator
                dataloader_iterator_test = iter(dataloader_in_gen_test)
                gen_imgs, _ = next(dataloader_iterator_test)

            # Optionally add Gaussian noise to the real and generated images
            if noise:
                real_images = torch.tensor(random_noise(real_images, mode='gaussian', mean=0, var=0.0001, clip=False))
                gen_imgs = torch.tensor(random_noise(gen_imgs.detach().cpu().numpy(), mode='gaussian', mean=0, var=0.0001, clip=False))

            # Ensure that the image tensors are of the correct type for PyTorch
            real_images = real_images.type(torch.FloatTensor)
            gen_imgs = gen_imgs.type(torch.FloatTensor)

            # Move generated images to the GPU if available
            if cuda:
                gen_imgs = gen_imgs.cuda()
                real_images = real_images.cuda()

            # Concatenate real and generated images to form the input batch
            X = torch.cat((real_images, gen_imgs), dim=0)
            # Create corresponding labels for real (1) and generated (0) images
            y = torch.cat((torch.ones(len(real_images)), torch.zeros(len(gen_imgs))), dim=0).long().cuda()

            # Check if the input X has 3 dimensions (batch_size, height, width)
            if X.ndimension() == 3:
                # Add a channel dimension to the input, assuming RGB images (3 channels)
                X = X.unsqueeze(1).repeat(1, 3, 1, 1)  # Now the shape will be [batch_size, 3, height, width]

            # Perform the forward pass through the model (ResNet)
            outputs = resnet(X)
            outputs = outputs.type(torch.FloatTensor).to(device)
            y = y.to(device)

            # Calculate the loss between the predicted outputs and true labels
            loss = criterion(outputs, y)
            losses.append(loss.item())  # Record the loss for this batch

            # Append the true labels for this batch
            y_true.append(y.cpu())

            # Get the predicted labels (class with highest probability)
            _, predicted = torch.max(outputs.data, 1)

            # Append the predicted outputs for this batch
            all_outs.append(predicted.cpu())

            # Calculate accuracy for this batch and store it
            acc_score = accuracy_score(y.cpu(), predicted.cpu())
            av_accs.append(acc_score)

    # Flatten all_outs and y_true so that they contain predictions and labels per image
    all_outs = torch.cat(all_outs).numpy()  # Convert to NumPy array if needed
    y_true = torch.cat(y_true).numpy()

    # Calculate the average loss and accuracy across all batches
    avg_loss = sum(losses) / len(losses)
    avg_acc = sum(av_accs) / len(av_accs)

    # Print the average accuracy and loss for the entire test set
    print(f"Image mode accuracy: {avg_acc:.4f}")
    print(f"Test loss: {avg_loss:.4f}")
    
    # Return the predicted outputs, true labels, and average accuracy
    return all_outs, y_true, avg_acc
