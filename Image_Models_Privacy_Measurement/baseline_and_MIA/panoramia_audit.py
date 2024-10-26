# Main file for the Panoramia pipeline 
# Use to run the MIA and Baseline attacks, after the target and helper models have been trained and non-member samples have been generated
import torch
import csv
import random
import os
import sys
import warnings
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, roc_curve, precision_recall_curve, auc, precision_score, recall_score
import pandas as pd
import argparse
import yaml
from torchvision import models
from image_module import *
from losses_module import *

# Suppress warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Set CUDA if available
cuda = torch.cuda.is_available()

# Load YAML configuration
with open('config.yaml', 'r') as f:
    print("reading config file...")
    config = yaml.safe_load(f)

# Set parameters from YAML config
opt_experiment = argparse.Namespace(**config['experiment'])
opt_paths = argparse.Namespace(**config['paths'])
opt_sampling = argparse.Namespace(**config['sampling'])
opt_models = argparse.Namespace(**config['models'])
opt_attacks = argparse.Namespace(**config['classifiers'])
opt_training = argparse.Namespace(**config['training'])
opt_dataset = argparse.Namespace(**config['dataset'])


opt = argparse.Namespace(**{**vars(opt_experiment), **vars(opt_paths), **vars(opt_sampling), **vars(opt_models), **vars(opt_models), **vars(opt_attacks), **vars(opt_training), **vars(opt_dataset) })

# Set the seed
random.seed(opt.seed)
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed_all(opt.seed)

print(opt)

# Load pre-trained models and also import any necessary modules for the target and helper models
net_g = torch.load(opt.model_g_path) 
net_f = torch.load(opt.model_f_path)

if cuda:
    net_f.cuda()
    net_g.cuda()

# CIFAR10 dataset
full_dataset = datasets.CIFAR10(
    root=opt.data_path,
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.Resize(config['dataset']['cifar10_transform']['resize']),
        transforms.ToTensor(),
        transforms.Normalize(config['dataset']['cifar10_transform']['normalize']['mean'],
                             config['dataset']['cifar10_transform']['normalize']['std'])
    ])
)

# Subset indices for training and testing
all_indices = [i for i in range(len(full_dataset))] # data used to train target model f
in_train_indices = all_indices[int(len(all_indices) / 6):int(len(all_indices) / 4)] # subset of data to train MIA and baseline
in_test_indices = all_indices[int(len(all_indices) / 4):int(len(all_indices) / 3)]  # subset of data to test MIA and baseline

# Create real data subsets for training and tesing MIA and Baseline 
subset_real_test = Subset(full_dataset, in_test_indices)
subset_in_train = Subset(full_dataset, in_train_indices)
# Define your criterion (loss function) for classification
criterion = nn.CrossEntropyLoss()
# Bernoulli sampling function for real and synthetic datasets
def bernoulli_sampling(p, real_data, synthetic_data):
    """
    Perform Bernoulli sampling between real and synthetic datasets.
    
    Args:
        p (float): Probability of choosing from the real dataset.
        real_data (Dataset): Real dataset.
        synthetic_data (Dataset): Synthetic dataset.
    
    Returns:
        Tuple: Sampled real and synthetic datasets.
    """
    real_sampled, synthetic_sampled = [], []
    
    for i in range(len(real_data)):
        if random.random() < p:
            real_sampled.append(real_data[i])
        else:
            synthetic_sampled.append(synthetic_data[i % len(synthetic_data)])  # Wrap around synthetic data
    
    return real_sampled, synthetic_sampled

# Custom Dataset Class for Synthetic Data
class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.filtered_dataset = [(image_path, labels) for image_path, labels in self.dataset if any(labels)]

    def __getitem__(self, index):
        image_path, labels = self.filtered_dataset[index]
        image = self.load_image(image_path)
        if self.transform is not None:
            image = self.transform(image)
        label = self.get_label(labels)
        return image, label

    def __len__(self):
        return len(self.filtered_dataset)

    @staticmethod
    def load_image(image_path):
        return Image.open(image_path)

    @staticmethod
    def get_label(labels):
        return labels.index(1)

# Load synthetic train data and labels
synthetic_data = []
synthetic_labels = []
with open(opt.label_train_file, 'r') as f:
    lines = f.readlines()
for line in lines:
    parts = line.strip().split('\t')
    image_name = parts[0]
    labels = [int(label) for label in parts[1:]]
    image_path = os.path.join(opt.synthetic_train_folder, image_name)
    synthetic_data.append(image_path)
    synthetic_labels.append(labels)

# Combine synthetic train data and labels into a dataset
synthetic_train_dataset = list(zip(synthetic_data, synthetic_labels))
synthetic_train_dataset = SyntheticDataset(synthetic_train_dataset, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(config['dataset']['cifar10_transform']['normalize']['mean'],
                         config['dataset']['cifar10_transform']['normalize']['std'])
]))


# Load synthetic test data and labels
synthetic_data = []
synthetic_labels = []
with open(opt.label_test_file, 'r') as f:
    lines = f.readlines()
for line in lines:
    parts = line.strip().split('\t')
    image_name = parts[0]
    labels = [int(label) for label in parts[1:]]
    image_path = os.path.join(opt.synthetic_test_folder, image_name)
    synthetic_data.append(image_path)
    synthetic_labels.append(labels)

# Combine synthetic test data and labels into a dataset
synthetic_test_dataset = list(zip(synthetic_data, synthetic_labels))
synthetic_test_dataset = SyntheticDataset(synthetic_test_dataset, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(config['dataset']['cifar10_transform']['normalize']['mean'],
                         config['dataset']['cifar10_transform']['normalize']['std'])
]))


# Shuffle and create indices for the datasets
def shuffle_and_create_indices(combined_in_train_dataset, combined_syn_train_dataset):
    """
    Shuffle indices for both real and synthetic training datasets.

    Parameters:
    combined_in_train_dataset (Dataset): Real dataset.
    combined_syn_train_dataset (Dataset): Synthetic dataset.

    Returns:
    Tuple: Shuffled indices for real and synthetic datasets.
    """
    gen_train_indices = [i for i in range(len(combined_syn_train_dataset))]
    in_train_indices = [i for i in range(len(combined_in_train_dataset))]
    random.shuffle(gen_train_indices)
    random.shuffle(in_train_indices)

    return gen_train_indices, in_train_indices

# Create DataLoaders for k samples
def create_dataloaders_for_k_samples(j, gen_train_indices, in_train_indices, combined_in_train_dataset, combined_syn_train_dataset, opt):
    """
    Create DataLoaders for k samples of both real and synthetic datasets.

    Parameters:
    j (int): Number of samples.
    gen_train_indices (list): Shuffled indices of the synthetic dataset.
    in_train_indices (list): Shuffled indices of the real dataset.
    combined_in_train_dataset (Dataset): Real dataset.
    combined_syn_train_dataset (Dataset): Synthetic dataset.
    opt: config arguments including batch size and CPU settings.

    Returns:
    Tuple[DataLoader, DataLoader]: DataLoaders for real and synthetic datasets.
    """
    k_data_subset = Subset(combined_in_train_dataset, random.sample(in_train_indices, j))
    k_gen_subset = Subset(combined_syn_train_dataset, random.sample(gen_train_indices, j))


    dataloader_in_train = DataLoader(
        k_data_subset,
        batch_size=opt.batch_size,
        num_workers=opt.n_cpu,
        shuffle=True
    )

    dataloader_in_gen_train = DataLoader(
        k_gen_subset,
        batch_size=opt.batch_size,
        num_workers=opt.n_cpu,
        shuffle=True
    )

    return dataloader_in_train, dataloader_in_gen_train

# Train the baseline and MIA classifiers
def train_baseline_and_mia_classifiers(dataloader_in_train, dataloader_in_gen_train, net_g, net_f, base_logit_classifier, mia_logit_classifier, image_trainer, logit_trainer, resnet, criterion, optimizer):
    """
    Train baseline and MIA classifiers using the raw training data.

    Parameters:
    dataloader_in_train (DataLoader): DataLoader for real training data.
    dataloader_in_gen_train (DataLoader): DataLoader for synthetic training data.
    net_g (torch.nn.Module): Model G.
    net_f (torch.nn.Module): Model F.
    base_logit_classifier (MLPClassifier): Baseline logit classifier.
    mia_logit_classifier (MLPClassifier): MIA logit classifier.
    image_trainer (function): Function to train the image model.
    logit_trainer (function): Function to train the logit classifier.
    resnet (torch.nn.Module): ResNet model.
    criterion (torch.nn.Module): Loss function.
    optimizer (torch.optim.Optimizer): Optimizer for training.

    Returns:
    Tuple[LogisticRegression, LogisticRegression]: Trained baseline and MIA meta models.
    """
    epochs = opt.n_epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    image_preds, y_train = image_trainer(epochs, resnet, dataloader_in_train, dataloader_in_gen_train, criterion, optimizer, scheduler, noise=False, early_stopping_patience=5, dropout_rate=0.5)
    # Train baseline logit classifier
    base_best_model, base_logit_preds = logit_trainer(dataloader_in_train, net_g, base_logit_classifier, dataloader_in_gen_train, noise=False)
    # Convert image_preds[0] and base_logit_preds to NumPy arrays if they are lists
    image_preds = np.array(image_preds) if isinstance(image_preds, list) else image_preds
    base_logit_preds = np.array(base_logit_preds) if isinstance(base_logit_preds, list) else base_logit_preds

    base_X_train_meta = np.concatenate([image_preds.reshape(-1, 1), np.reshape(base_logit_preds, (-1, 1))], axis=1)
    base_meta_model = LogisticRegression(random_state=opt.seed)
    base_meta_model.fit(base_X_train_meta, y_train)

    # Train MIA logit classifier
    mia_best_model, mia_logit_preds = logit_trainer(dataloader_in_train, net_f, mia_logit_classifier, dataloader_in_gen_train, noise=False)
    
    mia_X_train_meta = np.concatenate([image_preds.reshape(-1, 1), np.reshape(mia_logit_preds, (-1, 1))], axis=1)
    mia_meta_model = LogisticRegression(random_state=opt.seed)
    mia_meta_model.fit(mia_X_train_meta, y_train)

    return base_meta_model, mia_meta_model, base_best_model, mia_best_model

# Test the classifiers and compute evaluation metrics
def test_classifiers(resnet, best_base_model, best_mia_model, base_meta_model, mia_meta_model, in_test_loader, gen_test_loader, logit_tester, net_g, net_f):
    """
    Test baseline and MIA classifiers and compute relevant metrics.

    Parameters:
    resnet (torch.nn.Module): ResNet model.
    base_meta_model (LogisticRegression): Trained baseline meta model.
    mia_meta_model (LogisticRegression): Trained MIA meta model.
    in_test_loader (DataLoader): DataLoader for real test data.
    gen_test_loader (DataLoader): DataLoader for synthetic test data.
    logit_tester (function): Function to test the logit classifier.
    net_g (torch.nn.Module): Model G.
    net_f (torch.nn.Module): Model F.

    Returns:
    Tuple[float, float, float, float, float, float]: Accuracy, loss, and ROC-AUC scores for baseline and MIA classifiers.
    """
    # Get predictions on raw test images
    image_test_preds, y_val, _ = image_tester(resnet, in_test_loader, gen_test_loader, criterion, noise=False)

    # Test baseline logit classifier
    base_logit_testpreds, _, _, _, _, _, _ = logit_tester(in_test_loader, net_g, gen_test_loader, best_base_model, noise=False)

    # Convert image_test_preds[0] and base_logit_preds to NumPy arrays if they are lists
    image_test_preds = np.array(image_test_preds) if isinstance(image_test_preds, list) else image_test_preds
    base_logit_testpreds = np.array(base_logit_testpreds) if isinstance(base_logit_testpreds, list) else base_logit_testpreds
   
    X_val_meta = np.concatenate([image_test_preds.reshape(-1, 1), np.reshape(base_logit_testpreds, (-1, 1))], axis=1)
    y_val_pred = base_meta_model.predict(X_val_meta)
    y_val_proba = base_meta_model.predict_proba(X_val_meta)
    
    # Ensure that y_val is a list of PyTorch tensors
    if isinstance(y_val, np.ndarray):
        y_val = [torch.tensor(val).to(device) for val in y_val]

    # Ensure all tensors in y_val have at least 1 dimension (e.g., by adding an extra dimension to scalars)
    y_val = [tensor.unsqueeze(0) if tensor.dim() == 0 else tensor for tensor in y_val]

    # Now concatenate the tensors into y_val_tensor
    y_val_tensor = torch.cat(y_val).cpu()  # Concatenate and move to CPU if necessary

    # Convert y_val_tensor to a NumPy array (since it is now a concatenated tensor)
    y_val = y_val_tensor.numpy()  # Use the concatenated tensor here

    # Proceed with the same check for y_val_proba
    if isinstance(y_val_proba, torch.Tensor):
        y_val_proba = y_val_proba.cpu().numpy()

    # Now calculate the log loss and other metrics using the converted y_val
    val_loss = log_loss(y_val, y_val_proba[:, 1])
    val_acc = accuracy_score(y_val, y_val_pred)
    val_roc_auc = roc_auc_score(y_val, y_val_proba[:, 1])

    # Test MIA logit classifier
    mia_logit_testpreds, _, _, _, _, _, _ = logit_tester(in_test_loader, net_f, gen_test_loader, best_mia_model, noise=False)
    mia_X_val_meta = np.concatenate([image_test_preds.reshape(-1, 1), np.reshape(mia_logit_testpreds, (-1, 1))], axis=1)
    mia_y_val_pred = mia_meta_model.predict(mia_X_val_meta)
    mia_y_val_proba = mia_meta_model.predict_proba(mia_X_val_meta)
    mia_val_loss = log_loss(y_val, mia_y_val_proba[:, 1])
    mia_val_acc = accuracy_score(y_val, mia_y_val_pred)
    mia_val_roc_auc = roc_auc_score(y_val, mia_y_val_proba[:, 1])
    base_fpr, base_tpr, baseline_thresholds = roc_curve(y_val, y_val_proba[:,1])
    mia_fpr, mia_tpr, mia_thresholds = roc_curve(y_val, mia_y_val_proba[:,1])
    return val_acc, val_loss, val_roc_auc, mia_val_acc, mia_val_loss, mia_val_roc_auc, mia_y_val_proba, y_val_proba,  mia_thresholds,  baseline_thresholds, y_val, mia_y_val_pred, y_val_pred

# Evaluate precision and recall at different thresholds
def evaluate_thresholds(y_val, y_val_proba, thresholds, label):
    """
    Evaluate precision, recall, and other metrics at various thresholds.

    Parameters:
    y_val (Tensor): Ground truth labels for the validation data.
    y_val_proba (array): Predicted probabilities for the positive class.
    thresholds (list): List of thresholds to evaluate.
    label (str): Either 'MIA' or 'Baseline' for logging purposes.

    Returns:
    List: A list of results containing recall, total correct guesses, true negatives, true positives, false positives, and precision.
    """
    results, predictions = [], []
    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_val_proba[:, 1]]
        precision = precision_score(y_val, y_pred)
        true_positives = sum((true == 1 and pred == 1) for true, pred in zip(y_val, y_pred))
        true_negs = sum((true == 0 and pred == 0) for true, pred in zip(y_val, y_pred))
        false_positives = sum((true == 0 and pred == 1) for true, pred in zip(y_val, y_pred))
        total_correct_guesses = true_positives + true_negs
        recall = recall_score(y_val, y_pred)

        # Log the metrics for each threshold (if needed)
        # print(f"{label} Threshold: {threshold}, Precision: {precision}, Recall: {recall}, Correct Guesses: {total_correct_guesses}")
        
        # Append the results
        results.append([recall, total_correct_guesses, true_negs, true_positives, false_positives, precision])
        predictions.append(y_pred)
    return results, predictions

# Save results to CSV
def save_results_to_csv(results, file_name, header):
    """
    Save the evaluation results to a CSV file.

    Parameters:
    results (list): The results list to be saved.
    file_name (str): The name of the CSV file.
    header (list): The header row for the CSV file.

    Returns:
    None
    """
    with open(file_name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(results)

# Log and save results
def log_and_save_results(seed, j, val_acc, val_loss, val_roc_auc, mia_val_acc, mia_val_loss, mia_val_roc_auc):
    """
    Log and save test results to CSV files.

    Parameters:
    seed (int): Random seed for reproducibility.
    j (int): Number of samples.
    val_acc (float): Validation accuracy for the baseline classifier.
    val_loss (float): Validation loss for the baseline classifier.
    val_roc_auc (float): ROC-AUC score for the baseline classifier.
    mia_val_acc (float): Validation accuracy for the MIA classifier.
    mia_val_loss (float): Validation loss for the MIA classifier.
    mia_val_roc_auc (float): ROC-AUC score for the MIA classifier.

    Returns:
    None
    """
    dict_bs = {
        'k': [j], 'bs_acc': [val_acc], 'mia_acc': [mia_val_acc],
        'bs_log_loss': [val_loss], 'mia_log_loss': [mia_val_loss]
    }
    df = pd.DataFrame(dict_bs)
    # Check if the directory exists, and if not, create it
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)

    df.to_csv(f'{opt.result_dir}/Seed{seed}Test_accuracies_cifar.csv', index=False)

# Process dataset sizes and evaluate classifiers
def process_dataset_sizes(data_prior, combined_in_train_dataset, combined_syn_train_dataset, opt, net_g, net_f, base_logit_classifier, mia_logit_classifier, resnet, image_trainer, logit_trainer, logit_tester, image_tester, in_test_loader, gen_test_loader):
    """
    Loop over different dataset sizes, train/test classifiers, and save results.

    Parameters:
    data_prior (list): List of dataset sizes to process.
    combined_in_train_dataset (Dataset): Real dataset.
    combined_syn_train_dataset (Dataset): Synthetic dataset.
    opt (argparse.Namespace): Command-line arguments including batch size and CPU settings.
    net_g (torch.nn.Module): Model G.
    net_f (torch.nn.Module): Model F.
    base_logit_classifier (MLPClassifier): Baseline logit classifier.
    mia_logit_classifier (MLPClassifier): MIA logit classifier.
    resnet (torch.nn.Module): ResNet model.
    image_trainer (function): Function to train the image model.
    logit_trainer (function): Function to train the logit classifier.
    logit_tester (function): Function to test the logit classifier.
    image_tester (function): Function to test the image model.
    in_test_loader (DataLoader): DataLoader for real test data.
    gen_test_loader (DataLoader): DataLoader for synthetic test data.

    Returns:
    Tuple[list, list, list, list, list]: Dataset sizes, baseline accuracies, MIA accuracies, baseline losses, and MIA losses.
    """
    js, baseline_test_accs, mia_test_accs = [], [], []
    baseline_test_losses, mia_test_losses = [], []

    # Shuffle indices for the combined datasets
    gen_train_indices, in_train_indices = shuffle_and_create_indices(combined_in_train_dataset, combined_syn_train_dataset)

    for j in data_prior:
        # Create dataloaders for k-sized subsets
        dataloader_in_train, dataloader_in_gen_train = create_dataloaders_for_k_samples(
            j, gen_train_indices, in_train_indices, combined_in_train_dataset, combined_syn_train_dataset, opt
        )

        # Train the classifiers
        base_meta_model, mia_meta_model = train_baseline_and_mia_classifiers(
            dataloader_in_train, dataloader_in_gen_train, net_g, net_f, base_logit_classifier, mia_logit_classifier, image_trainer, logit_trainer, resnet, criterion, optimizer
        )

        # Test the classifiers
        val_acc, val_loss, val_roc_auc, mia_val_acc, mia_val_loss, mia_val_roc_auc = test_classifiers(
            resnet, base_best_model, mia_best_model, base_meta_model, mia_meta_model, in_test_loader, gen_test_loader, logit_tester, net_g, net_f
        )

        # Log and save the results
        log_and_save_results(opt.seed, j, val_acc, val_loss, val_roc_auc, mia_val_acc, mia_val_loss, mia_val_roc_auc)

        # Append results for plotting
        js.append(j)
        baseline_test_accs.append(val_acc)
        mia_test_accs.append(mia_val_acc)
        baseline_test_losses.append(val_loss)
        mia_test_losses.append(mia_val_loss)

    return js, baseline_test_accs, mia_test_accs, baseline_test_losses, mia_test_losses


# Main function to orchestrate the entire process
def main():
    
    # Bernoulli sampling for training and testing
    real_train_sampled, synthetic_train_sampled = bernoulli_sampling(
        config['sampling']['train_data_probability'], subset_in_train, synthetic_train_dataset)
    real_test_sampled, synthetic_test_sampled = bernoulli_sampling(
        config['sampling']['test_data_probability'], subset_real_test, synthetic_test_dataset)
    print("member and non-member datasets created..")
    # Shuffle indices for the combined datasets
    gen_train_indices, in_train_indices = shuffle_and_create_indices(real_train_sampled, synthetic_train_sampled)

    # Create DataLoaders
    real_train_loader = DataLoader(subset_in_train, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
    synthetic_train_loader = DataLoader(synthetic_train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
    real_test_loader = DataLoader(real_test_sampled, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    synthetic_test_loader = DataLoader(synthetic_test_sampled, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    # print(f"Number of samples in the member train dataset: {len(real_train_loader.dataset)}")
    # print(f"Number of samples in the non-member train dataset: {len(synthetic_train_loader.dataset)}")
    print(f"Number of samples in the member test dataset: {len(real_test_loader.dataset)}")
    print(f"Number of samples in the non-member test dataset: {len(synthetic_test_loader.dataset)}")

    #  Load pre-trained target and helper models respectively
    net_f = torch.load(opt.model_f_path)
    net_g = torch.load(opt.model_g_path)
    if cuda:
        net_f.cuda()
        net_g.cuda()

    print("target and helper models loaded")

    # Setup ResNet model for image only binary classification task
    resnet = models.resnet50(pretrained=config['models']['resnet']['pretrained'])
    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_ftrs, config['models']['resnet']['num_classes'])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet.parameters(), lr=config['models']['resnet']['optimizer']['learning_rate'],
                                momentum=config['models']['resnet']['optimizer']['momentum'])
    if cuda:
        resnet = resnet.cuda()

    #  Initialize classifiers (baseline and MIA)
    print("initialize baseline and mia attacks")
    base_logit_classifier = MLPClassifier(hidden_layer_sizes=config['classifiers']['baseline_logit_classifier']['hidden_layer_sizes'],
                                          activation=config['classifiers']['baseline_logit_classifier']['activation'],
                                          solver=config['classifiers']['baseline_logit_classifier']['solver'],
                                          alpha=config['classifiers']['baseline_logit_classifier']['alpha'],
                                          random_state=config['classifiers']['baseline_logit_classifier']['random_state'])
    mia_logit_classifier = MLPClassifier(hidden_layer_sizes=config['classifiers']['mia_logit_classifier']['hidden_layer_sizes'],
                                         activation=config['classifiers']['mia_logit_classifier']['activation'],
                                         solver=config['classifiers']['mia_logit_classifier']['solver'],
                                         alpha=config['classifiers']['mia_logit_classifier']['alpha'],
                                         random_state=config['classifiers']['mia_logit_classifier']['random_state'])

    
    # Placeholders for evaluation metrics
    js, baseline_test_accs, mia_test_accs = [], [], []
    baseline_test_losses, mia_test_losses = [], []
    data_prior = opt.data_prior
    # Loop through data_prior to process different dataset sizes
    for j in data_prior:
        print(f"Processing {j} samples of member and non-members each...")
    
        # Step 1: Create DataLoaders for k-sized subsets of real and synthetic data
        dataloader_in_train, dataloader_in_gen_train = create_dataloaders_for_k_samples(
            j, gen_train_indices, in_train_indices, real_train_sampled, synthetic_train_sampled, opt
        )
        
        # Step 2: Train baseline and MIA classifiers using the DataLoaders
        print("Training MIA and Baseline")
        base_meta_model, mia_meta_model, base_best_model, mia_best_model = train_baseline_and_mia_classifiers(
            dataloader_in_train, dataloader_in_gen_train, net_g, net_f, base_logit_classifier, mia_logit_classifier,
            image_trainer, logit_trainer, resnet, criterion, optimizer
        )

        # Step 3: Test classifiers on the validation/test set and collect metrics
        print("Testing MIA and Baseline")
        val_acc, val_loss, val_roc_auc, mia_val_acc, mia_val_loss, mia_val_roc_auc, mia_y_val_proba, y_val_proba, mia_thresholds, baseline_thresholds, y_val, mia_predictions, baseline_predictions = test_classifiers(
            resnet, base_best_model, mia_best_model, base_meta_model, mia_meta_model, real_test_loader, synthetic_test_loader, logit_tester, net_g, net_f
        )
        
        print(val_acc, val_loss, val_roc_auc, mia_val_acc, mia_val_loss, mia_val_roc_auc)
        # Step 4: Save and log accuracy and loss results for current dataset size `j`
        print("logging results...")
        log_and_save_results(opt.seed, j, val_acc, val_loss, val_roc_auc, mia_val_acc, mia_val_loss, mia_val_roc_auc)
    
        # Step 5: Append results to lists for later analysis or plotting
        js.append(j)
        baseline_test_accs.append(val_acc)
        mia_test_accs.append(mia_val_acc)
        baseline_test_losses.append(val_loss)
        mia_test_losses.append(mia_val_loss)
    
        # Step 6: Evaluate thresholds and save and log recall, precision, number of guesses etc to generate prec-recall and privacy-recall plots
        # Use this function if you want to use the thresholds from the roccurve sklearn function
        # Otherwise use the csv saved in line 617 "mia_baseline_predictions_Seed{opt.seed}k{j}.csv" that is compatible with the plotting scripts in the root of this repository
        print("evaluating thresholds, recall, precision, TP, FP,...")
        mia_results, mia_preds = evaluate_thresholds(y_val, mia_y_val_proba, mia_thresholds, label="MIA")
        save_results_to_csv(mia_results, f'{opt.result_dir}/mia_results_Seed{opt.seed}k{j}.csv',
                            header=["Recall", "Correct Guesses", "True Negatives", "True Positives", "False Positives", "Precision"])
    
        base_results, baseline_preds = evaluate_thresholds(y_val, y_val_proba, baseline_thresholds, label="Baseline")
        save_results_to_csv(base_results, f'{opt.result_dir}/baseline_results_Seed{opt.seed}k{j}.csv',
                            header=["Recall", "Correct Guesses", "True Negatives", "True Positives", "False Positives", "Precision"])
        # Path to the checkpoint
        target_checkpoint_path = opt.model_f_path

        # Extract the file name without extension (but including epoch)
        target_model_name = os.path.splitext(os.path.basename(target_checkpoint_path))[0]  # model_name_epoch_10
        # Combine data for plotting into a DataFrame, saving labels as well as baseline and MIA predictions 
        data = {
            'member': y_val,
            'baseline_pred': baseline_predictions,
            f'mia_pred_{target_model_name}': mia_predictions
        }

        df = pd.DataFrame(data)

        # Save as CSV with headers
        csv_filename = f'{opt.result_dir}/mia_baseline_predictions_Seed{opt.seed}k{j}.csv'
        df.to_csv(csv_filename, index=False)
        
        # Re-Setup ResNet model for image only binary classification task for the next dataset size j
        resnet = models.resnet50(pretrained=config['models']['resnet']['pretrained'])
        num_ftrs = resnet.fc.in_features
        resnet.fc = torch.nn.Linear(num_ftrs, config['models']['resnet']['num_classes'])
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(resnet.parameters(), lr=config['models']['resnet']['optimizer']['learning_rate'],
                                    momentum=config['models']['resnet']['optimizer']['momentum'])
        if cuda:
            resnet = resnet.cuda()
    
        #  Re-Initialize classifiers (baseline and MIA)  for the next dataset size j
        print("initialize baseline and mia attacks")
        base_logit_classifier = MLPClassifier(hidden_layer_sizes=config['classifiers']['baseline_logit_classifier']['hidden_layer_sizes'],
                                              activation=config['classifiers']['baseline_logit_classifier']['activation'],
                                              solver=config['classifiers']['baseline_logit_classifier']['solver'],
                                              alpha=config['classifiers']['baseline_logit_classifier']['alpha'],
                                              random_state=config['classifiers']['baseline_logit_classifier']['random_state'])
        mia_logit_classifier = MLPClassifier(hidden_layer_sizes=config['classifiers']['mia_logit_classifier']['hidden_layer_sizes'],
                                             activation=config['classifiers']['mia_logit_classifier']['activation'],
                                         solver=config['classifiers']['mia_logit_classifier']['solver'],
                                         alpha=config['classifiers']['mia_logit_classifier']['alpha'],
                                         random_state=config['classifiers']['mia_logit_classifier']['random_state'])

       

# Run the main function
if __name__ == "__main__":
    main()
