import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from lightgbm import LGBMClassifier
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import optuna
from torch.utils.data import Dataset, DataLoader


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Slower with cpu 
device = torch.device('cpu')
print(device)

EPSILON=10.
DELTA=10**-6
#MAX_GRAD_NORM= 5
MAX_PHYSICAL_BATCH_SIZE=512
SCHEDULER=True
#STEP_SIZE=10
#GAMMA=.1

class DNN(nn.Module):
    def __init__(self, input_space):
        super().__init__()
        self.input_layer= nn.Linear(input_space,150)
        self.relu= nn.ReLU()
        self.l1= nn.Linear(150, 100)
        self.l2= nn.Linear(100, 50)
        self.l3= nn.Linear(50, 10)
        self.logit=nn.Linear(10,1)
        self.sig= nn.Sigmoid()
    def forward(self,x):
        a= self.relu(self.input_layer(x))
        a= self.relu(self.l1(a))
        a=self.relu(self.l2(a))
        a=self.relu(self.l3(a))
        return self.sig(self.logit(a))
    
    def extract_emb(self,x):
        a= self.relu(self.input_layer(x))
        a= self.relu(self.l1(a))
        a=self.relu(self.l2(a))
        return self.relu(self.l3(a))


def train_network(net, epochs,train_dataloader, valid_dataloader, optim, criterion, pth):
    nb_batch_train= len(train_dataloader)
    nb_batch_valid= len(valid_dataloader)
    best_loss= 1000000
    for epoch in tqdm(range(epochs)):
        total_loss=0
        total_loss_valid=0
        correct=0
        total=0
        for i, batch in enumerate(train_dataloader):
            feature, target= batch[0], batch[1]
            feature.to(device)
            target.to(device)
            optim.zero_grad()
            pred= net(feature)
            loss= criterion(pred, target)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        with torch.no_grad():
            for batch in valid_dataloader:
                feature, target= batch[0], batch[1]
                feature.to(device)
                target.to(device)
                pred= net(feature)
                loss= criterion(pred, target)
                total_loss_valid+= loss
                predicted = np.where(pred.detach().numpy()>.5, 1,0)
                correct+= (predicted==target.numpy()).sum().item()
                total += target.size(0)
        print('epoch {}/{} training loss: {}, valid loss: {}, valid accuracy: {}'.format(epoch, epochs, total_loss/ nb_batch_train, total_loss_valid/ nb_batch_valid, correct/total ))
        if total_loss_valid/ nb_batch_valid < best_loss:
            print('\n best loss improove from {} to {} saving model'.format(best_loss,total_loss_valid/ nb_batch_valid ))
            best_loss= total_loss_valid/ nb_batch_valid
            torch.save(net, pth) #put as param


def train_epoch_dp(model, train_loader, optimizer, epoch, privacy_engine, logging):
    nb_batch_train= len(train_loader)
    model.train()
    criterion = nn.BCELoss()

    losses = []
    top1_acc = []
    
    with BatchMemoryManager(
        data_loader=train_loader, 
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
        optimizer=optimizer
    ) as memory_safe_data_loader:

        for i, (images, target) in enumerate(memory_safe_data_loader):   
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()

            if logging:
                if (i+1) % nb_batch_train == 0:
                    epsilon = privacy_engine.get_epsilon(DELTA)
                    print(
                        f"\tTrain Epoch: {epoch} \t"
                        f"Loss: {np.mean(losses):.6f} "
                        f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                        f"(ε = {epsilon:.2f}, δ = {DELTA})"
                    )
    return np.mean(losses)

def train_network_dp(net, epochs,train_dataloader, valid_dataloader, optim, criterion, pth, epsilon, max_grad_norm,scheduler=SCHEDULER, logging=True):
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=net,
        optimizer=optim,
        data_loader=train_dataloader,
        epochs=epochs,
        target_epsilon=epsilon,
        target_delta=DELTA,
        max_grad_norm=max_grad_norm,
    )
    if scheduler:
        lr_schedule= torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.1)
    for epoch in range(epochs):
        if logging:
            loss= train_epoch_dp(model, train_loader, optimizer, epoch + 1, privacy_engine, logging)
        elif epoch+1== epochs: 
            loss= train_epoch_dp(model, train_loader, optimizer, epoch + 1, privacy_engine, True)
        else:
            loss= train_epoch_dp(model, train_loader, optimizer, epoch + 1, privacy_engine, False)
        if scheduler:
            lr_schedule.step()
    torch.save(net, pth)
    return loss
    

# class LGB():
#     def __init__(self):
#         self.gb= LGBMClassifier(max_depth=3, n_estimators=100, learning_rate=0.01,class_weight='balanced')
#     def train(self, train_x, train_y):
#         self.gb.fit(train_x, train_y)
#     def oracle_pred(self, x,y, evaluate=False):
#         pred=self.gb.predict(x)
#         if evaluate:
#             total= x.shape[0]
#             correct= (pred==y).sum()
#             print('lr oracle accuracy:', correct/total)
#         return pred

def accuracy(preds, labels):
    return (preds == labels).mean()


def optimize_hyp_param(dataset, nb_trial):
    def objective(trial):
        # 1. Define search space
        MAX_GRAD_NORM = trial.suggest_categorical("MAX_GRAD_NORM", [i/10 for i in list(range(5,105,5))])
        SCHEDULER = trial.suggest_categorical("SCHEDULER", [True, False])
        STEP_SIZE = trial.suggest_int("STEP_SIZE", 1, 100)
        GAMMA = trial.suggest_float("GAMMA", 0.001, 1.0)
        EPOCHS=trial.suggest_categorical("EPOCHS", list(range(10,105,5)))
        BATCH_SIZE=trial.suggest_categorical("BATCH_SIZE", [32, 64, 128, 256, 512, 1024])
        LR= trial.suggest_float("LR", 0.0001, 0.01)

        
        # 2. Define and train your model using these hyperparameters
        net = DNN(106)
        optim = torch.optim.Adam(net.parameters(), lr=LR) 
        
        if SCHEDULER:
            scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=STEP_SIZE, gamma=GAMMA)
        
        criterion = nn.BCELoss()
        train_dataloader = DataLoader(dataset, BATCH_SIZE, True)

        # Train your network (you'll need to pass the hyperparameters to your train function)
        loss=train_network_dp(net, EPOCHS, train_dataloader, None,optim, criterion, 'models/temp.pth', EPSILON, MAX_GRAD_NORM, SCHEDULER, logging=False)

        # 3. Compute the value you're trying to optimize (e.g. validation loss, accuracy, etc.)
        # For demonstration purposes, I'm assuming the validation loss is returned by your training function
        
        return loss

    # Create a study object and specify the direction is 'minimize'.
    study = optuna.create_study(direction='minimize')

    # Optimize the study, the objective function is passed in as the first argument.
    study.optimize(objective, n_trials=nb_trial, n_jobs=4)

    # Results
    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)
    print('Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
    return trial
