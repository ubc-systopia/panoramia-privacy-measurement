import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import argparse
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import opacus
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torchvision import models
import random
import torchvision
from opacus import PrivacyEngine
import wandb
import tqdm
from ResNet import BasicBlock
from wideresnet_28_10 import WideResNet

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="$HOME/data")
parser.add_argument("--out_path", type=str, default="$HOME/dp_baselines/outputs")
parser.add_argument("--save_ckpt", default=False, action='store_true')
parser.add_argument("--non_private", default=False, action='store_true')
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--debug", default=False, action='store_true')
parser.add_argument("--exp_group", type=str, default="tmp")
parser.add_argument("--exp_name", type=str, default="tmp")
parser.add_argument("--model_type", choices=['resnet18', 'resnet50', 'wide-resnet'])
parser.add_argument("--lr", default=.1, type=float)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--dp_noise_multiplier", default=1., type=float)
parser.add_argument("--dp_l2_norm_clip", default=1., type=float)
parser.add_argument("--virtual_batch_size", type=int, default=32)
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument("--dp_epsilon", default=1., type=float)
parser.add_argument("--eval_every", default=100, type=int)
parser.add_argument("--seed", default=1024, type=int)


def main(conf):
    seed = conf.seed
    random.seed(seed)
    # These values, specific to the CIFAR10 dataset, are assumed to be known.
    # If necessary, they can be computed with modest privacy budgets.
    CIFAR10_MEAN = (0.485, 0.456, 0.406)
    CIFAR10_STD_DEV = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
    ])

    trainset = torchvision.datasets.CIFAR10(root=conf.data_path, train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=conf.data_path, train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=conf.batch_size, shuffle=False, num_workers=2)

    if conf.model_type == 'resnet50':
        model = models.resnet50(num_classes=10)
    elif conf.model_type == 'resnet18':
        model = models.resnet18(num_classes=10)
    elif conf.model_type == 'wide-resnet':
        model = WideResNet(block=BasicBlock, num_blocks=[4, 4, 4], widen_factor=2)
    else:
        raise NotImplementedError

    if not conf.non_private:
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            print(errors[-1])
            model = ModuleValidator.fix(model)  # batch norm -> group norm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=conf.lr)  # momentum=0.9, weight_decay=0.0005

    if not conf.non_private:
        privacy_engine = PrivacyEngine(
            accountant='prv'
            # accountant='rdp'
        )

        # if calculating noise multiplier given target epsilon
        # model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        #     module=model,
        #     optimizer=optimizer,
        #     data_loader=train_loader,
        #     epochs=conf.epochs,
        #     target_epsilon=conf.dp_epsilon,
        #     target_delta=conf.delta,
        #     max_grad_norm=conf.dp_l2_norm_clip,
        # )

        # if setting noise multiplier
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=conf.dp_noise_multiplier,
            max_grad_norm=conf.dp_l2_norm_clip,
        )

    # if not conf.debug:
    #     wandb.log({'noise_multiplier': optimizer.noise_multiplier, 'clipping_value': optimizer.max_grad_norm})

    cur_best_test_acc = 0
    cur_best_model = None
    cur_best_step = None
    step_counter = 0

    scaler = torch.cuda.amp.GradScaler(enabled=False)

    for epoch in tqdm.tqdm(range(conf.epochs)):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        if conf.non_private:
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                step_counter += 1
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            train_L = train_loss / (len(train_loader))
            train_acc = 100. * correct / total

            if not conf.debug:
                wandb.log({'epoch': epoch, 'train_step': step_counter, 'train_loss': train_L, 'train_acc': train_acc})
        else:
            with BatchMemoryManager(
                    data_loader=train_loader,
                    max_physical_batch_size=conf.virtual_batch_size,
                    optimizer=optimizer
            ) as memory_safe_data_loader:

                for inputs, targets in memory_safe_data_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    step_counter += 1
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                train_L = train_loss / (len(train_loader))
                train_acc = 100. * correct / total
                eps = privacy_engine.get_epsilon(delta=conf.delta)

                if not conf.debug:
                    wandb.log({'epoch': epoch, 'train_step': step_counter, 'train_loss': train_L, 'train_acc': train_acc, 'train_eps': eps})

                if eps > conf.dp_epsilon:
                    raise ValueError('Exceeding target epsilon value.')

        if step_counter % conf.eval_every == 0:
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                test_L = test_loss / (len(test_loader))

            test_acc = 100. * correct / total

            if conf.non_private:
                eps = np.nan

            if not conf.debug:
                wandb.log({'test_step': step_counter, 'test_loss': test_L, 'test_acc': test_acc, 'test_eps': eps})

    # save ckpts
    if conf.save_ckpt:
        if not os.path.isdir(conf.out_path):
            os.mkdir(conf.out_path)
        if conf.non_private:
            file_name = "ckpt_{}_np_s{}_last_torch1.pth".format(conf.model_type, step_counter)
        else:
            file_name = "ckpt_{}_eps{}_s{}_last_torch1.pth".format(conf.model_type, conf.dp_epsilon, step_counter)
        torch.save(model, os.path.join(conf.out_path, file_name))


if __name__ == '__main__':
    conf = parser.parse_args()
    main(conf)
