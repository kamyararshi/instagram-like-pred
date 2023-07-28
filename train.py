import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from argparse import ArgumentParser
from tqdm import tqdm, trange
import os
from time import gmtime, strftime


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import datasetclass
from models import network
from utils import *

import colorama
from colorama import Fore, Back, Style

tb_writer = SummaryWriter('runs/LikesRegression')
colorama.init(autoreset=True)

def train_one_epoch(train_loader, optimizer, criterion, model, device):
    """
    
    """
    running_loss = 0.
    avg_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, batch in enumerate(tqdm(train_loader)):
        # Every data instance is an input + label pair
        data, img, y = batch
        data, img, y = data.to(device), img.to(device), y.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model((data, img))

        # Compute the loss and its gradients
        y = y.view(-1, 1)

        loss = criterion(outputs, y.to(torch.float32))
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        batch_size = i

    # Average of all Loss in Running Losses
    avg_loss = running_loss/batch_size

    # Return Average Loss in Whole Batch
    return avg_loss

def train(num_epochs, train_loader, optimizer, criterion, scheduler, model, test_loader=None, device='cuda'):
    """
    
    """
    if device=='cuda:0':
        device = ('cpu' if not torch.cuda.is_available() else 'cuda:0')

    avg_loss = []
    model.train(True)
    for epoch in trange(num_epochs, desc="epoch"):
        ep_loss = train_one_epoch(train_loader, optimizer, criterion, model, device)
        avg_loss.append(ep_loss)
        tb_writer.add_scalar('Loss/train', ep_loss, int(epoch))
        
        # Do eval on test set after training is done
        if test_loader is not None:
            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(test_loader):
                data, img, vlabels = vdata
                vinputs = (data.to(device), img.to(device))
                vlabels = vlabels.to(device)
                vlabels = vlabels.view(-1,1)
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels.to(torch.float32))
                running_vloss += vloss
            print(Fore.GREEN + f"Average Validation Loss After Training = {vloss/i}\n")
            tb_writer.add_scalar('Loss/validation', vloss, int(epoch))

        scheduler.step()
    

    return avg_loss, epoch


def calculate_accuracy(predicted_labels, true_labels):
    total_samples = len(true_labels)
    correct_predictions = (predicted_labels == true_labels).sum().item()
    accuracy = correct_predictions / total_samples
    return accuracy

def calculate_mse(predicted_labels, true_labels):
    return ((predicted_labels - true_labels) ** 2).mean().item()

def calculate_rmse(predicted_labels, true_labels):
    return ((predicted_labels - true_labels) ** 2).mean().sqrt().item()

def calculate_r2(predicted_labels, true_labels):
    true_labels = true_labels.float()
    predicted_labels = predicted_labels.float()
    ss_res = ((predicted_labels - true_labels) ** 2).sum()
    ss_tot = ((true_labels - true_labels.mean()) ** 2).sum()
    r2_score = 1 - ss_res/ss_tot
    return r2_score.item()

def calculate_mae(predicted_labels, true_labels):
    return torch.abs(predicted_labels - true_labels).mean().item()


def save_model(model, optimizer,  epoch, log_dir, loss):
    """Saves the weights and loss logs
    """
    log_path = os.path.join(log_dir, 'model.pth.tar')

    # Save loss vals
    with open(os.path.join(log_dir, 'losses.txt'), 'w') as file:
        # Iterate over the list
        for element in avg_losses:
            # Write each element to the file
            file.write(str(element) + '\n')
    
    # Save model
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss[-1],
            }, log_path)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--path', default="preprocessing/final_data_pictures_numberLikes.csv")
    parser.add_argument('--log', default="log")
    parser.add_argument('--name', default="LikesRegressionCategory")
    parser.add_argument('--num_epochs', default=2)
    parser.add_argument("--eval", dest="eval", action="store_true", help="Do eval on test set")
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(eval=True)
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    if not os.path.exists(args.log):
        os.mkdir(args.log)
    

    if not os.path.exists(args.path):
        raise ValueError(f"Dataset does not exist here {args.path}")
    
    device = args.device
    log_dir = os.path.join(args.log, args.name+ ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime()))
    
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        


    # Prepare datasets
    # Train
    dataset_train = datasetclass.InstagramUserData(args.path, seperator=';', device=device, train=True)
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, collate_fn=collate_fn)
    if args.eval:
        # Test
        dataset_test = datasetclass.InstagramUserData(args.path, seperator=';', device=device,train=False)
        test_loader = DataLoader(dataset_test, batch_size=16, shuffle=False, collate_fn=collate_fn)

    else:
        test_loader = None
    
    # In_dim
    some_data, some_image, some_label = next(iter(train_loader))
    in_dim = some_data.shape[1]
    image_in_dim = some_image.shape[1]

    # Init Model
    model = network.LikeNumberPredictor(in_dim=in_dim, image_in_dim=image_in_dim, out_classes=1).to(device=device)

    print("Testing the Model -----------------------------------------------------------------------------------")
    # First Check if everything is compatible then:
    with torch.no_grad():
        some_data = some_data.to(device)
        some_image = some_image.to(device)
        some_label = some_label.to(device)
        _ = model((some_data, some_image))
    print("Tests Finished --------------------------------------------------------------------------------------")


    # Define loss and optimizer
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=.001)

    # MultiTepLR
    scheduler = lr_scheduler.MultiStepLR(optimizer, [15, 30])

    # Training Loop
    print("Training the Model --------------------------------------------------------------------------------------")
    avg_losses, epoch = train(args.num_epochs, train_loader, optimizer, criterion, scheduler, model, test_loader=test_loader, device=device)
    print("Training Finished ---------------------------------------------------------------------------------------")

    print("Results on Some Data:\n")
    model.eval()
    some_output = model((some_data, some_image))
    print("MSE: ", round(calculate_mse(some_output, some_label), 4))
    print("RMSE: ", round(calculate_rmse(some_output, some_label), 4))
    print("MAE: ", round(calculate_mae(some_output, some_label), 4))
    print("R2: ", round(calculate_r2(some_output.float(), some_label.float()), 4))

    # Save model
    print("Saving the Model at: ", log_dir)
    save_model(model, optimizer,  epoch, log_dir, avg_losses)

    
