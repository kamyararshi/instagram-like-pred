import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam


from argparse import ArgumentParser
from tqdm import tqdm, trange
import os
from time import gmtime, strftime


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import datasetclass
from models import network


def train_one_epoch(train_loader, optimizer, criterion, model, device):
    """
    
    """
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, batch in enumerate(tqdm(train_loader, desc=f'batch')):
        # Every data instance is an input + label pair
        x, y = batch
        x, y = x.to(device), y.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(x)

        # Compute the loss and its gradients
        loss = criterion(outputs, y.to(torch.long))
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            #tb_x = epoch_index * len(train_loader) + i + 1
            #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def train(num_epochs, train_loader, optimizer, criterion, model, test_loader=None, device='cuda'):
    """
    
    """

    epoch_number = 0

    if device=='cuda':
        device = ('cpu' if not torch.cuda.is_available() else 'cuda')

    avg_loss = []
    for epoch in trange(num_epochs, desc="epoch"):
        model.train(True)
        avg_loss.append(train_one_epoch(train_loader, optimizer, criterion, model, device))

        if test_loader is not None:
            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(test_loader):
                    vinputs, vlabels = vdata
                    vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                    voutputs = model(vinputs)
                    vloss = criterion(voutputs, vlabels.to(torch.long))
                    running_vloss += vloss

        epoch_number += 1

    return avg_loss, epoch


def calculate_accuracy(predicted_labels, true_labels):
    total_samples = len(true_labels)
    correct_predictions = (predicted_labels == true_labels).sum().item()
    accuracy = correct_predictions / total_samples
    return accuracy



def save_model(model, optimizer,  epoch, log_dir, loss):
    """
    """
    log_path = os.path.join(log_dir, 'model.pth.tar')

    # Save loss vals
    with open(os.path.join(log_dir, 'output.txt'), 'w') as file:
        # Iterate over the list
        for element in avg_losses:
            # Write each element to the file
            file.write(element + '\n')
    
    # Save model
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss[-1],
            }, log_path)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--path', default="./preprocessing/final_data_ready.csv")
    parser.add_argument('--log', default="./weights")
    parser.add_argument('--name', default="LikesCategory")
    parser.add_argument('--num_epochs', default=100)
    parser.add_argument("--eval", dest="eval", action="store_true", help="Do eval on test set")
    parser.add_argument('--device', default='cpu')
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
    dataset_train = datasetclass.InstagramUserData(args.path, device, train=True)
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    if args.eval:
        # Test
        dataset_test = datasetclass.InstagramUserData(args.path, device, train=False)
        test_loader = DataLoader(dataset_test, batch_size=16, shuffle=False)

    else:
        test_loader = None
    
    # In_dim
    some_data, some_label = next(iter(train_loader))
    in_dim = some_data.shape[1]

    # Init Model
    model = network.LikeCategoryPredictor(in_dim=in_dim).to(device=device)

    print("Testing")
    # First Check if everything is compatible then:
    with torch.no_grad():
        some_data = some_data.to(device)
        _ = model(some_data)
    print("Testing Done!")


    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=.0005)

    # Training Loop
    print("Training Starts ...")
    avg_losses, epoch = train(args.num_epochs, train_loader, optimizer, criterion, model, test_loader=test_loader, device='cuda')

    # Save model
    save_model(model, optimizer,  epoch, log_dir, avg_losses)

    
