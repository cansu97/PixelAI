import torch
import torch.nn as nn

import os
import argparse

import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pickle
import matplotlib.pyplot as plt

from conv_decoder import *

import sys
sys.path.append("..")

from process_data import *

parser = argparse.ArgumentParser(description='Model training')
parser.add_argument('--verbose', default='1', type=int, help='verbose = [0, 1]')
parser.add_argument('--max_epochs', default='20000', type=int, help='max epochs')


args = parser.parse_args()


# Get Data from Pickle:
with open(os.path.join('conv_decoder_sim', 'session_data.pkl'), 'rb') as input:
    data = pickle.load(input)

print('Training Dataset size: ', data.m)
print('Test Dataset size: ', data.test_m)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: ', device)


net = Conv_decoder()
net = net.to(device)

# Training function:
def train_net(max_epochs=50000, batch_size=200):
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.95)
    criterion = nn.MSELoss()

    best_test_perf = np.Inf
    epoch_train_loss = []
    epoch_test_loss = []
    for epoch in range(max_epochs):
        x, y = shuffle_unison(data.X_train, data.Y_train)
        net.train()
        batch_train_loss = 0
        for i in range(int(data.m / batch_size)):
            tensor_x = torch.from_numpy(np.float32(x[i * batch_size:(i + 1) * batch_size]))
            tensor_y = torch.from_numpy(np.float32(y[i * batch_size:(i + 1) * batch_size]))
            input_x = Variable(tensor_x)
            target_y = Variable(tensor_y, requires_grad=False)
            input_x, target_y = input_x.to(device), target_y.to(device)
            optimizer.zero_grad()  # zero the gradient buffers
            output_y = net(input_x)
            loss = criterion(output_y, target_y)
            loss.backward()
            optimizer.step()  # Does the update
            batch_train_loss += loss.item()
        scheduler.step()
        epoch_train_loss.append(batch_train_loss / batch_size)
        # Evaluate test set performance
        test_loss = eval_test_set()
        epoch_test_loss.append(test_loss)
        if test_loss < best_test_perf:
            if (args.verbose):
                print("-->Saving Checkpoint at epoch ", epoch)
                print("Loss over test set:", test_loss)
            state = {
                'net': net.state_dict(),
                'test_error': test_loss,
                'epoch': epoch,
            }
            if not os.path.isdir('Checkpoint'):
                os.mkdir('Checkpoint')
            torch.save(state, 'Checkpoint/checkpoint_state')
            best_test_perf = test_loss
        if args.verbose and epoch%100==0:
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, batch_train_loss / int(data.m / batch_size)))
    return epoch_train_loss, epoch_test_loss

def eval_test_set():
    net.eval()
    with torch.no_grad():
        x_test_variable = Variable(torch.from_numpy(np.float32(data.X_test)))
        y_test_variable = Variable(torch.from_numpy(np.float32(data.Y_test)), requires_grad=False)
        x_test_variable, y_test_variable = x_test_variable.to(device), y_test_variable.to(device)

        output_variables = net(x_test_variable)
        criterion = nn.MSELoss()
        loss = criterion(output_variables, y_test_variable)
        test_loss = loss.item()
        return test_loss


epoch_train_loss, epoch_test_loss = train_net(args.max_epochs)

# Save the network after training has been finished
torch.save(net.state_dict(), 'Checkpoint/net_end_of_training')


# Plot the training and test set losses after the training has been completed:
plt.figure(1)
x = np.arange(1, args.max_epochs + 1, 1)
plt.subplot(211)
plt.title("Epoch training loss")
plt.plot(x, epoch_train_loss)
plt.subplot(212)
plt.title("Epoch test loss")
plt.plot(x, epoch_test_loss)
plt.tight_layout()
plt.show()

# Since the ros versions use python 2.7 and cuda isn't install there, we store the cpu versions for later
if device=='cuda': 
    print('==> End of training network to CPU..')
    net = net.to('cpu')
    torch.save(net.state_dict(), 'Checkpoint/net_end_of_training_cpu')

    print('==> Checkpoint to CPU..')
    net = net.to(device)
    checkpoint = torch.load('Checkpoint/checkpoint_state')
    net.load_state_dict(checkpoint['net'])
    # lowest_test_error = checkpoint['test_error']
    print('Best epoch:', checkpoint['epoch'])
    net = net.to('cpu')
    torch.save(net.state_dict(), 'Checkpoint/checkpoint_cpu')

