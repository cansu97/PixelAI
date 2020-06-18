import os
import argparse

import numpy as np
import random 

import torch
import torch.nn as nn
from torch.autograd import Variable

import pickle
import matplotlib.pyplot as plt
from conv_decoder import *

import sys
sys.path.append('..')

from process_data import *

def eval_data_perf(net, input_data, target_data):
    net.eval()
    with torch.no_grad():
        x_test_variable = Variable(torch.from_numpy(np.float32(input_data)))
        y_test_variable = Variable(torch.from_numpy(np.float32(target_data)), requires_grad=False)
        x_test_variable, y_test_variable = x_test_variable.to(device), y_test_variable.to(device)

        output_variables = net(x_test_variable)
        criterion = nn.MSELoss()
        loss = criterion(output_variables, y_test_variable)
        test_loss = loss.item()
    return test_loss, output_variables.to('cpu')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Model evaluation')
    parser.add_argument('--eval_data', default='train_set', type=str, help='opt = [train_set, test_set, rand_data]')
    parser.add_argument('--vis', default='1', type=int, help='verbose = [0, 1]')

    args = parser.parse_args()

    # Get Data from Pickle:
    with open(os.path.join('conv_decoder_sim', 'session_data.pkl'), 'rb') as input:
        data = pickle.load(input)
    print('Training Dataset size: ', data.m)
    print('Test Dataset size: ', data.test_m)



    net = Conv_decoder()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device', device)
    net = net.to(device)

    net.load_state_dict(torch.load('conv_decoder_sim/trained_model'))
    
    # For the cuda version:
    # checkpoint = torch.load('Checkpoint/checkpoint_state')
    # net.load_state_dict(checkpoint['net'])

    if args.eval_data == 'train_set':
        eval_data_input = data.X_train
        eval_data_target = data.Y_train
        print('---> Evaluating data on the training set...')
    elif args.eval_data == 'test_set':
        eval_data_input = data.X_test
        eval_data_target = data.Y_test
        print('---> Evaluating data on the test set...')


    loss_val, output_variables = eval_data_perf(net, eval_data_input, eval_data_target)
    print('====>Loss over the data: {:.4f}'.format(loss_val))

    if args.vis:
        print('---> Visualizing 20 network outputs (randomly sampled from the evaluation data)...')
        output_images = output_variables.data.numpy()
        rand_ind = random.sample(range(eval_data_input.shape[0]), 20)
        for i in rand_ind:
            im_true = np.squeeze(np.squeeze(eval_data_target[i,0,:,:]))
            im_pred = np.squeeze(np.squeeze(output_images[i, 0, :, :]))

            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(im_true, cmap='gray', vmin=0, vmax=1)
            axarr[0].set_title('True image')
            axarr[0].axis('off')

            axarr[1].imshow(im_pred, cmap='gray', vmin=0, vmax=1)
            axarr[1].set_title('Predicted image')
            axarr[1].axis('off')
            plt.show()


