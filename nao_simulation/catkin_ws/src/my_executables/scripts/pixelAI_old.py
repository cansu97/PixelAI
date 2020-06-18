#!/usr/bin/env python

from __future__ import division

import numpy as np
from collections import namedtuple
import cv2
import os

import torch
from torch.autograd import Variable

#Import own functions:
from Conv_decoder_model import *
from process_data import process_image, minmax_normalization

# import random
# import sys
# import rospy

#Define the joint names and the safe sitting joint states for the safe sitting method:
larm_min=[  -2.0857, -0.3142, -2.0857, -1.5446]
larm_max=[  2.0857,  1.3265, 2.0857, -0.0349]

actors = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll']

class PixelAI:
    def __init__(self, env=0):
        # initialize class variables
        self.env = env #Mode 0 for simulation, 1 for real NAO.

        # attractor_im and s_v store images from the bottom camera which are processed using the function
        # process_image, the original image size is reduced to the values stored in the properties
        # width and height:
        self.width = 80
        self.height = 56

        # Visual attactor for active inference:
        self.attractor_im = None
        self.attractor_pos = None

        self.mu = np.empty((1, 4))
        self.sv = np.empty((1,1,self.height, self.width)) # Currently observed visual sensation
        self.g_mu = np.empty((1,1,self.height, self.width)) # Predicted visual sensation
        self.pred_error = None
        self.attr_error = None
        self.a = None
        self.a_thres = None

        # Visual forward model:
        self.model = 'Google_check'
        self.model_path = os.path.join('Conv_decoder_model', self.model)
        # Load network for the visual forward model
        self.load_model()

        # Will be initialized inside the reset_pixelAI function
        self.dt = None  
        self.beta = None
        self.sigma_v = None
        self.sv_mu_gain = None
        self.sigma_mu = None
        self.sigma_v_gamma = None

        self.active_inference_mode = 0    # 1 If active inference (actions with attractor dynamics), if 0: perceptual inference
        self.adapt_sigma = 0

    def reset(self, params, start_mu, attractor_pos = None):
        # torch.manual_seed(0)
        # np.random.seed(0)
        self.mu = np.reshape(start_mu, (1,4))
        self.sv = np.empty((1,1,self.height, self.width))
        self.g_mu = np.empty((1,1,self.height, self.width))

        #Read the parameters:
        (self.dt, self.beta, self.sigma_v, self.sv_mu_gain, self.sigma_mu, self.sigma_v_gamma) = params

        if np.all(attractor_pos) != None:
            self.active_inference_mode = 1 #Active inference activated
            self.attractor_pos = np.reshape(attractor_pos, (1,4))
            self.attractor_im = np.empty((1,1,self.height, self.width))
            self.a = np.zeros((1, 4))
            self.a_thres = 2.0 * (np.pi / 180) / self.dt # Maximum 2 degrees change per timestep
        else: add perceptual inference
            if self.sigma_v_gamma != None:
                self.adapt_sigma = 1

    def set_attractor_im(self, attractor_im):
        camera_im = process_image(attractor_im, self.env)
        self.attractor_im = camera_im.reshape([1,1,camera_im.shape[0],camera_im.shape[1]])

    def set_visual_sensation(self, cv_image):
        camera_im = process_image(cv_image, self.env)
        self.sv = camera_im.reshape([1, 1, camera_im.shape[0], camera_im.shape[1]])

    def load_model(self):
        print("Loading Deconvolutional Network...")
        self.network = Conv_decoder()
        self.network.load_state_dict(torch.load(os.path.join(self.model_path,'checkpoint_cpu_version')))
        self.network.eval()

        #Load data range used for training for minmax-normalization:
        data_range = np.genfromtxt(os.path.join(self.model_path,"data_range_sim.csv"), delimiter=",")
        self.data_max = data_range[0,:]
        self.data_min = data_range[1,:]

    def visual_forward(self):
        input = torch.FloatTensor(minmax_normalization(self.mu, self.data_max, self.data_min))
        input = Variable(input, requires_grad=True)
        # prediction
        out = self.network.forward(input)
        return input, out 

    def get_dF_dmu_vis(self, input, out):
        neg_dF_dg = (1/self.sigma_v) * self.pred_error
        # Set the gradient to zero before the backward pass to make sure there is no accumulation from previous backward passes
        input.grad = torch.zeros(input.size())
        out.backward(torch.Tensor(neg_dF_dg),retain_graph=True)
        return input.grad.data.numpy() # dF_dmu_vis

    def get_dF_dmu_dyn(self, input, out):
        self.attr_error = self.attractor_im - self.g_mu
        A = self.beta * self.attr_error
        # Set the gradient to zero before the backward pass to make sure there is no accumulation from previous backward passes
        input.grad = torch.zeros(input.size())
        out.backward(torch.Tensor(A*(1/self.sigma_mu)),retain_graph=True)
        return input.grad.data.numpy() # mu_dot_dyn (1x4 vector)

    def get_dF_da_visual(self, dF_dmu_vis):
        # dF_dsv = (1 / self.sigma_v) * self.pred_error (= neg_dF_dg)
        return (-1) * dF_dmu_vis * self.dt

    def iter(self):
        input, out = self.visual_forward()
        self.g_mu = out.data.numpy()
        self.pred_error = self.sv - self.g_mu

        # dF/dmu using visual information:
        dF_dmu_vis = self.get_dF_dmu_vis(input, out)
        mu_dot_vis = self.sv_mu_gain * dF_dmu_vis    # mu_dot_vis is a 1x4 vector

        if self.active_inference_mode: 
            # dF/dmu with attractor:
            mu_dot = mu_dot_vis + self.get_dF_dmu_dyn(input, out)
        else:
            mu_dot = mu_dot_vis
        
        # Compute the action:
        a_dot = self.get_dF_da_visual(dF_dmu_vis)

        # Update mu:
        self.mu = self.mu + self.dt * mu_dot

        if self.active_inference_mode: 
            # Update a:
            self.a = self.a + self.dt * a_dot

        # Clip the action value in the case of action saturation:
        self.a = np.clip(self.a, -self.a_thres, self.a_thres)

        if self.adapt_sigma and np.square(self.pred_error).mean() <= 0.01:
            self.sigma_v = self.sigma_v * self.sigma_v_gamma
            self.adapt_sigma = 0 # Increase Sigma only once!
        




