from __future__ import division

import numpy as np
import os
import cv2
import torch

import random
import argparse
import sys 
from collections import namedtuple
import matplotlib.pyplot as plt

from pixelAI import PixelAI


actors = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll']
larm_min=[  -2.0857, -0.3142, -2.0857, -1.5446]
larm_max=[  2.0857,  1.3265, 2.0857, -0.0349]
env_opts = ['sim', 'real']

def ganged_plot(imgs, timesteps, save_opt):
    """
    ==========================
    Creating adjacent subplots
    ==========================

    """
    imgs_comb = np.hstack(imgs)
    fig, ax = plt.subplots()
    ax.imshow(imgs_comb, cmap="gray", vmin=0, vmax=255)
    ax.set_xticks(imgs_comb.shape[1]/len(timesteps)*np.arange(0.5,len(timesteps),1))
    ax.set_xticklabels([str(i) for i in timesteps])
    ax.set_xlabel(r'timestep')
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.get_yaxis().set_visible(False)

    for i in range(len(timesteps)):
        ax.axvline(x=imgs_comb.shape[1]/len(timesteps)*(i), color='black', linewidth=0.5)

    plt.tight_layout()
    if save_opt:
        plt.savefig("internal_traj.pdf", bbox_inches='tight')
    plt.show()

def check_area(im_check):
    # return percentage of area covered by the robot's arm in the input image:
    ret, thresh = cv2.threshold(im_check, 127, 255, cv2.THRESH_BINARY_INV)
    im_size = thresh.shape
    if sys.version_info[0]<3:       #Python 2.x version
        _, contour, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:                           #Python 3.x version (Not supported by ROS)
        contour = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contour[0]
    if len(contour)>0:
        largest_blob = max(contour, key=cv2.contourArea)
        return cv2.contourArea(largest_blob)/np.prod(im_size)
    else:
        return 0

def format_im_cv(im_np):
    return np.squeeze(np.squeeze((im_np*255).astype(np.uint8)))

def perturb(range):
    sign=np.random.randint(2, size=(len(actors)))
    delta=((range[0]-range[1])*np.random.rand(len(actors))+range[1])*(-1)**sign
    return delta
    
class Central:
    def __init__(self, env):
        self.env = env #Mode 0 for simulation, 1 for real NAO.

        self.level = None
        self.test = None
        self.core_id = None

        #Initialize the model path based on environment type (env):
        self.cores_path = 'Benchmark_core_' + env_opts[self.env]

        # Instantiate member of PixelAI class:
        self.pixelAI = PixelAI(self.env)

        # Parameters for perceptual/active inference
        self.Params = namedtuple('Params', ['dt', 'beta', 'sigma_v', 'sv_mu_gain', 'sigma_mu', 'sigma_v_gamma'])

        # Read the distribution parameters!
        self.clip_range = np.genfromtxt(os.path.join('Data_distribution', 'data_range_org.csv'), delimiter=",")
        self.mean_data = np.genfromtxt(os.path.join('Data_distribution', 'mean.csv'), delimiter=",")
        self.sigma_data = np.genfromtxt(os.path.join('Data_distribution', 'sigma.csv'), delimiter=",")

        self.max_iter = 5000
        
        self.int_traj_im = []
        self.int_traj_iter = []
        # self.int_traj_step = namedtuple('Step', ['iter', 'im'])


    def reset_belief(self, level_id, core_id, test_id, start_mu):
        self.level = level_id
        self.test = test_id
        self.core_id = core_id

        if self.level == 1:
            params_tuple = self.Params(0.1, None , 0.2 * 1e5 , 1 , None , None )
        elif self.level == 2:
            params_tuple = self.Params(0.1, None , 0.2 * 1e5 , 1 , None , None )
        else:
            params_tuple = self.Params(0.1, None , 0.2 * 1e4 , 1 , None , 10)


        self.pixelAI.reset(params_tuple, start_mu)

        im = cv2.imread(os.path.join(self.cores_path, ('core' +str(self.core_id)+'.png')))
        self.pixelAI.set_visual_sensation(im)


    def generate_mu(self, truePos, level):
        if level == 1:
            print('Generating for level 1!')
            range_angle = np.array([10.0, 5.0]) * np.pi / 180
            delta = perturb(range_angle)
            chosen_actor = np.random.randint(4, size=(1))
            sampled_mu = np.squeeze(np.copy(truePos))
            print('Chosen actor', chosen_actor[0])
            if (np.sign(delta[chosen_actor[0]]) == 1 and larm_max[chosen_actor[0]] > (
                sampled_mu[chosen_actor[0]] + delta[chosen_actor[0]])) or (
                    np.sign(delta[chosen_actor[0]]) == -1 and larm_min[chosen_actor[0]] < (
                sampled_mu[chosen_actor[0]] + delta[chosen_actor[0]])):
                sampled_mu[chosen_actor[0]] = sampled_mu[chosen_actor[0]] + delta[chosen_actor[0]]
            else:
                sampled_mu[chosen_actor[0]] = sampled_mu[chosen_actor[0]] - delta[chosen_actor[0]]
        elif level == 2:
            range_angle = np.array([10.0, 5.0]) * np.pi / 180
            delta = perturb(range_angle)
            print('Generating for level 2!')
            sampled_mu = np.squeeze(np.copy(truePos))
            for i in range(0, delta.shape[0]):
                if (np.sign(delta[i]) == 1 and (sampled_mu[i] + delta[i]) > larm_max[i]) or (
                        np.sign(delta[i]) == -1 and (sampled_mu[i] + delta[i]) < larm_min[i]):
                    sampled_mu[i] = sampled_mu[i] - delta[i]
                    print('Old: ', truePos[0, i] + delta[i], 'New', truePos[0, i] - delta[i], 'limits [',
                          larm_min[i], larm_max[i], ',]')
                else:
                    sampled_mu[i] = sampled_mu[i] + delta[i]
        else:
            print('Generating for level 3!')
            sampled_mu = np.random.multivariate_normal(self.mean_data, self.sigma_data, 1)
            # Clip the sample:
            sampled_mu = np.clip(sampled_mu, larm_min, larm_max)
        return sampled_mu

    def perceptual_inference_run(self):
        im_true = format_im_cv(self.pixelAI.s_v)
        for i in range(self.max_iter):
            self.pixelAI.iter()
            im_belief = format_im_cv(self.pixelAI.g_mu)
            if i == 0:
                if check_area(im_belief)<0.04:
                    return 0
            if i%50==0 and i<460:
                # self.int_traj.append(self.int_traj_step(i, cv2.addWeighted(im_belief, 0.8, im_true, 0.2, 0.0)))
                self.int_traj_im.append(cv2.addWeighted(im_belief, 0.8, im_true, 0.2, 0.0))
                self.int_traj_iter.append(i)
        # Save the last pic:
        _, g_mu = self.pixelAI.visual_forward()
        # self.int_traj.append(self.int_traj_step(i, cv2.addWeighted(format_im_cv(g_mu.data.numpy()), 0.8, im_true, 0.2, 0.0)))
        self.int_traj_im.append(cv2.addWeighted(format_im_cv(g_mu.data.numpy()), 0.8, im_true, 0.2, 0.0))
        self.int_traj_iter.append(self.max_iter)
        return 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Perceptual inference test')
    parser.add_argument('--env', default=0, type=int, help='env = [0: simulation, 1: real NAO]')
    parser.add_argument('--level', default=2, type=int, help='levels = [1, 2, 3]')
    parser.add_argument('--core', default=0, type=int, help='core = [0..19 or 49]')
    parser.add_argument('--test', default=0, type=int, help='core = [0..9 or -1 for new random]')
    parser.add_argument('--save', default=0, type=int, help='save = [0, 1]')

    args = parser.parse_args()
    env = args.env
    level = args.level
    core = args.core
    test = args.test

    if env == 0:
        core_angles = np.genfromtxt('Benchmark_core_sim/core_angles.csv', delimiter=",")
    else:
        core_angles = np.genfromtxt('Benchmark_core_real/core_read_angles.csv', delimiter=",")

    num_cores = core_angles.shape[0] # 20 for the real robot and 50 for the simulation
    num_tests = 10

    central = Central(env)

    truePos = np.reshape(core_angles[core,:], (1, 4))
    if test == -1:
        mu = central.generate_mu(truePos, level)
    else:
        start_mu=np.genfromtxt('Benchmark_mu'+str(level)+'/mu_'+str(core)+'.csv', delimiter=",")
        mu = start_mu[test,:]
    central.reset_belief(level, core, test, mu)
    while True:
        success = central.perceptual_inference_run()
        if success:
            print('Level', level, 'Core ', core, ' test ', test, ' completed!')
            break
    ganged_plot(central.int_traj_im, central.int_traj_iter, args.save)


