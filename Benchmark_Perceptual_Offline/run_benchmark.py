from __future__ import division

import numpy as np
import os
import cv2
import torch

import random
import argparse
import sys 
from collections import namedtuple

from pixelAI import PixelAI

actors = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll']
larm_min=[  -2.0857, -0.3142, -2.0857, -1.5446]
larm_max=[  2.0857,  1.3265, 2.0857, -0.0349]
env_opts = ['sim', 'real']

def perturb(range):
    sign=np.random.randint(2, size=(len(actors)))
    delta=((range[0]-range[1])*np.random.rand(len(actors))+range[1])*(-1)**sign
    return delta

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

class Central:
    def __init__(self, env, log, vis):
        self.env = env #Mode 0 for simulation, 1 for real NAO.
        self.log = log
        self.vis = vis

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

        if self.log:
            # Create the directories for the log files if they aren't instantiated yet:
            self.res_path = 'Benchmark_res_' + env_opts[self.env]
            if not os.path.isdir(self.res_path):
                os.mkdir(self.res_path)

            self.level_path = self.res_path + '/Perceptual_Level' + str(self.level)
            if not os.path.isdir(self.level_path):
                os.mkdir(self.level_path)

            self.core_path = self.level_path + '/Core' + str(self.core_id)
            if not os.path.isdir(self.core_path):
                os.mkdir(self.core_path)

            # Create the directories for the images if they aren't instantiated yet:
            self.level_path_pics = self.res_path + '/Perceptual_End_results_' + str(self.level)
            if not os.path.isdir(self.level_path_pics):
                os.mkdir(self.level_path_pics)

            self.path_pics = self.level_path_pics + '/Core' + str(self.core_id)
            if not os.path.isdir(self.path_pics):
                os.mkdir(self.path_pics)

            # For the log-files:
            self.store_mu = self.pixelAI.mu
            if hasattr(self, 'store_vis_err') == True:
                delattr(self, 'store_vis_err')


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
            # if sampled_mu[chosen_actor[0]] > larm_max[chosen_actor[0]] or sampled_mu[chosen_actor[0]] < larm_min[
            #     chosen_actor[0]]:
            #     print('ERROR!!!')
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
                if sampled_mu[i] > larm_max[i] or sampled_mu[i] < larm_min[i]:
                    print('ERROR!!!')
        else:
            print('Generating for level 3!', self.level)
            sampled_mu = np.random.multivariate_normal(self.mean_data, self.sigma_data, 1)
            # Clip the sample:
            sampled_mu = np.clip(sampled_mu, larm_min, larm_max)
            # sampled_mu = np.clip(sampled_mu, self.clip_range[1, :], self.clip_range[0, :])
        return sampled_mu


    def store_iter_local(self):
        self.store_mu = np.append(self.store_mu, self.pixelAI.mu, axis=0)
        if hasattr(self, 'store_vis_err') == False:
            self.store_vis_err = np.square(self.pixelAI.pred_error).mean()
        else:
            self.store_vis_err = np.append(self.store_vis_err, np.square(self.pixelAI.pred_error).mean())

    def log_test(self):
        # Store the trajectories in csv files:
        np.savetxt(self.core_path+"/mu_log"+str(self.test)+".csv", self.store_mu, delimiter=",")
        np.savetxt(self.core_path+"/vis_err_log"+str(self.test)+".csv", self.store_vis_err, delimiter=",")

    def perceptual_inference_run(self):
        im_true = format_im_cv(self.pixelAI.s_v)
        for i in range(self.max_iter):
            self.pixelAI.iter()
            im_belief = format_im_cv(self.pixelAI.g_mu)
            if i == 0:
                if check_area(im_belief)<0.025:
                    return 0
            if self.vis:
                cv2.imshow('Results', cv2.addWeighted(im_belief, 0.8, im_true, 0.2, 0.0))
                cv2.waitKey(3)
            if self.log:
                self.store_iter_local()
                if i == 0:
                    cv2.imwrite(os.path.join(self.path_pics, 'beginning' +str(self.test)+'.png'),cv2.addWeighted(im_belief, 0.8,im_true,0.2,0.0))
                elif i == (self.max_iter-1):
                    self.log_test()
                    # Save the last pic:
                    _, g_mu = self.pixelAI.visual_forward()
                    cv2.imwrite(os.path.join(self.path_pics, 'end_result' +str(self.test)+'.png'),cv2.addWeighted(format_im_cv(g_mu.data.numpy()), 0.8,im_true,0.2,0.0))
        return 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Perceptual inference test')
    parser.add_argument('--level', default=2, type=int, help='levels = [1, 2, 3]')
    parser.add_argument('--env', default=0, type=int, help='env = [0: simulation, 1: real NAO]')
    parser.add_argument('--mode', default=1, type=int, help='mode = [0: generate_new, 1: read_from_generated]')
    parser.add_argument('--log', default=0, type=int, help='log = [0: do not log 1: save logs and images]')
    parser.add_argument('--vis', default=1, type=int, help='vis = [0: no visualization, 1: show internal belief]')
    # parser.add_argument('--int_traj', default=0, type=int, help='int_traj = [0: no images saved from intermediate steps'
    #                                                             ', 1: store images of internal trajectory]')

    args = parser.parse_args()
    level = args.level
    env = args.env
    mode = args.mode
    log = args.log
    vis = args.vis

    if env == 0:
        core_angles = np.genfromtxt('Benchmark_core_sim/core_angles.csv', delimiter=",")
    else:
        core_angles = np.genfromtxt('Benchmark_core_real/core_read_angles.csv', delimiter=",")

    num_cores = core_angles.shape[0] # 20 for the real robot and 50 for the simulation
    num_tests = 10

    central = Central(env, log, vis)

    for c in range(num_cores):
        truePos = np.reshape(core_angles[c,:], (1, 4))
        if mode:
            start_mu=np.genfromtxt('Benchmark_mu'+str(level)+'/mu_'+str(c)+'.csv', delimiter=",")
        elif mode == 0 and level == 3:
            selector=[x for x in range(core_angles.shape[0]) if x!=c]
            cores_excluded=core_angles[selector,:]
            p = np.random.permutation(cores_excluded.shape[0])
            cores_excluded=cores_excluded[p]
            start_mu=cores_excluded[:num_tests,:]
        for i in range(num_tests):
            if mode or level==3:
                # Get sample from log files or for level 3 newly generated:
                mu = start_mu[i,:]
                print('Start_mu', mu)
            else:
                # Generate sample:
                mu = central.generate_mu(truePos, level)
            central.reset_belief(level, c, i, mu)
            while True:
                success = central.perceptual_inference_run()
                if success:
                    print('Core ', c, ' test ', i, ' completed!')
                    break


