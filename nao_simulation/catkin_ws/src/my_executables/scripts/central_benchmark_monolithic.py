#!/usr/bin/env python

from __future__ import division

import sys
import rospy
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetJointProperties, GetJointPropertiesResponse

from sensor_msgs.msg import Image,JointState
from std_msgs.msg import Float64
from cv_bridge import CvBridge, CvBridgeError
import cv2
import random
import csv

import os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

#Import own functions:
from Conv_decoder_model import *
from process_data_sim import process_image, minmax_normalization


#Define the joint names and the safe sitting joint states for the safe sitting method:
larm_min=[  -2.0857, -0.3142, -2.0857, -1.5446]
larm_max=[  2.0857,  1.3265, 2.0857, -0.0349]

actors = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll']
actors_ind=[5,6,3,2]
class Central:
    def __init__(self):
        # initialize class variables
        self.joint_names = []
        self.joint_angles = []
        self.joint_velocities = []
        self.jointPub = 0
        
        self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)


        #Position for the fixed head:
        self.headyaw=0.27
        self.headpitch=0.224

        joints = [
    
    # Head
    {
        "joint_name": "HeadYaw",
        "publisher": None,
        "limit_min": -2.08567,
        "limit_max": 2.08567
    },
    {
        "joint_name": "HeadPitch",
        "publisher": None,
        "limit_min": -0.671952,
        "limit_max": 0.514872
    },
    {
        "joint_name": "LShoulderPitch",
        "publisher": None,
        "limit_min": -2.08567,
        "limit_max": 2.08567
    },
    {
        "joint_name": "LShoulderRoll",
        "publisher": None,
        "limit_min": -0.314159,
        "limit_max": 1.32645
    },
    {
        "joint_name": "LElbowYaw",
        "publisherg": None,
        "limit_min": -2.08567,
        "limit_max": 2.08567
    },
    {
        "joint_name": "LElbowRoll",
        "publisher": None,
        "limit_min": -1.54462,
        "limit_max": -0.0349066
    },
    ]

    # Initialize joints (currently only head and left arm joints)
        topics = []
        for joint in joints:
            topic_name = '/nao_dcm/' + joint['joint_name'] + '_position_controller/command';
            topics.append(topic_name);
            joint['publisher'] = rospy.Publisher(topic_name, Float64, queue_size=5)

        self.joints = joints
        

        #Necessary for Benchmarking (Accesing/Storing logs etc.):
        self.level = 1
        self.num_cores = 50
        self.num_test = 10

        self.level_path='Level'+str(self.level)
        self.log_path='Benchmark_mu'+str(self.level)

        self.cores=np.genfromtxt('Benchmark_core_sim/core_angles.csv', delimiter=",")

        #----------Active inference: -----
        self.iteration = 0
        self.max_iteration = 1500
        # Initialize active inference internal parameters:
        if self.level == 1:
            self.beta=2*1e-5 # 1/self.sigma_v
            self.sigma_v=0.6*1e3
            self.sv_mu_gain=1e-3
        elif self.level == 2:
            self.beta=5*1e-5 # 1/self.sigma_v
            self.sigma_v=0.2*1e3
            self.sv_mu_gain=1e-3
        else:
            self.beta = 5*1e-4 # 1/self.sigma_v
            self.sigma_v = 0.2*1e2
            self.sv_mu_gain=1e-3

        self.sigma_mu=1 # The gain parameter beta of the attractor will be used for tuning
        self.sp_active=0 # Main focus is on the visual feedback, but the proprioceptive info can also be integrated with a larger sigma -> small updates
        self.sigma_p=1e5
        self.dt = 0.065 # Determined by timing the one loop of active inference
        self.a_thres=2.0*(np.pi/180)/self.dt

        # attractor_im and s_v store images from the bottom camera which are processed using the function
        #process_image, the original image size is reduced to the values stored in the properties 
        #width and height:
        self.width = 80
        self.height = 56
        self.crop = True
        # Zero initialization for the visual attactor, observed and predicted visual sensations and action:
        self.attractor_im = np.zeros((1,1,self.height, self.width))
        self.sv = np.zeros((1,1,self.height, self.width))

        self.g_mu = np.zeros((1,1,self.height, self.width)) #Predicted visual sensation:
        self.a = np.zeros((1,4))
        
        

        # Visual forward model:
        self.model='conv_decoder_sim'
        self.model_path = os.path.join('Conv_decoder_model', self.model)
        # Load network for the visual forward model
        self.load_model()

        #Load data range used for training for minmax-normalization:
        data_range = np.genfromtxt(os.path.join(self.model_path,"data_range_sim.csv"), delimiter=",")
        self.data_max = data_range[0,:]
        self.data_min = data_range[1,:]

        # Storage buffers:
        # self.store_mu=self.mu
        # self.store_q
        # self.store_goal_err -> Oberserved camera image vs. Attractor image
        # self.store_attractor_err -> Internal predicted sensation vs. Attractor image
        # self.store_vis_err -> Internal predicted sensation vs. Observed camera image

        #Flags to synchronize sensory input:
        self.get_s_p = 0
        self.get_im = 0

        
        #Last target stored to monitor target q for the controller vs. arrived position
        self.last_target=[] #will be set in reset_robot

        
        pass
    
    def reset_robot(self, core_id, test_id):

        # torch.manual_seed(0)
        # np.random.seed(0)

        self.core_id=core_id
        self.test=test_id

        self.attractor_pos=np.reshape(self.cores[core_id,:], (1,4))
        mu_buff=np.genfromtxt(self.log_path+"/mu_"+str(core_id)+".csv", delimiter=",")
        self.mu=np.reshape(mu_buff[test_id,:], (1,4))

        self.sv = np.zeros((1,1,self.height, self.width))
        self.g_mu = np.zeros((1,1,self.height, self.width))
        self.attractor_im = np.zeros((1,1,self.height, self.width))
        self.a=np.zeros((1,4))

        #Resetting flags etc.
        self.sp_active=0
        self.get_s_p=0
        self.get_im=0

        self.last_target=np.squeeze(self.mu)

        self.iteration=0

        #For the log-files:
        self.store_mu=self.mu

        if hasattr(self, 'store_q') == True:
            delattr(self, 'store_q')

        if hasattr(self, 'store_vis_err') == True:
            delattr(self, 'store_vis_err')
        
        if hasattr(self, 'store_attractor_err') == True:
            delattr(self, 'store_attractor_err')

        if hasattr(self, 'store_goal_err') == True:
            delattr(self, 'store_goal_err')

        #Create the directories for the log files if they aren't instantiated yet:
        self.level_path='Level'+str(self.level)
        self.core_path=self.level_path+'/Core'+str(self.core_id)
        
        if not os.path.isdir(self.level_path):
            os.mkdir(self.level_path)
        
        if not os.path.isdir(self.core_path):
            os.mkdir(self.core_path)
    
        self.level_path_pics='End_results_'+ str(self.level)
        if not os.path.isdir(self.level_path_pics):
            os.mkdir(self.level_path_pics)
        
        self.path_pics=self.level_path_pics+'/Core'+str(self.core_id)
        if not os.path.isdir(self.path_pics):
            os.mkdir(self.path_pics)

    def load_model(self):
        print("Loading Deconvolutional Network...")
        self.network = Conv_decoder()
        self.network.load_state_dict(torch.load(os.path.join(self.model_path,'trained_model')))
        self.network.eval()

    def proprioceptive_cb(self,data):
        #rospy.loginfo("joint states )
        self.joint_names = data.name
        self.joint_angles = data.position
        self.joint_velocities = data.velocity

        if self.get_s_p==1:
            print('Proprioceptive callback activated')
            self.s_p=[self.joint_angles[x] for x in actors_ind]
            self.s_p=np.reshape(np.array(self.s_p), (1,4))
            if hasattr(self, 'store_q') == False:
                self.store_q=self.s_p
            self.get_s_p=0
            print('Difference between last target and  current reading (in degrees):', (self.s_p-self.last_target)*180/np.pi)
            if np.any(abs((self.s_p-self.last_target)*180/np.pi)>1):
                print("\n")
                print("\n")
                print('Larger sensor ERROR STOPPP')
                print("\n")
                print("\n")

            print(self.s_p)

    def image_cb(self, data):
        #current_im_cb = rospy.get_time()
        #print('Time passed since last image callback: ', current_im_cb-self.last_im_cb)
        #print('image callback rate: ', 1/(current_im_cb-self.last_im_cb))
        #self.last_im_cb = current_im_cb

        bridge_instance = CvBridge()
        try:
            cv_image = bridge_instance.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
        
        if self.get_im==1:
            self.cv_image=cv_image
            self.get_im=0

        if hasattr(self, 'attractor_im_org') == True:
            alpha=0.25
            cv2.imshow("Image Observations combined",cv2.addWeighted(self.attractor_im_org, alpha,cv_image,1-alpha,0.0))
            #cv2.imwrite(os.path.join('Camera','comb' +str(self.iteration)+'.png'),cv_image)
            cv2.waitKey(3) # a small wait time is needed for the image to be displayed correctly

    def set_head_angles(self,head_angle1,head_angle2): ## controls head movement
        self.joints[0]['publisher'].publish(head_angle1)
        self.joints[1]['publisher'].publish(head_angle2)

    def set_arm_angles(self,larm_angles): ## controls left arm movement
        self.joints[2]['publisher'].publish(larm_angles[0])
        self.joints[3]['publisher'].publish(larm_angles[1])
        self.joints[4]['publisher'].publish(larm_angles[2])
        self.joints[5]['publisher'].publish(larm_angles[3])
    
    def set_angles(self,joint_num, angle_to_set): ## controls head movement
        self.joints[joint_num]['publisher'].publish(angle_to_set)

    def ai(self):
        start = rospy.get_rostime()

        input, out=self.visual_forward()
        self.g_mu=out.data.numpy()
        self.pred_error=(self.sv - self.g_mu)

        #dF/dmu using visual information:
        mu_dot=self.dF_dmu_dyn(input, out)
        
        #dF/dmu with attractor:
        mu_dot=mu_dot + self.dF_dmu_visual(input, out)
        
        # if self.sp_active==1:
        #     mu_dot=mu_dot+(1/self.sigma_p)*(self.s_p-self.mu)

        #Compute the action:
        a_dot=self.dF_da_visual(input, out)

        # if self.sp_active==1:
        #     a_dot=a_dot-(1/self.sigma_p)*(self.s_p-self.mu)

        #Update mu:
        self.mu = self.mu + self.dt * mu_dot
        #Update a:
        self.a = self.a + self.dt * a_dot

        end=rospy.get_rostime()
        
        print('AI loop duration: ', start.nsecs-end.nsecs)


    def move(self):
        q=self.s_p
        diff_thres=[0.002, 0.002, 0.002, 0.002] #0.11 degrees threshold

        if np.any(abs(self.a)>self.a_thres):
            print('\n')
            print('ACTION SATURATION', np.sum(abs(self.a)>self.a_thres))
            print('\n')

        for i in range(0, len(actors)):
            self.a[0,i]=min(max(self.a[0,i], -self.a_thres), self.a_thres)
            diff=self.a[0][i]*self.dt

            if abs(diff)>=diff_thres[i]:
                targ=q[0,i]+ diff
                if targ<larm_max[i] and targ>larm_min[i]:
                    self.set_angles(i+2, targ)
                    self.last_target[i]=targ 
                    print("Action has been carried out")
                else:
                    targ=np.minimum(larm_max[i],np.maximum(targ, larm_min[i]))
                    if abs(targ-q[0,i])>=diff_thres[i]:
                        self.set_angles(i+2, targ)
                        self.last_target[i]=targ 
                    print('Joints limit reached in:', actors[i])

        print('Last target', self.last_target)

    def visual_forward(self):
        input = torch.FloatTensor(minmax_normalization(self.mu, self.data_max, self.data_min))
        input = Variable(input, requires_grad=True)
        # prediction
        out = self.network.forward(input)
        return input, out 

    def dF_dmu_visual(self, input, out):
        
        dF_dg=(1/self.sigma_v) * self.pred_error

        #Set the gradient to zero before the backward pass to make sure there is no accumulation from previous backward passes
        input.grad=torch.zeros(input.size())
        out.backward(torch.Tensor(np.ones(out.shape)*dF_dg),retain_graph=True)
        #print("mu_dot_vis:", input.grad) # this will give non-zero value

        mu_dot_vis=self.sv_mu_gain*input.grad.data.numpy() #This is a 1x4 vector
        return mu_dot_vis 

    def dF_dmu_dyn(self, input, out):
        self.attr_error=(self.attractor_im - self.g_mu)
        A=self.beta * self.attr_error

        #Set the gradient to zero before the backward pass to make sure there is no accumulation from previous backward passes
        input.grad=torch.zeros(input.size())
        out.backward(torch.Tensor(np.ones(out.shape)*A*(1/self.sigma_mu)),retain_graph=True)
        #print("mu_dot_dyn", input.grad) # this will give non-zero value

        mu_dot_dyn=input.grad.data.numpy() #This is a 1x4 vector
        return mu_dot_dyn 
    

    def dF_da_visual(self, input, out):
        
        dF_dsv=(1/self.sigma_v) * self.pred_error
        
        #Set the gradient to zero before the backward pass to make sure there is no accumulation from previous backward passes
        input.grad=torch.zeros(input.size())
        out.backward(torch.Tensor(np.ones(out.shape)*dF_dsv),retain_graph=True)

        a_dot_vis=input.grad.data.numpy()*self.dt*(-1) #This is a 1x4 vector
        return a_dot_vis 

    def store_iter_local(self):
        self.store_q=np.append(self.store_q, self.s_p, axis=0)
        self.store_mu=np.append(self.store_mu, self.mu, axis=0)

        if hasattr(self, 'store_vis_err') == False:
            self.store_vis_err=np.square(self.pred_error).mean()
        else:
            self.store_vis_err=np.append(self.store_vis_err, np.square(self.pred_error).mean())

        if hasattr(self, 'store_attractor_err') == False:
            self.store_attractor_err=np.array([np.square(self.attr_error).mean()])
        else:
            self.store_attractor_err=np.append(self.store_attractor_err, np.square(self.attr_error).mean())

        if hasattr(self, 'store_goal_err') == False:
            self.store_goal_err=np.array([np.square(self.attractor_im - self.sv).mean()])
        else:
            self.store_goal_err=np.append(self.store_goal_err, np.square(self.attractor_im - self.sv).mean())

    def log_images(self):
        im_belief=np.squeeze(np.squeeze((self.g_mu*255).astype(np.uint8)))
        im_real=np.squeeze(np.squeeze((self.sv*255).astype(np.uint8)))
        im_attractor=np.squeeze(np.squeeze((self.attractor_im*255).astype(np.uint8)))
        
        if self.iteration==0:
            cv2.imwrite(os.path.join(self.path_pics, 'beginning_observed' +str(self.test)+'.png'),cv2.addWeighted(self.attractor_im_org, 0.25,self.cv_image,0.75,0.0))
            #cv2.imwrite(os.path.join(self.path_pics, 'beginning_internal' +str(self.test)+'.png'),cv2.addWeighted(im_attractor, 0.25,im_belief,0.75,0.0))
            cv2.imwrite(os.path.join(self.path_pics, 'beginning_internal' +str(self.test)+'.png'), im_belief)
        
        else:
            cv2.imwrite(os.path.join(self.path_pics, 'end_observed' +str(self.test)+'.png'),cv2.addWeighted(self.attractor_im_org, 0.25,self.cv_image,0.75,0.0))
            #cv2.imwrite(os.path.join(self.path_pics, 'end_internal' +str(self.test)+'.png'),cv2.addWeighted(im_attractor, 0.25,im_belief,0.75,0.0))
            cv2.imwrite(os.path.join(self.path_pics, 'end_internal' +str(self.test)+'.png'),im_belief)

    def log_test(self):
        
        #Store the trajectories in csv files:
        np.savetxt(self.core_path+"/mu_log"+str(self.test)+".csv", self.store_mu, delimiter=",")
        np.savetxt(self.core_path+"/q_log"+str(self.test)+".csv", self.store_q, delimiter=",")
        np.savetxt(self.core_path+"/vis_err_log"+str(self.test)+".csv", self.store_vis_err, delimiter=",")
        np.savetxt(self.core_path+"/attractor_err_log"+str(self.test)+".csv", self.store_attractor_err, delimiter=",")
        np.savetxt(self.core_path+"/goal_err"+str(self.test)+".csv", self.store_goal_err, delimiter=",")


    def plot_progress(self):
        #Plot for the mu values:
        plt.figure(1)
        plt.subplot(221)
        plt.title('mu: LShoulderPitch')
        plt.plot(self.store_mu[:,0], label='mu')
        plt.plot(self.store_q[:,0], label='q')
        plt.axhline(y=self.attractor_pos[0,0], linewidth=3, color='r') 
        plt.legend()
       

        plt.subplot(222)
        plt.title('mu: LShoulderRoll')
        plt.plot(self.store_mu[:,1], label='mu')
        plt.plot(self.store_q[:,1], label='q')

        plt.axhline(y=self.attractor_pos[0,1], linewidth=3, color='r') 
        plt.legend()


        plt.subplot(223)
        plt.title('mu: LElbowYaw')
        plt.plot(self.store_mu[:,2], label='mu')
        plt.plot(self.store_q[:,2], label='q')
        plt.axhline(y=self.attractor_pos[0,2], linewidth=3, color='r') 
        plt.legend()


        plt.subplot(224)
        plt.title('mu: LElbowRoll')
        plt.plot(self.store_mu[:,3], label='mu')
        plt.plot(self.store_q[:,3], label='q')

        plt.axhline(y=self.attractor_pos[0,3], linewidth=3, color='r') 
        plt.legend()

        plt.tight_layout()

        #Plot the visual error:
        plt.figure(2)
        plt.title('Error in Internal Visual Prediction')
        plt.plot(self.store_attractor_err)
        plt.tight_layout()
        plt.show()

    def central_execute(self):
        rospy.init_node('central_node',anonymous=True) #initilizes node, sets name

        # create several topic subscribers
        rospy.Subscriber("/nao_dcm/joint_states",JointState,self.proprioceptive_cb)

        rospy.Subscriber('/nao_robot/camera_bottom/image_raw',Image, self.image_cb)
        srv_joint_state = rospy.ServiceProxy('/gazebo/get_joint_properties', GetJointProperties)

        rospy.Subscriber("joint_states",JointState,self.proprioceptive_cb)


        rospy.sleep(2.0)
        self.set_head_angles(self.headyaw, self.headpitch)
        time.sleep(1.0)
        
        
        for c in range(0, self.num_cores):
            for t in range(0, self.num_test):
                self.reset_robot(c,t)

                #Bring the robot to the attractor position to store attractor image:
                self.set_arm_angles(np.squeeze(self.attractor_pos)) 
                time.sleep(1.0)

                self.get_im=1
                while not rospy.is_shutdown():
                    if self.im_flag==1:
                        self.im_flag=0
                        break
                self.attractor_im_org=self.cv_image
                camera_im=process_image(cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY), self.crop)
                self.attractor_im=camera_im.reshape([1,1,camera_im.shape[0],camera_im.shape[1]])

                #Bring the robot arm to the mu position:
                self.set_arm_angles(np.squeeze(self.mu)) 
                print("Robot arm has been brought to the initial position")
                rospy.sleep(1.0)

                while  self.iteration<self.max_iteration and (not rospy.is_shutdown()):
                    current_time = rospy.get_time()
                    print('iteration', self.iteration)
                    self.get_s_p=1
                    self.get_im=1
                    while not rospy.is_shutdown():
                        if self.get_s_p==0 and self.get_im==0:
                            break
                    
                    camera_im=process_image(cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY), self.crop)
                    self.sv=camera_im.reshape([1,1,camera_im.shape[0],camera_im.shape[1]])
                    self.ai()
                    self.move()
                    

                    self.store_iter_local()
                    if self.iteration==0 or self.iteration==self.max_iteration-1:
                        self.log_images()

                    self.iteration+=1

                    end_time = rospy.get_time()
                    print('Time passed: ', end_time-current_time)
                self.log_test()
                rospy.sleep(2.0)
                #self.plot_progress()

        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            print('FINISHED THE BENCHMARKING TESTS!')
            rate.sleep()

if __name__=='__main__':
    # instantiate class and start loop function
    central_instance = Central()
    central_instance.central_execute()
