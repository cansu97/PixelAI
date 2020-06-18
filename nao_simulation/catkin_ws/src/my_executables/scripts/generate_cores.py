#!/usr/bin/env python
import sys
import os
import rospy
from sensor_msgs.msg import Image,JointState
from gazebo_msgs.srv import GetJointProperties, GetJointPropertiesResponse
from std_msgs.msg import Float64, String
from std_srvs.srv import Empty
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import random
import csv
import numpy as np


larm_min = np.array([  -2.0857, -0.3142, -2.0857, -1.5446])
larm_max = np.array([  2.0857,  1.3265, 2.0857, -0.0349])

def check_area(im_check):
    #return percentage of area covered by the robot's arm in the self vision image:
    im_gray = cv2.cvtColor(im_check, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(im_gray, 127, 255, cv2.THRESH_BINARY_INV)
    im_size=thresh.shape
    if sys.version_info[0]<3:       #Python 2.x version
        _, contour, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:                           #Python 3.x version (Not supported by ROS)
        contour = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contour1[0]
    if len(contour)>0:
        largest_blob = max(contour, key=cv2.contourArea)
        return cv2.contourArea(largest_blob)/np.prod(im_size)
    else: 
        return 0

class Central:
    def __init__(self):
        # initialize class variables
        self.joint_names = []
        self.joint_angles = []
        self.joint_velocities = []
        self.jointPub = 0
        self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.head_yaw=0.27
        self.head_pitch=0.224
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
        
        #Read the distribution parameters!
        self.clip_range = np.genfromtxt(os.path.join('Data_distribution', 'data_range_org.csv'), delimiter=",")
        self.mean_data = np.genfromtxt(os.path.join('Data_distribution', 'mean.csv'), delimiter=",")
        self.sigma_data = np.genfromtxt(os.path.join('Data_distribution', 'sigma.csv'), delimiter=",")
        
    

    def joints_cb(self,data):
        #store current joint information in class variables
        self.joint_names = data.name 
        self.joint_angles = data.position
        self.joint_velocities = data.velocity


    def image_cb(self, data):
        bridge = CvBridge()
        try:
            cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')
            #print('image received')
        except CvBridgeError as e:
            print(e)
        self.cv_image=cv_image
        
        cv2.imshow('Image', cv_image)
        cv2.waitKey(3)

    def set_head_angles(self,head_angle1,head_angle2): ## controls head movement
        self.joints[0]['publisher'].publish(head_angle1)
        self.joints[1]['publisher'].publish(head_angle2)

    def set_arm_angles(self,larm_angles): ## controls left arm movement
        self.joints[2]['publisher'].publish(larm_angles[0])
        self.joints[3]['publisher'].publish(larm_angles[1])
        self.joints[4]['publisher'].publish(larm_angles[2])
        self.joints[5]['publisher'].publish(larm_angles[3])
    

    def generate_sample(self):
        sample=np.random.multivariate_normal(self.mean_data, self.sigma_data, 1)
        return np.clip(sample, larm_min, larm_max)


    def central_execute(self):
        # Initialize ROS
        rospy.init_node('central_node', anonymous=True)
        srv_joint_state = rospy.ServiceProxy('/gazebo/get_joint_properties', GetJointProperties)

        rospy.sleep(2.0)
        # Subscribe to camera
        self.image_sub=rospy.Subscriber('/nao_robot/camera_bottom/image_raw',Image, self.image_cb)
        
        # Set the head and arm angles:
        self.set_head_angles(self.head_yaw, self.head_pitch)
        rospy.sleep(1.0)

        # dest_folder = '~/Cansu/nao_simulation/catkin_ws/src/my_executables/scripts/Benchmark_core_sim'
        dest_folder = '~/catkin_ws/src/my_executables/scripts/Benchmark_core_sim'
        target_file = 'benchmark_core.csv'

        if not os.path.isdir(dest_folder):
            os.mkdir(dest_folder)
        
        counter = 0
        no_collected_samples=0
        while(no_collected_samples<50 and (not rospy.is_shutdown())):
            sample = self.generate_sample()
            self.set_arm_angles(np.squeeze(sample))
            rospy.sleep(1.0)
            if check_area(self.cv_image)>0.04:
                im_name = 'core{}.png'.format(counter)
                cv2.imwrite(os.path.join(dest_folder, im_name), self.cv_image)
                with open(os.path.join(dest_folder, target_file), 'a') as samples:
                    samples_writer = csv.writer(samples)
                    samples_writer.writerow(np.squeeze(sample).tolist())
                samples.close()
                print('Sample {} completed'.format(counter))
                counter+=1
                no_collected_samples+=1
            else: 
                print("Disregarding this sample since area is very small!!!")


if __name__=='__main__':
    # instantiate class and start loop function
    central_instance = Central()
    central_instance.central_execute()
