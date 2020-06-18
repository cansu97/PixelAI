#!/usr/bin/env python
import sys
import rospy
from sensor_msgs.msg import Image,JointState
from gazebo_msgs.srv import GetJointProperties, GetJointPropertiesResponse
from std_msgs.msg import Float64, String
from std_srvs.srv import Empty
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import csv
import numpy as np
import os


class Central:


    def __init__(self):
        # initialize class variables
        self.joint_names = []
        self.i = 0
        self.joint_angles = []
        self.joint_velocities = []
        self.jointPub = 0

        self.Button_Number = 0  # indicates status of last button pressed
        self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.head_yaw=0.27
        self.head_pitch=0.224
        self.moved=0
        self.rshoulder_pitch=87.2*np.pi/180
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
    {
        "joint_name": "RShoulderPitch",
        "publisher": None,
        "limit_min": -2.08567,
        "limit_max": 2.08567
    },
    ]

    # Initialize joints (currently only head and left arm joints)
        topics = []
        for joint in joints:
            topic_name = '/nao_dcm/' + joint['joint_name'] + '_position_controller/command';
            topics.append(topic_name);
            joint['publisher'] = rospy.Publisher(topic_name, Float64, queue_size=5)

        self.joints = joints
        
        pass



    def key_cb(self,data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

    def joints_cb(self,data):
        #rospy.loginfo("joint states "+str(data.name)+str(data.position))
        # store current joint information in class variables
        self.joint_names = data.name 
        self.joint_angles = data.position
        self.joint_velocities = data.velocity

        pass


    def image_cb(self, data):
        bridge = CvBridge()
        try:
            cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            print(e)
        #cv2.imshow('image window',cv_image)
        self.cv_image = cv_image

    def set_head_angles(self, head_angle1, head_angle2): ## controls head movement
        self.joints[0]['publisher'].publish(head_angle1)
        self.joints[1]['publisher'].publish(head_angle2)

    def set_arm_angles(self, larm_angles): ## controls left arm movement
        self.joints[2]['publisher'].publish(larm_angles[0])
        self.joints[3]['publisher'].publish(larm_angles[1])
        self.joints[4]['publisher'].publish(larm_angles[2])
        self.joints[5]['publisher'].publish(larm_angles[3])
    
    def set_angles(self, joint_num, angle_to_set): ## controls arbitrary movement
        self.joints[joint_num]['publisher'].publish(angle_to_set)


    def central_execute(self):
        # Initialize ROS
        rospy.init_node('utility_node', anonymous=True)
        srv_joint_state = rospy.ServiceProxy('/gazebo/get_joint_properties', GetJointProperties)
        rospy.sleep(2.0)
        self.image_sub = rospy.Subscriber('/nao_robot/camera_bottom/image_raw',Image, self.image_cb)
        # Set the head and arm angles:
        self.set_head_angles(self.head_yaw, self.head_pitch)
        rospy.sleep(1.0)

        recorded_angles = np.genfromtxt("saved_angles_kinesthetic.csv", delimiter=",")
        if not os.path.isdir('self_vision_imgs'):
            os.mkdir('self_vision_imgs')
        # Put right arm down!
        self.set_angles(6, 1.5)

        for i in range(0, recorded_angles.shape[0]):
            self.set_arm_angles(recorded_angles[i,:])
            self.moved = 1
            rospy.sleep(1.0)
            im_name = 'sv_camera{}.png'.format(i)
            cv2.imwrite(os.path.join('self_vision_imgs', im_name), self.cv_image)
            print("Step {} completed!".format(i))

        rate = rospy.Rate(10) # sets the sleep time to 10ms
        while not rospy.is_shutdown():
            rate.sleep()

if __name__=='__main__':
    # instantiate class and start loop function
    central_instance = Central()
    central_instance.central_execute()
