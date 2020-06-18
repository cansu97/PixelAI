import numpy as np
import os
import glob
import sys
#if sys.version_info[0]<3:
import cv2

def shuffle_unison(a, b):
    assert a.shape[0] == len(b)
    p = np.random.permutation(len(b))
    return a[p], b[p]

def minmax_normalization(x_input , max_val, min_val):
    # Minmax normalization to get an input x_val in [-1,1] range
    return 2*(x_input-min_val)/(max_val-min_val)-1.0

def minmax_denorm(x_norm, max_val, min_val):
    # Minmax denormalization of the normalized input x_norm:
    return (x_norm+1.0)*(max_val-min_val)*(1/2)+min_val

def process_image(im, env=0):
    if env:
        # For the real NAO mode:
        # Apply a mask to put a plain background to the image:
        im = cv2.medianBlur(im, 11)
        hsv_frame = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([56, 100, 100])
        upper_bound = np.array([88, 255, 255])
        im_thres = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        im[im_thres != 0] = [49, 51, 53]
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)    
    # resize the image and normalize to 0-1 range:
    im_ready = cv2.resize(im_gray,(80,60))
    im_ready = im_ready.astype(float)/255
    # Crop two pixels from the bottom and top so that image size is 56x80
    return im_ready[2:-2,:]

def get_image_list(dataset_path, env):
    num_images = len(glob.glob(dataset_path+'/*.png'))
    image_list = []
    for i in range(0, num_images):
        im_name = 'robot_self_vision{}.png'.format(i)
        im = cv2.imread(os.path.join(dataset_path, im_name)) # Read image
        im_array = process_image(im, env)
        image_list.append(im_array)
    return image_list

class Data:
    def __init__(self, joint_angles, image_list):
        # Turn image list into numpy array:
        images = np.array(image_list)

        self.test_m = int((joint_angles.shape[0] * 0.1) / 100) * 100  # number of samples in the test set
        test_ind = np.array(random.sample(range(len(image_list)), self.test_m))

        self.X_test = joint_angles[test_ind, :]
        self.Y_test = images[test_ind, :, :]

        self.X_train = np.delete(joint_angles, test_ind, axis=0)
        self.Y_train = np.delete(images, test_ind, axis=0)
        self.m = self.X_train.shape[0]  # number of samples in the training set

        # Shuffle the dataset:
        self.X_train, self.Y_train = shuffle_unison(self.X_train, self.Y_train)

        # Store the min and max input values in the training set for data normalization:
        self.max_val = np.amax(self.X_train, axis=0)
        self.min_val = np.amin(self.X_train, axis=0)

        self.X_train = minmax_normalization(self.X_train, self.max_val, self.min_val)
        self.X_test = minmax_normalization(self.X_test, self.max_val, self.min_val)
        # Reshaping the output y to make it compatible for CNNs in pytorch:
        # num_images x num_channels x height x width
        self.Y_train = self.Y_train.reshape([-1, 1, self.Y_train.shape[1], self.Y_train.shape[2]])
        self.Y_test = self.Y_test.reshape([-1, 1, self.Y_test.shape[1], self.Y_test.shape[2]])

    def get_data_range(self):
        data_range = np.array([self.max_val, self.min_val])
        np.savetxt("data_range.csv", data_range, delimiter=',')

if __name__ == '__main__':
    if os.path.isdir('Dataset'):
        joint_angles = np.genfromtxt(os.path.join('Dataset','dataset_joint_angles.csv'), delimiter=",")
        image_list = get_image_list('Dataset', 0) #Create data for simulation!
        data = Data(joint_angles, image_list)
        print('Data generation has been completed')

        print('Training Dataset size: ', data.m)
        print('Test Dataset size: ', data.test_m)
    else: 
        print('You do not have access to the data files!')
