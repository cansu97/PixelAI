Benchmark_mu1&2&3: 
-These folders contain the starting internal belief (mu) values that were randomly generated in the perceptual inference tests. So the same start angles are used for the perceptual and active inference tests.

Dataset:
-Dataset used to train the convolutional decoder: Images + csv file containing the corresponding joint angle values

Benchmark_core:
-Core arm positions used for the statistical tests (Same cores for both the perceptual and active inference tests).

generate_cores.py: Generates random samples on the go and asks the user whether it should be disregarded or not!
generate_random_samples.py -> Not worked over yet! Check whether there is a contour above a certain threshold and if yes store it inside the dataset! -> Simple thresholding on monochrome image

Data distribution: 
-Parameters for a multivariate Gaussian that was fit into the first part of the dataset that was collected kinesthetically. 

Conv_decoder_model: This model contains the utility functions for the 


Nao_comm: Real robot communication
1) get angles records the angles with head button touch (emulating kinesthetic teaching) to form the first part of the dataset (saved_angles.csv contains 803 samples that were collected this way).

2) get_images.py goes through the prerecorded angles and records the images from the bottom camera of Nao in the real robot. 



