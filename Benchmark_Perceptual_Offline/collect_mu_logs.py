import numpy as np
import os
import glob
import sys

if __name__ == '__main__':

    level=1
    num_cores=20
    num_test=10
    env='sim'

    res_path='Benchmark_res_'+env+'/'
    level_path=res_path+'Level'+str(level)
    log_path='Benchmark_mu'+str(level)
    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    for core in range(0, num_cores):
        mu_core_log = []
        core_path=level_path+'/Core'+str(core)
        for test in range (0, num_test):
            store_mu=np.genfromtxt(core_path+"/mu_log"+str(test)+".csv", delimiter=",")
            #Accumulate the vis error and the mu trajectories:
            mu_core_log.append(store_mu[0,:])
        #Save the mu-start-log file:
        np.savetxt(log_path+"/mu_"+str(core)+".csv", mu_core_log, delimiter=",")
