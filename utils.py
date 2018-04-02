import os.path
import pickle
import numpy as np

def write_data(sub_arm_list, total_rewards_list, name):
    """
    Write data in file
    """
    filename = 'data/scenario_1_' + name + '.pkl'

    if os.path.exists(filename) :
       update_data_file(sub_arm_list, total_rewards_list, filename)
    else :
       create_data_file(sub_arm_list, total_rewards_list, filename)

def create_data_file(sub_arm_list, total_rewards_list, filename):
    """
    Create file and add data
    """
    parameters = {
        'sub_arm_list': sub_arm_list,
        'total_rewards_list': total_rewards_list
    }
    output = open(filename, 'wb')
    pickle.dump(parameters, output)
    output.close()

def update_data_file(sub_arm_list, total_rewards_list, filename):
    """
    Update data in file
    """
    parameters = read_file(filename)
    #update
    parameters['sub_arm_list'] += sub_arm_list
    parameters['total_rewards_list'] = np.concatenate((parameters['total_rewards_list'], total_rewards_list), axis=0)
    #write
    write = open(filename, 'wb')
    pickle.dump(parameters, write)
    write.close()

def get_data(filename):
    """
    Get data from file
    """
    parameters = read_file(filename)
    return parameters['sub_arm_list'], parameters['total_rewards_list']

def read_file(filename):
    """
    Read file
    """
    read = open(filename, 'rb')
    parameters = pickle.load(read)
    read.close()
    return parameters

def get_results(total_rewards_list, sub_arm_list):
    """
    return results data for scenario 1 and 2 graphs
    """
    runs = len(sub_arm_list)
    mean_total_rewards_list = np.mean(total_rewards_list, axis=0)
    mean_sub_arm = np.mean(sub_arm_list, axis=0 ) #Mean number of the suboptimal arm as a function of time
    sub_arm_draws_T = np.zeros(runs) #number of draws of the suboptimal arm at tim n=5000
    for i in range(runs):
        sub_arm_draws_T[i] = np.sum(sub_arm_list[i][:5000])
    
    return mean_total_rewards_list, mean_sub_arm, sub_arm_draws_T
