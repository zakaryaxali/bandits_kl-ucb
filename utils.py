import os.path
import pickle
import numpy as np

def write_data(sub_arm_list, total_rewards_list, name):
    """
    Write data in file
    """
    filename = 'data/scenario_1' + name + '.pkl'

    if os.path.exists(filename) :
       update_data_file(sub_arm_list, total_rewards_list, filename)
    else :
       create_data_file(sub_arm_list, total_rewards_list, filename)

def create_data_file(sub_arm_list, filename):
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

def update_data_file(sub_arm_list, filename):
    """
    Update data in file
    """
    parameters = read_file(filename)
    #update
    parameters['sub_arm_list'] += sub_arm_list
    parameters['total_rewards_list'] = np.concatenate((parameters['total_rewards_list'], total_rewards_list), axis=0)
    #write
    write = open(file, 'wb')
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
