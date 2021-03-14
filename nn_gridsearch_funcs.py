from AdversarialCNN import AdversarialCNN
from sklearn.model_selection import ParameterGrid
from Utils import DataLoader
import numpy as np
from scipy.signal import resample
import pickle
import os 
import tensorflow as tf

#main function
def gridsearch(grid, eeg_list, emg_list):
    '''
    input: gridded dictionary Example format: {'signal' : ['eeg','emg','both'],
                                        'class' : ['texture','weight'],
                                        'participants' : ['1','2','loo_2']}  #loo = leave one out
                                        
                        
           eeg_list, emg_list. obtained from: get_individiual_participant_data(9). 
             
    
    
    eeg_list, emg_list = get_individiual_participant_data(9)
    '''
    
    for params in grid:
                
        if(params['participants'][0:4] == 'loo_'):
            
            batch = 20
            epoch = 50
            
            train, test = handle_leave_one_out(eeg_list, emg_list, params['signal'], 
                                               params['class'], int(params['participants'][4]) )

        else:
            
            batch = 5
            epoch = 25
            
            train, test = handle_single_participant(eeg_list, emg_list, params['signal'],
                                                    params['class'], int(params['participants']) )
    
        if(params['signal'] == 'emg'): chans = 5

        elif(params['signal'] == 'eeg'): chans = 32

        elif(params['signal']=='both'): chans=37

            
        print('\n\nMODEL: ', params, '\n-----------------------------------------------------\n\n')
        log = params['signal'] + '_' + params['class'] + '_participant' + params['participants']
        
        if not (os.path.isdir(log)):
            os.mkdir(log)
        
        net = AdversarialCNN( chans=chans, samples=train[0].shape[2], n_output=2, 
                         n_nuisance=3, architecture='EEGNet', adversarial=False, lam=0 )  
    
        net.train( train, test, log = log, epochs=epoch, batch_size=batch )
        
        model = net.acnn
        model.save(log+'_msave')
    
#######################################################################################    
#supporting functions
def get_individiual_participant_data(num_participants):
    '''
    params: (int) total number of participants
    
    returns: list of all participant tupled data
    '''
    
    participant_data = DataLoader((1,num_participants),(1,9))
    
    participant_eeg_tuples = []
    participant_emg_tuples = []
     
    for i in range(num_participants):
        
        participant_data.load_data_for_nn(participant=i+1, sigtype='both')
        
        participant_eeg_tuple, participant_emg_tuple = participant_data.return_subset_for_nn(sigtype = 'both',
                                                                                   weights = [1,2,3], 
                                                                                    textures = [1,2,3])

        participant_eeg_tuples.append(participant_eeg_tuple)
        participant_emg_tuples.append(participant_emg_tuple)
        
        print('Loaded participant', i+1)
        
    return participant_eeg_tuples, participant_emg_tuples

def combine_eeg_emg(eegtuple, emgtuple):
    min_size = eegtuple[0].shape[2]
    emgtrials = emgtuple[0]
    resampled_trials = []
    for trial in list(np.split(emgtrials, emgtrials.shape[0])):
        resampled_channels = []
        for channel in list(np.split(trial, trial.shape[1], axis=1)):
            one_d = np.squeeze(channel)
            resampled_channel = resample(one_d, min_size)
            resampled_channel = np.reshape(resampled_channel, (-1,1))
            resampled_channels.append(resampled_channel)
        resampled_trials.append(np.hstack(resampled_channels))
    resampled_trials = np.dstack(resampled_trials)
    resampled_trials = np.reshape(resampled_trials, (resampled_trials.shape[0], resampled_trials.shape[1], resampled_trials.shape[2], 1))
    resampled_trials = np.moveaxis(np.moveaxis(resampled_trials, 0, 2), 0, 1)
    resampled_trials.shape


    combined = np.hstack((resampled_trials, eegtuple[0]))
    return (combined, eegtuple[1], eegtuple[2])

def get_min_samplesize(data):
    '''
    params: data: list of tuples
    
    return: minimum sample of list
    '''
    minimum = data[0][0].shape[2]
    for i in range(len(data)):
        if(data[i][0].shape[2] < minimum):
            minimum = data[i][0].shape[2]

    return minimum

def get_shuffled_binary_class_data_loo(train_ele_0,train_ele_1,train_ele_2,test_ele_0,test_ele_1,test_ele_2, classifier):
    '''
    description: remove third class to turn data set from multi class to binary
                    (loo = leave one out)
    params: all train/test tuple elements (6)
    
    return: elements passed in tuple form ready for fitting. shuffled and binary
    '''
    if(classifier=='texture'):
        #drop extra class
        weight_constant = np.array([0,1,0])   #we hold weight 2 constant
        texture_unwanted = np.array([0,0,1])  #we dont want texture 2
        
        train_ele_sub_0 = np.array([train_ele_0[i] for i in range(len(train_ele_0)) if np.all(train_ele_1[i] == weight_constant) 
                       and ~(np.all(train_ele_2[i] == texture_unwanted))])
        train_ele_sub_1 = np.array([train_ele_1[i] for i in range(len(train_ele_1)) if np.all(train_ele_1[i] == weight_constant) 
                       and ~(np.all(train_ele_2[i] == texture_unwanted))])
        train_ele_sub_2 = np.array([train_ele_2[i] for i in range(len(train_ele_2)) if np.all(train_ele_1[i] == weight_constant) 
                       and ~(np.all(train_ele_2[i] == texture_unwanted))])  

        test_ele_sub_0 = np.array([test_ele_0[i] for i in range(len(test_ele_0)) if np.all(test_ele_1[i] == weight_constant) 
                       and ~(np.all(test_ele_2[i] == texture_unwanted))])
        test_ele_sub_1 = np.array([test_ele_1[i] for i in range(len(test_ele_1)) if np.all(test_ele_1[i] == weight_constant) 
                       and ~(np.all(test_ele_2[i] == texture_unwanted))])  
        test_ele_sub_2 = np.array([test_ele_2[i] for i in range(len(test_ele_2)) if np.all(test_ele_1[i] == weight_constant) 
                       and ~(np.all(test_ele_2[i] == texture_unwanted))])
        
        train_ele_sub_2 = np.delete(train_ele_sub_2, 2, 1) #drop textuer column 2
        test_ele_sub_2 = np.delete(test_ele_sub_2, 2, 1) #drop textuer column 2
        
        #shuffle
        train_shuffle = np.random.permutation(range(train_ele_sub_0.shape[0]))
        test_shuffle = np.random.permutation(range(test_ele_sub_0.shape[0]))
        
        train_ele_sub_0 = train_ele_sub_0[train_shuffle,:,:,:]
        train_ele_sub_1 = train_ele_sub_1[train_shuffle,:]
        train_ele_sub_2 = train_ele_sub_2[train_shuffle,:]
        
        test_ele_sub_0 = test_ele_sub_0[test_shuffle,:,:,:]
        test_ele_sub_1 = test_ele_sub_1[test_shuffle,:]
        test_ele_sub_2 = test_ele_sub_2[test_shuffle,:]
        
        train_tuple = (train_ele_sub_0,train_ele_sub_2,train_ele_sub_1)
        test_tuple = (test_ele_sub_0,test_ele_sub_2,test_ele_sub_1)
        
    elif(classifier=='weight'):
        #drop extra class
        weight_unwanted = np.array([0,1,0])  #we dont want weight 2
        texture_constant = np.array([0,0,1]) #we hold texture 2 constant
    
        train_ele_sub_0 = np.array([train_ele_0[i] for i in range(len(train_ele_0)) if np.all(train_ele_2[i] == texture_constant) 
                       and ~(np.all(train_ele_1[i] == weight_unwanted))])
        train_ele_sub_1 = np.array([train_ele_1[i] for i in range(len(train_ele_1)) if np.all(train_ele_2[i] == texture_constant) 
                       and ~(np.all(train_ele_1[i] == weight_unwanted))])  
        train_ele_sub_2 = np.array([train_ele_2[i] for i in range(len(train_ele_2)) if np.all(train_ele_2[i] == texture_constant) 
                       and ~(np.all(train_ele_1[i] == weight_unwanted))])  

        test_ele_sub_0 = np.array([test_ele_0[i] for i in range(len(test_ele_0)) if np.all(test_ele_2[i] == texture_constant) 
                       and ~(np.all(test_ele_1[i] == weight_unwanted))])
        test_ele_sub_1 = np.array([test_ele_1[i] for i in range(len(test_ele_1)) if np.all(test_ele_2[i] == texture_constant) 
                       and ~(np.all(test_ele_1[i] == weight_unwanted))])  
        test_ele_sub_2 = np.array([test_ele_2[i] for i in range(len(test_ele_2)) if np.all(test_ele_2[i] == texture_constant) 
                       and ~(np.all(test_ele_1[i] == weight_unwanted))])      
        
        train_ele_sub_1 = np.delete(train_ele_sub_1, 1, 1) #drop textuer column 2
        test_ele_sub_1 = np.delete(test_ele_sub_1, 1, 1) #drop textuer column 2
    
        #shuffle
        train_shuffle = np.random.permutation(range(train_ele_sub_0.shape[0]))
        test_shuffle = np.random.permutation(range(test_ele_sub_0.shape[0]))
        
        train_ele_sub_0 = train_ele_sub_0[train_shuffle,:,:,:]
        train_ele_sub_1 = train_ele_sub_1[train_shuffle,:]
        train_ele_sub_2 = train_ele_sub_2[train_shuffle,:]
        
        test_ele_sub_0 = test_ele_sub_0[test_shuffle,:,:,:]
        test_ele_sub_1 = test_ele_sub_1[test_shuffle,:]
        test_ele_sub_2 = test_ele_sub_2[test_shuffle,:]
        
        train_tuple = (train_ele_sub_0,train_ele_sub_1,train_ele_sub_2)
        test_tuple = (test_ele_sub_0,test_ele_sub_1,test_ele_sub_2)
        
    return train_tuple,test_tuple

def get_shuffled_binary_class_data_single(data,classifier):
    '''
    description: remove third class to turn data set from multi class to binary
                    (loo = leave one out)
    params: single participant data in tuple form
            string classifier (texture or weight)
    
    return: elements passed in tuple form ready for fitting. shuffled and binary
    '''
    #shuffle
    shuffled_index = np.random.permutation(range(data[0].shape[0]))

    tuple_tmp_element_0 = data[0][shuffled_index,:,:,:]
    tuple_tmp_element_1 = data[1][shuffled_index,:]
    tuple_tmp_element_2 = data[2][shuffled_index,:]
    
    #train test split
    train_test_split = int(data[0].shape[0] * .75)
    
    train_ele_0 = tuple_tmp_element_0[0:train_test_split,:,:,:]
    train_ele_1 = tuple_tmp_element_1[0:train_test_split,:]
    train_ele_2 = tuple_tmp_element_2[0:train_test_split,:]
    
    test_ele_0 = tuple_tmp_element_0[train_test_split:,:,:,:]
    test_ele_1 = tuple_tmp_element_1[train_test_split:,:]
    test_ele_2 = tuple_tmp_element_2[train_test_split:,:]
    
    #dropping extra classes, forming tuple
    if(classifier=='texture'):
        weight_constant = np.array([0,1,0])   #we hold weight 2 constant
        texture_unwanted = np.array([0,0,1])  #we dont want texture 2
        
        train_ele_sub_0 = np.array([train_ele_0[i] for i in range(len(train_ele_0)) if np.all(train_ele_1[i] == weight_constant) 
                       and ~(np.all(train_ele_2[i] == texture_unwanted))])
        train_ele_sub_1 = np.array([train_ele_1[i] for i in range(len(train_ele_1)) if np.all(train_ele_1[i] == weight_constant) 
                       and ~(np.all(train_ele_2[i] == texture_unwanted))])
        train_ele_sub_2 = np.array([train_ele_2[i] for i in range(len(train_ele_2)) if np.all(train_ele_1[i] == weight_constant) 
                       and ~(np.all(train_ele_2[i] == texture_unwanted))])  

        test_ele_sub_0 = np.array([test_ele_0[i] for i in range(len(test_ele_0)) if np.all(test_ele_1[i] == weight_constant) 
                       and ~(np.all(test_ele_2[i] == texture_unwanted))])
        test_ele_sub_1 = np.array([test_ele_1[i] for i in range(len(test_ele_1)) if np.all(test_ele_1[i] == weight_constant) 
                       and ~(np.all(test_ele_2[i] == texture_unwanted))])  
        test_ele_sub_2 = np.array([test_ele_2[i] for i in range(len(test_ele_2)) if np.all(test_ele_1[i] == weight_constant) 
                       and ~(np.all(test_ele_2[i] == texture_unwanted))])
        
        train_ele_sub_2 = np.delete(train_ele_sub_2, 2, 1) #drop textuer column 2
        test_ele_sub_2 = np.delete(test_ele_sub_2, 2, 1) #drop textuer column 2
        
        train_tuple = (train_ele_sub_0,train_ele_sub_2,train_ele_sub_1)
        test_tuple = (test_ele_sub_0,test_ele_sub_2,test_ele_sub_1)
        
    elif(classifier=='weight'):
        weight_unwanted = np.array([0,1,0])  #we dont want weight 2
        texture_constant = np.array([0,0,1]) #we hold texture 2 constant
    
        train_ele_sub_0 = np.array([train_ele_0[i] for i in range(len(train_ele_0)) if np.all(train_ele_2[i] == texture_constant) 
                       and ~(np.all(train_ele_1[i] == weight_unwanted))])
        train_ele_sub_1 = np.array([train_ele_1[i] for i in range(len(train_ele_1)) if np.all(train_ele_2[i] == texture_constant) 
                       and ~(np.all(train_ele_1[i] == weight_unwanted))])  
        train_ele_sub_2 = np.array([train_ele_2[i] for i in range(len(train_ele_2)) if np.all(train_ele_2[i] == texture_constant) 
                       and ~(np.all(train_ele_1[i] == weight_unwanted))])  

        test_ele_sub_0 = np.array([test_ele_0[i] for i in range(len(test_ele_0)) if np.all(test_ele_2[i] == texture_constant) 
                       and ~(np.all(test_ele_1[i] == weight_unwanted))])
        test_ele_sub_1 = np.array([test_ele_1[i] for i in range(len(test_ele_1)) if np.all(test_ele_2[i] == texture_constant) 
                       and ~(np.all(test_ele_1[i] == weight_unwanted))])  
        test_ele_sub_2 = np.array([test_ele_2[i] for i in range(len(test_ele_2)) if np.all(test_ele_2[i] == texture_constant) 
                       and ~(np.all(test_ele_1[i] == weight_unwanted))])      
        
        train_ele_sub_1 = np.delete(train_ele_sub_1, 1, 1) #drop textuer column 2
        test_ele_sub_1 = np.delete(test_ele_sub_1, 1, 1) #drop textuer column 2
        
        train_tuple = (train_ele_sub_0,train_ele_sub_1,train_ele_sub_2)
        test_tuple = (test_ele_sub_0,test_ele_sub_1,test_ele_sub_2)
        
    return train_tuple,test_tuple
    
    
    

def handle_leave_one_out(participant_eeg_list, participant_emg_list, signal, classifier, p_num_loo):
    '''
    Description: have to seperate leave one out as it requires different handling of
                    train test splitting and requires extracting/stacking participant data
                    
    params: participant_eeg_list: list of eeg tupled data
            participant_emg_list: list of emg tupled data
            p_num_loo: string of particiant to be used as the test
            classifier: string of class to be classified. needed for correct tuple element order (data,weight,texture)
            signal: string of signal being used for classification
            
    returns: train, test tuples ready to be passed to train function
    '''
    
    
    #setting up data list based off signal type. For eeg, emg also run sample equalization code
    if(signal == 'eeg'):
        data_list = participant_eeg_list
        minimum = get_min_samplesize(data_list)
        
    elif(signal == 'emg'):
        data_list = participant_emg_list
        minimum = get_min_samplesize(data_list)
        
    elif(signal=='both'):
        data_list = []
        for i in range(9):
            combined_data = combine_eeg_emg(participant_eeg_list[i],participant_emg_list[i])
            data_list.append(combined_data)
        minimum = get_min_samplesize(data_list)
    
    
    #TODO: add dropping of columns for different classes / SHUFFLE
    #filling train/test lists by defined data type
    #change appending system to stacking np arrays
    first = True
    for i in range(len(data_list)):
        if(signal=='eeg' or signal=='emg' or signal=='both'):
            if((i+1) != int(p_num_loo)): #if participant in list is not the one to leave and be tested on, add to train
                if(first):
                    train_ele_0 = resample(data_list[i][0],minimum,axis=2)
                    train_ele_1 = data_list[i][1]
                    train_ele_2 = data_list[i][2]
                    first = False

                else:
                    train_ele_0 = np.append( train_ele_0 , resample(data_list[i][0],minimum,axis=2) , 0 )
                    train_ele_1 = np.append( train_ele_1 , data_list[i][1] , 0)
                    train_ele_2 = np.append( train_ele_2 , data_list[i][2] , 0)


            elif((i+1) == int(p_num_loo)): #finding the one to leave out and appending to test lists
                test_ele_0 = resample(data_list[i][0],minimum,axis=2)
                test_ele_1 = data_list[i][1]
                test_ele_2 = data_list[i][2]
    
#     print('\n')
#     print(train_ele_0.shape,train_ele_1.shape,train_ele_2.shape)
#     print(test_ele_0.shape,test_ele_1.shape,test_ele_2.shape) 
#     print('\n')
    
    #converting to binary classification and shuffle
    train_tuple,test_tuple = get_shuffled_binary_class_data_loo(train_ele_0,train_ele_1,train_ele_2,test_ele_0,
                                                            test_ele_1,test_ele_2,classifier)
    
    print('tr texture1:', np.sum(train_tuple[1][:,0]))
    print('tr texture2:', np.sum(train_tuple[1][:,1]))
    
    print('te texture1:', np.sum(test_tuple[1][:,0]))
    print('te texture2:', np.sum(test_tuple[1][:,1])) 
    
    return train_tuple,test_tuple
        

#handle_leave_one_out(participant_eeg_list, participant_emg_list, signal, classifier, p_num_loo):
def handle_single_participant(participant_eeg_list,participant_emg_list,signal,classifier,p_num):
    
    if(signal == 'eeg'):
        data = participant_eeg_list[int(p_num)-1]
        
    elif(signal == 'emg'):
        data = participant_emg_list[int(p_num)-1]
        
    elif(signal=='both'):
        data = combine_eeg_emg(participant_eeg_list[int(p_num)-1],participant_emg_list[int(p_num)-1])

    
    
    train_tuple,test_tuple = get_shuffled_binary_class_data_single(data,classifier)
    
    print('tr texture1:', np.sum(train_tuple[1][:,0]))
    print('tr texture2:', np.sum(train_tuple[1][:,1]))
    
    print('te texture1:', np.sum(test_tuple[1][:,0]))
    print('te texture2:', np.sum(test_tuple[1][:,1])) 
    
    return train_tuple,test_tuple
