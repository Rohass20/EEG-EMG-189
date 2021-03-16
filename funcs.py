#!/usr/bin/env python
# coding: utf-8

# # Data loading Functions

# In[5]:


import numpy as np
from scipy.io import loadmat  
import matplotlib.pyplot as plt
import pandas as pd

def load_cols():
    mat = loadmat('WS_P1_S1.mat')
    eegnames = mat['ws']['names'][0][0][0][0][0][0]
    emgnames = mat['ws']['names'][0][0][0][0][2][0]
    eegcolumns = []
    emgcolumns = []

    for name in eegnames:
        eegcolumns.append(name[0])
    for name in emgnames:
        emgcolumns.append(name[0])
    return eegcolumns, emgcolumns

def load_all_trials_from_file(participant,session,eegcolumns,emgcolumns):
    mat = loadmat('WS_P' + str(participant) + '_S' + str(session) + '.mat')
    
    trials = {}
    for i, trial in enumerate(mat['ws']['win'][0][0][0]):
        trialdict = {}
        trialdict['eeg'] = pd.DataFrame(trial[0],columns=eegcolumns)
        trialdict['emg'] = pd.DataFrame(trial[2],columns=emgcolumns)
        trialdict['eegt'] = trial[3][0]
        trialdict['emgt'] = trial[4][0]
        trialdict['start_time'] = trial[5][0][0]
        trialdict['LEDon'] = trial[7][0][0]
        trialdict['LEDoff'] = trial[8][0][0]
        trialdict['weight_in_grams'] = float(trial[10][0].strip('g'))
        trialdict['texture'] = trial[12][0]

        trials['trial ' + str(i+1)] = trialdict
    
    return trials

def load_participants_ws(firstp,lastp,firsts,lasts):
    eegcols, emgcols = load_cols()
    data = {}
    for participant in range(firstp, lastp + 1):
        participantdict = {}
        for session in range(firsts, lasts + 1):
            participantdict['session ' + str(session)] = load_all_trials_from_file(participant,session,eegcols,emgcols)
        data['participant ' + str(participant)] = participantdict
    return data

def load_to_df(filename):
    mat = loadmat(filename)
    mdata = mat['hs']  
    mdtype = mdata.dtype  
    ndata = {n: mdata[n][0, 0] for n in mdtype.names}

    emg = ndata['emg']
    eeg = ndata['eeg']
    emgsamplingrate = 4000
    eegsamplingrate = 500

    eegdata = eeg[0][0][1]
    eegcolumnnames = eeg[0][0][0][0]
    emgcolumnnames = emg[0][0][1][0]
    emgdata = emg[0][0][0]

    eegcolumns = []
    for element in eegcolumnnames:
        eegcolumns.append(element[0])

    emgcolumns = []
    for element in emgcolumnnames:
        emgcolumns.append(element[0])

    eegdf = pd.DataFrame(eegdata,columns=eegcolumns)
    emgdf = pd.DataFrame(emgdata,columns=emgcolumns)
    eegt = np.arange(0,eegdf.shape[0]/eegsamplingrate,1/eegsamplingrate)
    emgt = np.arange(0,emgdf.shape[0]/emgsamplingrate,1/emgsamplingrate)

    return eegdf, emgdf, eegt, emgt

def load_dfs_to_dict(participant, session):
    filename = 'HS_P' + str(participant) + '_S' + str(session) +'.mat'
    eegdf, emgdf, eegt, emgt = load_to_df(filename)
    list = [eegdf, emgdf, eegt, emgt]
    dictionary = {}
    attribute_list = ['eegdf','emgdf','eegt','emgt']
    for i, att in enumerate(attribute_list):
        dictionary[att] = list[i]

    return dictionary

def load_participants_hs(firstp, lastp, firsts, lasts):
    data = {}
    for participant in range(firstp, lastp + 1):
        participantdict = {}
        for session in range(firsts, lasts + 1):
            participantdict['session ' + str(session)] = load_dfs_to_dict(participant,session)
        data['participant ' + str(participant)] = participantdict
    return data


# In[ ]:




