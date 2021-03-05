import numpy as np
from scipy.io import loadmat  
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
from scipy.signal import resample



class DataLoader:

    def __init__(self, participant, session):
        '''
        initializes object and loads in specified data. Data can then be retrieved
        with get_data or transformed and loaded for nn use with load_data_for_nn
        '''
        self.participant = participant
        self.session = session
        self.load_data()

    def which_participants(self):
        return self.participant

    def which_sessions(self):
        return self.session
        
    
    def load_data(self):
        self.data = None
        if type(self.participant) is int:
            if type(self.session) is int:
                self.data = self.load_participants_ws(self.participant, self.participant, self.session, self.session)
            if type(self.session) is tuple:
                self.data = self.load_participants_ws(self.participant, self.participant, self.session[0],self.session[1])
        if type(self.participant) is tuple:
            if type(self.session) is int:
                self.data = self.load_participants_ws(self.participant[0], self.participant[1], self.session, self.session)
            if type(self.session) is tuple:
                self.data = self.load_participants_ws(self.participant[0], self.participant[1], self.session[0],self.session[1])
        if self.data == None:
            print(type(self.participant), type(self.session))
            print("participant must be int or tuple and session must be int or tuple")
        return self.data

    
    def get_data(self):
        return self.data


    def load_cols(self):
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

    
    def load_all_trials_from_file(self,participant,session,eegcolumns,emgcolumns):
        mat = loadmat('WS_P' + str(participant) + '_S' + str(session) + '.mat')
        trials = {}
        for i, trial in enumerate(mat['ws']['win'][0][0][0]):
            trialdict = {}
            trialdict['eeg'] = pd.DataFrame(trial[0],columns=eegcolumns)
            trialdict['emg'] = pd.DataFrame(trial[2],columns=emgcolumns)
            trialdict['eegt'] = trial[3][0]
            trialdict['emgt'] = trial[4][0]
            trialdict['start_time'] = trial[5][0][0]
            trialdict['LEDon'] = trial[6][0][0]
            trialdict['LEDoff'] = trial[7][0][0]
            trialdict['weight_class'] = trial[9][0][0]
            trialdict['weight_in_grams'] = float(trial[10][0].strip('g'))
            trialdict['texture_class'] = trial[11][0][0]
            trialdict['texture'] = trial[12][0]

            trials['trial ' + str(i+1)] = trialdict
    
        return trials

    def load_participants_ws(self,firstp,lastp,firsts,lasts):
        eegcols, emgcols = self.load_cols()
        data = {}
        for participant in range(firstp, lastp + 1):
            participantdict = {}
            for session in range(firsts, lasts + 1):
                participantdict['session ' + str(session)] = self.load_all_trials_from_file(participant,session,eegcols,emgcols)
            data['participant ' + str(participant)] = participantdict
            print("done with p")
        return data


    def load_data_for_nn(self, participant, sigtype):
        '''
        Loads data for NN use

        arguments:
        int participant: participant number (1 through 9)
        string sigtype: type of data to return, ('eeg','emg','both')

        returns:
        list sessions: list of sessions for either eeg, emg, or one list of each
        each list entry is a list of numpy arrays for each session with
        one hot encoded weight and texture features
        
        eeg, emg
        '''
        if sigtype == 'both':
            self.both = True
        else:
            self.both = False

        data = self.data['participant {}'.format(str(participant))]

        if self.both:
            eegsessions = []
            emgsessions = []
        else:
            sessions = []

        for session in data.values():
            first = True
            for trial in session.values():
                if sigtype == 'eeg':
                    signal = trial['eeg']
                elif sigtype == 'emg':
                    signal = trial['emg']
                elif self.both:
                    eegsignal = trial['eeg']
                    emgsignal = trial['emg']
                else:
                    print("sigtype should be eeg, emg, or both")
                    return None

                w_class = trial['weight_class']
                t_class = trial['texture_class']

                if w_class == 4:
                    w_class = 3
        
                if self.both:
                    eegarr = eegsignal.to_numpy()
                    emgarr = emgsignal.to_numpy()
                else:
                    arr = signal.to_numpy()

                w_encoding = 1*(np.array([(w_class == 1),(w_class == 2),(w_class == 3)]))
                t_encoding = 1*(np.array([(t_class == 1),(t_class == 2),(t_class == 3)]))

                if self.both:
                    e_w_encoding = np.vstack([w_encoding]*eegarr.shape[0])
                    e_t_encoding = np.vstack([t_encoding]*eegarr.shape[0])
                    m_w_encoding = np.vstack([w_encoding]*emgarr.shape[0])
                    m_t_encoding = np.vstack([t_encoding]*emgarr.shape[0])
                else:
                    w_encoding = np.vstack([w_encoding]*arr.shape[0])
                    t_encoding = np.vstack([t_encoding]*arr.shape[0])

                if self.both:
                    eegarr = np.hstack((eegarr,e_w_encoding,e_t_encoding))
                    emgarr = np.hstack((emgarr,m_w_encoding,m_t_encoding))
                else:
                    arr = np.hstack((arr,w_encoding,t_encoding))

                if first:
                    if self.both:
                        all_eeg_arr = []
                        all_emg_arr = []
                        all_eeg_arr.append(eegarr)
                        all_emg_arr.append(emgarr)
                    else:
                        all_arr = []
                        all_arr.append(arr)
                    first = False
                    continue
                else:
                    if self.both:
                        all_eeg_arr.append(eegarr)
                        all_emg_arr.append(emgarr)
                    else:
                        all_arr.append(arr)
            if self.both:
                eegsessions.append(all_eeg_arr)
                emgsessions.append(all_emg_arr)
            else:
                sessions.append(all_arr)
        if self.both:
            self.eegsessions = eegsessions
            self.emgsessions = emgsessions
        else:
            self.sessions = sessions
        
        return self.resample_and_store_tuple()

    def resample_and_store_tuple(self):
        self.find_min_samples()
        if self.both:
            first = True
            for session in self.eegsessions:
                for trial in session:
                    resampled_columns = []
                    column_list = np.hsplit(trial, trial.shape[1])
                    for array in column_list:
                        array_resampled = resample(x=array, num=self.min_eeg_samples)
                        resampled_columns.append(array_resampled) 
                    if first:
                        self.all_eeg_trials = np.hstack(resampled_columns)
                        self.eeg_weight_encodings = self.all_eeg_trials[0,32:35]
                        self.eeg_texture_encodings = self.all_eeg_trials[0,35:]
                        first = False
                    else:
                        this_trial = np.hstack(resampled_columns)
                        self.all_eeg_trials = np.dstack((self.all_eeg_trials, this_trial))
                        self.eeg_weight_encodings = np.vstack((self.eeg_weight_encodings, this_trial[0,32:35]))
                        self.eeg_texture_encodings = np.vstack((self.eeg_texture_encodings, this_trial[0,35:]))
            self.eeg_tuple = (self.all_eeg_trials, self.eeg_weight_encodings, self.eeg_texture_encodings)
            first = True
            for session in self.emgsessions:
                for trial in session:
                    resampled_columns = []
                    column_list = np.hsplit(trial, trial.shape[1])
                    for array in column_list:
                        array_resampled = resample(x=array, num=self.min_emg_samples)
                        resampled_columns.append(array_resampled) 
                    if first:
                        self.all_emg_trials = np.hstack(resampled_columns)
                        self.emg_weight_encodings = self.all_emg_trials[0,5:8]
                        self.emg_texture_encodings = self.all_emg_trials[0,8:]
                        first = False
                    else:
                        this_trial = np.hstack(resampled_columns)
                        self.all_emg_trials = np.dstack((self.all_emg_trials, this_trial))
                        self.emg_weight_encodings = np.vstack((self.emg_weight_encodings, this_trial[0,5:8]))
                        self.emg_texture_encodings = np.vstack((self.emg_texture_encodings, this_trial[0,8:]))
            self.emg_tuple = (self.all_emg_trials, self.emg_weight_encodings, self.emg_texture_encodings)
        else:
            first = True
            for session in self.sessions:
                for trial in session:
                    resampled_columns = []
                    column_list = np.hsplit(trial, trial.shape[1])
                    for array in column_list:
                        array_resampled = resample(x=array, num=self.min_samples)
                        resampled_columns.append(array_resampled) 
                    if first:
                        self.all_trials = np.hstack(resampled_columns)
                        self.weight_encodings = self.all_trials[0,32:35]
                        self.texture_encodings = self.all_trials[0,35:]
                        first = False
                    else:
                        this_trial = np.hstack(resampled_columns)
                        self.all_trials = np.dstack((self.all_trials, this_trial))
                        self.weight_encodings = np.vstack((self.weight_encodings, this_trial[0,32:35]))
                        self.texture_encodings = np.vstack((self.texture_encodings, this_trial[0,35:]))
            self.tuple = (self.all_trials, self.weight_encodings, self.texture_encodings)
        if self.both:
            return self.eeg_tuple, self.emg_tuple
        else:
            return self.tuple


    def find_min_samples(self):
        if self.both:
            min_eeg_samples = self.eegsessions[0][0].shape[0]
            for session in self.eegsessions:
                for trial in session:
                    if trial.shape[0] < min_eeg_samples:
                        min_eeg_samples = trial.shape[0]
            min_emg_samples = self.emgsessions[0][0].shape[0]
            for session in self.emgsessions:
                for trial in session:
                    if trial.shape[0] < min_emg_samples:
                        min_emg_samples = trial.shape[0]     
            self.min_eeg_samples = min_eeg_samples
            self.min_emg_samples = min_emg_samples
        else:
            min_samples = self.sessions[0][0].shape[0]
            for session in self.sessions:
                for trial in session:
                    if trial.shape[0] < min_samples:
                        min_samples = trial.shape[0]
            self.min_samples = min_samples
    
    

    """
    def multi(self,n):
        if n == 1:
            data = self.load_participants_ws(1,3,1,9)
        if n == 2:
            data = self.load_participants_ws(4,6,1,9)
        if n == 3:
            data = self.load_participants_ws(7,9,1,9)
        if n == 4:
            data = self.load_participants_ws(10,12,1,9)
        return data


    def parallel_load(self):
        with mp.Pool(4) as pool:
            data = [pool.map(self.multi, [1,2,3,4])]
            pool.close()
            pool.join()
            return data
    

    def get_weight_averages(self, participant):
        pdata = self.data['participant {}'.format(str(self.participant))]
        averages = []
        counter = 0
        onecounter = 0
        twocounter = 0
        threecounter = 0
        for session in pdata.values():
            trialweights = []
            trialeeg = []
            for trial in session.values():
                trialweights.append(trial['weight_class'])
                trialeeg.append((trial['eeg'],trial['eegt']))

            arrays1 = []
            arrays2 = []
            arrays3 = []
            for i in range(len(trialweights)):
                if trialweights[i] == 1:
                    array = trialeeg[i][0]
                    arrays1.append(array)
                if trialweights[i] == 2:
                    array = trialeeg[i][0]
                    arrays2.append(array)
                if trialweights[i] == 4:
                    array = trialeeg[i][0]
                    arrays3.append(array)
    """

            
