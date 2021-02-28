import numpy as np
from scipy.io import loadmat  
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp



class EegData:

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
            both = True

        data = self.data['participant {}'.format(str(participant))]

        if both:
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
                elif both:
                    eegsignal = trial['eeg']
                    emgsignal = trial['emg']
                else:
                    print("sigtype should be eeg, emg, or both")
                    return None

                w_class = trial['weight_class']
                t_class = trial['texture_class']

                if w_class == 4:
                    w_class = 3
        
                if both:
                    eegarr = eegsignal.to_numpy()
                    emgarr = emgsignal.to_numpy()
                else:
                    arr = signal.to_numpy()

                w_encoding = 1*(np.array([(w_class == 1),(w_class == 2),(w_class == 3)]))
                t_encoding = 1*(np.array([(t_class == 1),(t_class == 2),(t_class == 3)]))

                if both:
                    e_w_encoding = np.vstack([w_encoding]*eegarr.shape[0])
                    e_t_encoding = np.vstack([t_encoding]*eegarr.shape[0])
                    m_w_encoding = np.vstack([w_encoding]*emgarr.shape[0])
                    m_t_encoding = np.vstack([t_encoding]*emgarr.shape[0])
                else:
                    w_encoding = np.vstack([w_encoding]*arr.shape[0])
                    t_encoding = np.vstack([t_encoding]*arr.shape[0])

                if both:
                    eegarr = np.hstack((eegarr,e_w_encoding,e_t_encoding))
                    emgarr = np.hstack((emgarr,m_w_encoding,m_t_encoding))
                else:
                    arr = np.hstack((arr,w_encoding,t_encoding))

                if first:
                    if both:
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
                    if both:
                        all_eeg_arr.append(eegarr)
                        all_emg_arr.append(emgarr)
                    else:
                        all_arr.append(arr)
            if both:
                eegsessions.append(all_eeg_arr)
                emgsessions.append(all_emg_arr)
            else:
                sessions.append(all_arr)
        if both:
            return eegsessions, emgsessions
        else:
            return sessions


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

            
