import matplotlib.pyplot as plt
import numpy as np

def calc_averages(data,participant,signal):
    
    weight1_emgs = []
    weight2_emgs = []
    weight3_emgs = []

    texture1_emgs = []
    texture2_emgs = []
    texture3_emgs = []

    weightclass1 = 165 #grams
    weightclass2 = 330
    weightclass3 = 660

    texture1 = 'suede'
    texture2 = 'sandpaper'
    texture3 = 'silk'

    #sorting emg data by weight class

    for j in range(len(data[participant])):

        session = 'session ' + str(j+1)

        for i in range(len(data[participant][session])):

            trial = 'trial ' + str(i+1)

            #appending weights
            
            if (data[participant][session][trial]['weight_in_grams'] == weightclass1 and 
                data[participant][session][trial]['texture'] == texture2 ):

                weight1_emgs.append(data[participant][session][trial][signal])

            elif (data[participant][session][trial]['weight_in_grams'] == weightclass2 and 
                data[participant][session][trial]['texture'] == texture2 ):

                weight2_emgs.append(data[participant][session][trial][signal])

            elif (data[participant][session][trial]['weight_in_grams'] == weightclass3 and 
                data[participant][session][trial]['texture'] == texture2 ):

                weight3_emgs.append(data[participant][session][trial][signal])

            #appending textures
            
            if (data[participant][session][trial]['weight_in_grams'] == weightclass2 and 
                data[participant][session][trial]['texture'] == texture1 ):

                texture1_emgs.append(data[participant][session][trial][signal])

            elif (data[participant][session][trial]['weight_in_grams'] == weightclass2 and 
                data[participant][session][trial]['texture'] == texture2 ):

                texture2_emgs.append(data[participant][session][trial][signal])

            elif (data[participant][session][trial]['weight_in_grams'] == weightclass2 and 
                data[participant][session][trial]['texture'] == texture3 ):

                texture3_emgs.append(data[participant][session][trial][signal])

    #average weights
    weight_sum1 = weight1_emgs[0]
    weight_sum2 = weight2_emgs[0]
    weight_sum3 = weight3_emgs[0]

    for i in range(len(weight1_emgs)):

        if(i==0): continue

        weight_sum1 += weight1_emgs[i]

    weight_avg1 = weight_sum1/len(weight1_emgs)


    for i in range(len(weight2_emgs)):

        if(i==0): continue

        weight_sum2 += weight2_emgs[i]

    weight_avg2 = weight_sum2/len(weight2_emgs)

    for i in range(len(weight3_emgs)):

        if(i==0): continue

        weight_sum3 += weight3_emgs[i]

    weight_avg3 = weight_sum3/len(weight3_emgs)
    
    #average textures
    texture_sum1 = texture1_emgs[0]
    texture_sum2 = texture2_emgs[0]
    texture_sum3 = texture3_emgs[0]

    for i in range(len(texture1_emgs)):

        if(i==0): continue

        texture_sum1 += texture1_emgs[i]

    texture_avg1 = texture_sum1/len(texture1_emgs)


    for i in range(len(texture2_emgs)):

        if(i==0): continue

        texture_sum2 += texture2_emgs[i]

    texture_avg2 = texture_sum2/len(texture2_emgs)

    for i in range(len(texture3_emgs)):

        if(i==0): continue

        texture_sum3 += texture3_emgs[i]

    texture_avg3 = texture_sum3/len(texture3_emgs)
    
    return [weight_avg1,weight_avg2,weight_avg3,texture_avg1,texture_avg2,texture_avg3]

def return_eeg_emg_averages(data,participant):
    eeg_list = calc_averages(data,participant,'eeg')
    emg_list = calc_averages(data,participant,'emg')
    return eeg_list,emg_list
    
def plot_emg_diffs(emg_list,pnum):
    fig, ax = plt.subplots(nrows=5, ncols=2,figsize=(30,30))
    fig.subplots_adjust(hspace=.5)
    fig.suptitle('\n\nAveraged Emg Differences Between Weights (Left) and Textures (Right) Across Different Muscle Electrodes On The Arm\n For Participant ' + str(pnum),fontsize = 30)

    i = 0
    for row in ax[:,0]:
        row.plot(emg_list[0].iloc[:,i]) #weight 1
        row.plot(emg_list[2].iloc[:,i],alpha=.5) #weight 3
        row.legend(['165 Grams','660 Grams'],prop={'size': 15},loc="upper left")
        row.set_title(r"$\bf{" + emg_list[0].columns[i] + "}$" + " Amplitudal Differences Between " + r"$\bf{" + 'Weight' + "}$" + " Classes",fontsize = 18)
        row.set_xlabel('Sample',fontsize = 13)
        row.set_ylabel('Amplitude',fontsize = 13)
        i = i + 1

    i = 0
    for row in ax[:,1]:
        row.plot(emg_list[3].iloc[:,i])
        row.plot(emg_list[5].iloc[:,i],alpha=.5)
        row.legend(['Suede','Silk'],prop={'size': 15},loc="upper left")
        row.set_title(r"$\bf{" + emg_list[0].columns[i] + "}$" + " Amplitudal Differences Between " + r"$\bf{" + 'Texture' + "}$" + " Classes",fontsize = 18)
        row.set_xlabel('Sample',fontsize = 13)
        row.set_ylabel('Amplitude',fontsize = 13)
        i = i + 1
    
def plot_eeg_diffs(eeg_list,pnum):
    fig, ax = plt.subplots(nrows=8, ncols=2,figsize=(30,30))
    fig.subplots_adjust(hspace=.5)
    fig.suptitle('\n\nAveraged Eeg Differences Between Weights (Left) and Textures (Right) Across Different Scalp Electrodes\nFor Participant ' + str(pnum),fontsize = 30)
    
    electrodes = ['FC1','FC2','C3','Cz','C4','CP1','CP2','FC5']

    i = 0
    for row in ax[:,0]:
        row.plot(eeg_list[0][electrodes[i]]) #weight 1
        row.plot(eeg_list[2][electrodes[i]]) #weight 3
        row.legend(['165 Grams','660 Grams'],prop={'size': 15},loc="upper left")
        row.set_title(r"$\bf{" + electrodes[i] + "}$" + " Amplitudal Differences Between " + r"$\bf{" + 'Weight' + "}$" + " Classes",fontsize = 18)
        row.set_xlabel('Sample',fontsize = 13)
        row.set_ylabel('Amplitude',fontsize = 13)
        i = i + 1

    i = 0
    for row in ax[:,1]:
        row.plot(eeg_list[3][electrodes[i]])  #3
        row.plot(eeg_list[5][electrodes[i]])  #5
        row.legend(['Suede','Silk'],prop={'size': 15},loc="upper left")
        row.set_title(r"$\bf{" + electrodes[i] + "}$" + " Amplitudal Differences Between " + r"$\bf{" + 'Texture' + "}$" + " Classes",fontsize = 18)
        row.set_xlabel('Sample',fontsize = 13)
        row.set_ylabel('Amplitude',fontsize = 13)
        i = i + 1
        

def plot_eeg_psd_weights(participant): 
    #plotting Cz and C4
    
    cz = 'Cz'
    c4 = 'C4'
    
    eeg_list,_ = nfuncs.return_eeg_emg_averages(data,participant)
    
    X1 = np.fft.fft(np.nan_to_num(eeg_list[0][cz]))
    freq1 = np.fft.fftfreq(n = len(eeg_list[0][cz]),d=1/500)

    X2 = np.fft.fft(np.nan_to_num(eeg_list[2][cz]))
    freq2 = np.fft.fftfreq(n = len(eeg_list[2][cz]),d=1/500)
    
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(20,5))
    fig.subplots_adjust(hspace=.5)
    fig.suptitle('Eeg PSD Differences Between Weight Classifications of Cz Electrode (Left) and C4 (Right) ',fontsize=15)

    ax[0].semilogy(freq1,np.abs(X1)**2) #weight 1
    ax[0].semilogy(freq2,np.abs(X2)**2,alpha=.5) #weight 3
    ax[0].set_xlim([0,250])
    ax[0].set_xlabel('Frequency (Hz)',fontsize = 12)
    ax[0].set_ylabel('Power Spectral Density (dB/Hz)',fontsize=12)
    ax[0].legend(['165 Grams','660 Grams'],prop={'size': 12},loc="upper right")
    
    X1 = np.fft.fft(np.nan_to_num(eeg_list[0][c4]))
    freq1 = np.fft.fftfreq(n = len(eeg_list[0][c4]),d=1/500)

    X2 = np.fft.fft(np.nan_to_num(eeg_list[2][c4]))
    freq2 = np.fft.fftfreq(n = len(eeg_list[2][c4]),d=1/500)

    ax[1].semilogy(freq1,np.abs(X1)**2) #weight 1
    ax[1].semilogy(freq2,np.abs(X2)**2,alpha=.5) #weight 3
    ax[1].set_xlim([0,250])
    ax[1].set_xlabel('Frequency (Hz)',fontsize = 12)
    ax[1].set_ylabel('Power Spectral Density (dB/Hz)',fontsize=12)
    ax[1].legend(['165 Grams','660 Grams'],prop={'size': 12},loc="upper right")
    
def plot_eeg_psd_textures(participant): 
    #plotting Cz and C4
    
    cz = 'Cz'
    c4 = 'C4'
    
    eeg_list,_ = nfuncs.return_eeg_emg_averages(data,participant)
    
    X1 = np.fft.fft(np.nan_to_num(eeg_list[3][cz]))
    freq1 = np.fft.fftfreq(n = len(eeg_list[3][cz]),d=1/500)

    X2 = np.fft.fft(np.nan_to_num(eeg_list[5][cz]))
    freq2 = np.fft.fftfreq(n = len(eeg_list[5][cz]),d=1/500)
    
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(20,5))
    fig.subplots_adjust(hspace=.5)
    fig.suptitle('Eeg PSD Differences Between Texture Classifications of Cz Electrode (Left) and C4 (Right) ',fontsize=15)

    ax[0].semilogy(freq1,np.abs(X1)**2) #weight 1
    ax[0].semilogy(freq2,np.abs(X2)**2,alpha=.5) #weight 3
    ax[0].set_xlim([0,250])
    ax[0].set_xlabel('Frequency (Hz)',fontsize = 12)
    ax[0].set_ylabel('Power Spectral Density (dB/Hz)',fontsize=12)
    ax[0].legend(['Suede','Silk'],prop={'size': 12},loc="upper right")
    
    X1 = np.fft.fft(np.nan_to_num(eeg_list[3][c4]))
    freq1 = np.fft.fftfreq(n = len(eeg_list[3][c4]),d=1/500)

    X2 = np.fft.fft(np.nan_to_num(eeg_list[5][c4]))
    freq2 = np.fft.fftfreq(n = len(eeg_list[5][c4]),d=1/500)

    ax[1].semilogy(freq1,np.abs(X1)**2) #weight 1
    ax[1].semilogy(freq2,np.abs(X2)**2,alpha=.5) #weight 3
    ax[1].set_xlim([0,250])
    ax[1].set_xlabel('Frequency (Hz)',fontsize = 12)
    ax[1].set_ylabel('Power Spectral Density (dB/Hz)',fontsize=12)
    ax[1].legend(['Suede','Silk'],prop={'size': 12},loc="upper right")
