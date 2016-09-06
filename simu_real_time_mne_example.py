# -*- coding: utf-8 -*-
"""
MNE-python "FieldTripClient"
guesses measurement info from the Fieldtrip Header object
class mne.realtime.FieldTripClient(info=None, host='localhost', 
 port=1972, wait_max=30, tmin=None, tmax=inf, buffer_size=1000, verbose=None)
 
"RtEpochs", realtime epochs object
class mne.realtime.RtEpochs(client, event_id, tmin, tmax, 
stim_channel='STI 014', sleep_time=0.1, baseline=(None, 0), 
picks=None, name='Unknown', reject=None, flat=None, proj=True, 
decim=1, reject_tmin=None, reject_tmax=None, detrend=None, add_eeg_ref=True, 
isi_max=2.0, find_events=None, verbose=None)

"""

import matplotlib.pyplot as plt
import mne
from mne.viz import plot_events
from mne.realtime import FieldTripClient, RtEpochs, MockRtClient

import numpy as np
from mne.datasets import sample
from sklearn import preprocessing  
from sklearn.svm import SVC  
from sklearn.pipeline import Pipeline  
from sklearn.cross_validation import cross_val_score, ShuffleSplit  
from mne.decoding import FilterEstimator 
# modified from original example because of version mismatch
if mne.__version__ == "0.12.dev0":
    from mne.decoding.transformer import EpochsVectorizer as Vectorizer
else:
    from mne.decoding import Vectorizer


#%%======== only works when a Fieldtrip buffer is on
event_id, tmin, tmax = 1, -0.2, 0.5  # select the left-auditory condition
# user must provide list of bad channels because
# FieldTrip header object does not provide that
bads = ['MEG 2443', 'EEG 053']
plt.ion()  # make plot interactive, i.e. update the plot with every command
_, ax = plt.subplots(2, 1, figsize=(8, 8))  # create subplots
# with Func as Obj   create the obj, when the job is done or failed, delete it
with FieldTripClient(host='localhost', port=1972,
                     tmax=150, wait_max=10) as rt_client:
    # get measurement info guessed by MNE-Python
    raw_info = rt_client.get_measurement_info()
    # select gradiometers
    picks = mne.pick_types(raw_info, meg='grad', eeg=False, eog=True,
                           stim=True, exclude=bads)
    # create the real-time epochs object
    rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax,
                         stim_channel='STI 014', picks=picks,
                         reject=dict(grad=4000e-13, eog=150e-6),
                         decim=1, isi_max=10.0, proj=None)
    # start the acquisition
    rt_epochs.start()
    for ii, ev in enumerate(rt_epochs.iter_evoked()):
        print("Just got epoch %d" % (ii + 1))
        ev.pick_types(meg=True, eog=False)
        if ii == 0:
            evoked = ev
        else:
            evoked += ev
        ax[0].cla()
        ax[1].cla()  # clear axis
        plot_events(rt_epochs.events[-5:], sfreq=ev.info['sfreq'],
                    first_samp=-rt_client.tmin_samp, axes=ax[0])
        evoked.plot(axes=ax[1])  # plot on second subplot
        ax[1].set_title('Evoked response for gradiometer channels'
                        '(event_id = %d)' % event_id)
        plt.pause(0.05)
        plt.draw()
    plt.close()
#%%========== simulation with a mock Client
# Fiff file to simulate the realtime client
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)

tr_percent = 60  # Training percentage
min_trials = 10  # minimum trials after which decoding should start
# select gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                       stim=True, exclude=raw.info['bads'])
# create the mock-client object
rt_client = MockRtClient(raw)
# create the real-time epochs object
rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax, picks=picks, decim=1,
                     reject=dict(grad=4000e-13, eog=150e-6))
# start the acquisition
rt_epochs.start()
# send raw buffers
rt_client.send_data(rt_epochs, picks, tmin=0, tmax=200, buffer_size=1000)
n_times = len(rt_epochs.times)


scores_x, scores, std_scores = [], [], []
# define a simple linear svm classifier
filt = FilterEstimator(rt_epochs.info, 1, 40)
scaler = preprocessing.StandardScaler()
vectorizer = Vectorizer()
clf = SVC(C=1, kernel='linear')
concat_classifier = Pipeline([('filter', filt), ('vector', vectorizer),
                              ('scaler', scaler), ('svm', clf)])
data_picks = mne.pick_types(rt_epochs.info, meg='grad', eeg=False, eog=True,
                            stim=False, exclude=raw.info['bads'])
for ev_num, ev in enumerate(rt_epochs.iter_evoked()):
    print("Just got epoch %d" % (ev_num + 1))
    print rt_epochs.get_data().shape
    print ev.data.shape
    if ev_num == 0:
        X = ev.data[None, data_picks, :]
        y = int(ev.comment)  # the comment attribute contains the event_id
    else:
        X = np.concatenate((X, ev.data[None, data_picks, :]), axis=0)
        y = np.append(y, int(ev.comment))
    if ev_num >= min_trials:
        cv = ShuffleSplit(len(y), 5, test_size=0.2, random_state=42)
        scores_t = cross_val_score(concat_classifier, X, y, cv=cv,
                                   n_jobs=1) * 100
        std_scores.append(scores_t.std())
        scores.append(scores_t.mean())
        scores_x.append(ev_num)

        # Plot accuracy
        plt.clf()
        plt.plot(scores_x, scores, '+', label="Classif. score")
        plt.hold(True)
        plt.plot(scores_x, scores)
        plt.axhline(50, color='k', linestyle='--', label="Chance level")
        # these stds are wrong, do not include them
        #hyp_limits = (np.asarray(scores) - np.asarray(std_scores),
        #              np.asarray(scores) + np.asarray(std_scores))
        #plt.fill_between(scores_x, hyp_limits[0], y2=hyp_limits[1],
        #                color='b', alpha=0.5)
        plt.xlabel('Trials')
        plt.ylabel('cv accuracy on cumulateive data')
        plt.xlim([min_trials, 50])
        plt.ylim([30, 105])
        plt.title('Real-time decoding')
        plt.show(block=False)
        plt.pause(0.01)
plt.show()
rt_epochs.stop()   
    
    
#========== MNE had a full example of real-time neurofeedback, with pyschopy

    