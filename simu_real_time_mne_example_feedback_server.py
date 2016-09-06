"""
==============================================
Real-time feedback for decoding :: Server Side
==============================================

This example demonstrates how to setup a real-time feedback
mechanism using StimServer and StimClient.

The idea here is to display future stimuli for the class which
is predicted less accurately. This allows on-demand adaptation
of the stimuli depending on the needs of the classifier.

To run this example, open ipython in two separate terminals.
In the first, run rt_feedback_server.py and then wait for the
message

    RtServer: Start

Once that appears, run rt_feedback_client.py in the other terminal
and the feedback script should start.

All brain responses are simulated from a fiff file to make it easy
to test. However, it should be possible to adapt this script
for a real experiment.

"""
# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

import mne
from mne.datasets import sample
from mne.realtime import StimServer
from mne.realtime import MockRtClient
from mne.decoding import FilterEstimator
# modified from original example because of version mismatch
if mne.__version__ == "0.12.dev0":
    from mne.decoding.transformer import EpochsVectorizer as Vectorizer
else:
    from mne.decoding import Vectorizer

print(__doc__)

# Load fiff file to simulate data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)

# Instantiating stimulation server

# The with statement is necessary to ensure a clean exit
with StimServer(port=4218) as stim_server:

    # The channels to be used while decoding
    picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                           stim=True, exclude=raw.info['bads'])
    rt_client = MockRtClient(raw)
    # Constructing the pipeline for classification
    filt = FilterEstimator(raw.info, 1, 40)
    scaler = preprocessing.StandardScaler()
    vectorizer = Vectorizer()
    clf = SVC(C=1, kernel='linear')
    concat_classifier = Pipeline([('filter', filt), ('vector', vectorizer),
                                  ('scaler', scaler), ('svm', clf)])
    stim_server.start(verbose=True)
    # Just some initially decided events to be simulated
    # Rest will decided on the fly
    ev_list = [4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4]
    score_c1, score_c2, score_x = [], [], []
    for ii in range(50):
        # Tell the stim_client about the next stimuli
        stim_server.add_trigger(ev_list[ii])
        # Collecting data
        if ii == 0:
            X = rt_client.get_event_data(event_id=ev_list[ii], tmin=-0.2,
                                         tmax=0.5, picks=picks,
                                         stim_channel='STI 014')[None, ...]
            y = ev_list[ii]
        else:
            X_temp = rt_client.get_event_data(event_id=ev_list[ii], tmin=-0.2,
                                              tmax=0.5, picks=picks,
                                              stim_channel='STI 014')
            X_temp = X_temp[np.newaxis, ...]
            X = np.concatenate((X, X_temp), axis=0)
            time.sleep(1)  # simulating the isi
            y = np.append(y, ev_list[ii])

        # Start decoding after collecting sufficient data
        if ii >= 10:
            # Now start doing rtfeedback
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=0.2,
                                                                random_state=7)
            # Ying: the filtering or vectorization seem to cause error, 
            # use just a single svm classifier instead                                                    
            #y_pred = concat_classifier.fit(X_train, y_train).predict(X_test)            

            y_pred = clf.fit(X_train.reshape([X_train.shape[0],-1]), y_train)\
                      .predict(X_test.reshape([X_test.shape[0],-1]))

            # Ying: if y_pred == y_test and only a single class was used, confusion matrix fails
            # labels here are hard coded
            tmp_score_c1 = (y_pred ==  y_test)[y_pred == 3].mean()
            tmp_score_c2 = (y_pred ==  y_test)[y_pred == 4].mean()
            #cm = confusion_matrix(y_test, y_pred)
            #score_c1.append(float(cm[0, 0]) / sum(cm, 1)[0] * 100)
            #score_c2.append(float(cm[1, 1]) / sum(cm, 1)[1] * 100)
            score_c1.append(tmp_score_c1)
            score_c2.append(tmp_score_c2)
          
            # do something if one class is decoded better than the other
            if score_c1[-1] < score_c2[-1]:
                print("We decoded class RV better than class LV")
                ev_list.append(3)  # adding more LV to future simulated data
            else:
                print("We decoded class LV better than class RV")
                ev_list.append(4)  # adding more RV to future simulated data

            # Clear the figure
            plt.clf()
            # The x-axis for the plot
            score_x.append(ii)
            # Now plot the accuracy
            plt.plot(score_x, score_c1)
            plt.hold(True)
            plt.plot(score_x, score_c2)
            plt.xlabel('Trials')
            plt.ylabel('Classification score (% correct)')
            plt.title('Real-time feedback')
            plt.ylim([0, 1.5])
            plt.legend(('LV', 'RV'), loc='upper left')
            plt.show()
