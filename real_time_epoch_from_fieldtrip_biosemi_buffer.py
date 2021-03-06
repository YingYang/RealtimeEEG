# -*- coding: utf-8 -*-
# Author Ying Yang,  yingyan1@andrew.cmu.edu
# implement the an MNE-rtEpoch-type object for the fieldtrip biosemi buffer 
# I need to rewrite many parts in mne-python fieldtrip client, and RtEpochs to make it work. 
# To be continued

# Fieldtrip buffer: multithresd C/C++, can be cleaned (flushed) 
# ring buffer: after a time, old data samples and events will not be accessible anymore
# http://www.fieldtriptoolbox.org/development/realtime/buffer_protocol
# In the buffer:  head structure, data matrix, a list of event structures
# 

#============ specific to the EEG aquisition machine
import sys
paths = ['C:\\ExperimentData\\YingYang\\tools\\mne-python-master\\',
         'C:\\ExperimentData\\YingYang\\Real_time\\fieldtrip-20160810\\realtime\\src\\buffer\\python\\']
for path0 in paths:
    sys.path.insert(0,path0)

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.realtime import FieldTripClient


class biosemi_fieldtrip_recent_epochs:
	
     def __init__(self,rt_client, n_recent_event = 1, trial_tmin = 0.2, 
                  trial_tmax = 1.0, event_types = [1]):
	self.rt_client = rt_client
        self.trial_tmin = trial_tmin
	self.tiral_tmax = trial_tmax
	self.n_recent_event = n_recent_event
	self.event_types = event_types

	self.__raw_info = self.__rt_client.get_measurement_info()
	sampling_rate = self.__raw_info['sfreq']
	self.time_step = 1.0/sampling_rate
    	# number of samples after the stimulus onset
    	self.n_sample_per_trial_pos = np.round(trial_tmax/time_step)
    	# number of samples before onset
    	self.n_sample_per_trial_neg = np.round(trial_tmin/time_step)
	self.n_times = self.n_sample_per_trial_pos + self.n_sample_per_trial_neg
	self.n_channel = len(self.raw_info['chs'])

    
    def get_recent_epochs(self):
	ftc = self.rt_client.ft_client()
	# get the latest n events, n = n_recent_event
	H = ftc.getHeader()
	if H is None:
	 raise ValueError('Failed to retrieve header!')
	# find the events
	recent_event_sample_number = []
	recent_event_counter = 0
	events_list = ftc.getEvents()
	current_nSamples = H.nSamples

	for e in events_list[-1]:
	# get event type and event sample
	print e 
	current_event_ind = 
	current_event_type = 
	if current_event_type in event_types \\
	    and current_event_ind+n_sample_per_trial_pos <= current_nSamples
	    events_list.append((current_event_ind,current_event_type))
	    recent_event_counter+= 1

	# create epochs using the data

	recent_data = np.zeros([recent_event_counter, n_channels, n_times])
	for ind, (current_event_ind, current_event_type) enumerate(events_list.reverse()):
	recent_data[ind] = ftc.getData([current_event_ind-n_sample_trial_neg,
		                current_event_ind+n_sample_trial_pos])
	recent_epochs = mne.EpochArray(recent_data, raw_info, 
		    tmin = trial_tmin, baseline = None, proj = False)
	
	return recent_epochs, events_list






if __name__=="__main__":
	with FieldTripClient(host='localhost', port=1972,
		             tmax=150, wait_max=10) as rt_client:
	    recent_epochs_obj = biosemi_fieldtrip_recent_epochs(rt_client)
	    recent_epochs, events_list = recent_epochs_obj.get_recent_epochs()

    
 

        plt.pause(0.05)
        plt.draw()
    plt.close()
'''
