# -*- coding: utf-8 -*-


#%%=================origional code =============
"""
Example from https://batchloaf.wordpress.com/2014/01/17/
real-time-analysis-of-data-from-biosemi-activetwo-via-tcpip-using-python/

However, this does not work, because we need a buffer that constantly store data
"""

#
# test_plot.py - Written by Jack Keegan
# Last updated 16-1-2014
#
# This short Python program receives data from the
# BioSemi ActiveTwo acquisition system via TCP/IP.
#
# Each packet received contains 16 3-byte samples
# for each of 8 channels. The 3 bytes in each sample
# arrive in reverse order (least significant byte first)
#
# Samples for all 8 channels are interleaved in the packet.
# For example, the first 24 bytes in the packet store
# the first 3-byte sample for all 8 channels. Only channel
# 1 is used here - all other channels are discarded.
#
# The total packet size is 8 x 16 x 3 = 384.
# (That's channels x samples x bytes-per-sample)
#
# 512 samples are accumulated from 32 packets.
# A DFT is calculated using numpy's fft function.
# the first DFT sample is set to 0 because the DC
# component will otherwise dominates the plot.
# The real part of the DFT (all 512 samples) is plotted.
# That process is repeated 50 times - the same
# matplotlib window is updated each time.
#
 
import numpy                     # Used to calculate DFT
import matplotlib.pyplot as plt  # Used to plot DFT
import socket                    # used for TCP/IP communication
  
# TCP/IP setup
TCP_IP = '127.0.0.1' # ActiView is running on the same PC
TCP_PORT = 778       # This is the port ActiView listens on
BUFFER_SIZE = 384    # Data packet size (depends on ActiView settings)
 
# Open socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))
 
# Create a 512-sample signal_buffer (arange fills the array with
# values, but these will be overwritten; We're just using arange
# to give us an array of the right size and type).
signal_buffer = numpy.arange(512)
 
# Calculate spectrum 50 times
for i in range(50):
    # Parse incoming frame data
    print("Parsing data")
     
    # Data buffer index (counts from 0 up to 512)
    buffer_idx = 0
     
    # collect 32 packets to fill the window
    for n in range(32):
        # Read the next packet from the network
        ## data is a string, hexadecimal numbers
        ## ord: return integer ordinal of a one-character string
        data = s.recv(BUFFER_SIZE)
         
        # Extract 16 channel 1 samples from the packet
        for m in range(16):
            offset = m * 3 * 8
            # The 3 bytes of each sample arrive in reverse order
            sample = (ord(data[offset+2]) << 16)
            sample += (ord(data[offset+1]) << 8)
            sample += ord(data[offset])
            # Store sample to signal buffer
            signal_buffer[buffer_idx] = sample
            buffer_idx += 1
     
    # Calculate DFT ("sp" stands for spectrum)
    sp = numpy.fft.fft(signal_buffer)
    sp[0] = 0 # eliminate DC component
     
    # Plot spectrum
    print("Plotting data")
    plt.plot(sp.real)
    plt.hold(False)
    plt.show()
 
# Close socket
s.close()

#%%===================additional testing with fieldtrip =========================
# Field trip was downloaded to C:\ExperimentData\YingYang\Real_time

# to start the Biosemi acquisition using the pre-compiled files 
"""
usage of the software
biosemi2ft <config-file> <gdf-file> <hostname> <port>

in Windows cmd
cd C:\ExperimentData\YingYang\Real_time
#C:\FieldTrip\fieldtrip-20160810\realtime\bin\win32\biosemi2ft C:\ExperimentData\YingYang\Real_time\Initial_test_script\Biosemi_fieldtrip_config_example.txt outputfile - 1972
C:\FieldTrip\fieldtrip-20160810\realtime\bin\win32\biosemi2ft C:\ExperimentData\YingYang\Real_time\Initial_test_script\biosemi2ft_128_channel_config.txt outputfile - 1972
# - replaces the <hostname>, and spawns its own buffer server on the given port
# for my testing 20160810, only setting it to "-" works, otherwise, it raised 
# error  "could not connect to buffer server at localhost:1972"
"""

# following python code
# modfied from
#http://www.fieldtriptoolbox.org/development/realtime/buffer_python

import sys
path0 = 'C:\\ExperimentData\\YingYang\\Real_time\\fieldtrip-20160810\\realtime\\src\\buffer\\python\\'
sys.path.insert(0,path0)
import FieldTrip
 
ftc = FieldTrip.Client()		
ftc.connect('localhost', 1972)    # might throw IOError
H = ftc.getHeader()
if H is None:
    print 'Failed to retrieve header!'
    sys.exit(1)
 
print H
print H.labels
if H.nSamples > 0:
    print 'Trying to read last sample...'
    index = H.nSamples - 1
    D = ftc.getData([index, index])
    print D
 
if H.nEvents > 0:
    print 'Trying to read (all) events...'
    E = ftc.getEvents()
    for e in E:
        print e 
ftc.disconnect()

#%% ==================== test if MNE works ================
"""
According to the MNE source code (mne-python/mne/realtime/fieldtrip_client.py, 
Line 17),the FieldTripClient class imports the original 
FieldTrip.Client class in FieldTrip module. 
MNE has an external copy of the FieldTrip.py file. 
"""

import sys
path0 = 'C:\\ExperimentData\\YingYang\\tools\\mne-python-master\\'
sys.path.insert(0,path0)

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.realtime import FieldTripClient, RtEpochs

# the Fieldtrip module in mne external package works
import mne.externals.FieldTrip as mne_copy_FieldTrip
ftc_mne = mne_copy_FieldTrip.Client()		
ftc_mne.connect('localhost', 1972)    # might throw IOError
print ftc_mne.getHeader().nSamples
ftc_mne.getHeader().labels
ftc_mne.disconnect()


# the FieldTripClient class in MNE initializes and connects only in the 
#__enter__ function

with FieldTripClient(host='localhost', port=1972,
                     tmax=150, wait_max=10) as rt_client:
    a = rt_client.get_data_as_epoch()


# debugging, simulate the with clause
rt_client = FieldTripClient(info = None, host = "localhost", port = 1972, 
                            tmax = 150, wait_max = 10)
rt_client.__enter__()
print rt_client.ft_client.getHeader().labels
n_sample = rt_client.ft_client.getHeader().nSamples
tmp_data = rt_client.ft_client.getData([n_sample-50000, n_sample])

# debugging: last 280 were all of it
tmp_data1 = tmp_data[:,-280::]
plt.figure()
plt.plot(tmp_data1.mean(axis = 0),'*-')
plt.figure()
plt.plot(tmp_data1.std(axis = 0),'*-')
plt.figure()  
plt.plot(tmp_data1[:,277])
plt.figure()
plt.plot(tmp_data1[:,278::])    
# it seemed that the 280 channels do not contain STI/status channels. 
# the trigger has to be obtain from the ft_client.getEvents()

           
# this line ony works if in the config file, 1=EEG_channel_name
# EEG must be the prefix, otherwise MNE can not handle it
raw_info = rt_client.get_measurement_info()        
# not working? why? When picks = None, the matrix dimension is wrong, this bug fixed with the manually updated source code from github. I might have to update it in the conda later
a = rt_client.get_data_as_epoch(n_samples = 100, picks = None)
## source code for get_data_as_epoch copied and modified here
n_samples =100
ft_header = rt_client.ft_client.getHeader()
last_samp = ft_header.nSamples - 1
start = last_samp - n_samples + 1
stop = last_samp
# one single event
events = np.expand_dims(np.array([start, 1, 1]), axis=0)
# get the data
data = rt_client.ft_client.getData([start, stop]).transpose()
# create epoch from data
info = rt_client.info
picks = range(1)
picks = None
#epoch = mne.EpochsArray(data[picks][np.newaxis], info, events)
epoch = mne.EpochsArray(data[np.newaxis, :,:], info, events)


## source code for getting the info
# not working, channel names do not start with EEG, after chaning the channel names, it worked. 
import re
from mne.io.constants import FIFF 
info = mne.io._empty_info(rt_client.ft_header.fSample)
info['comps'] = list()
info['projs'] = list()
info['bads'] = list()
info['chs'] = []
for idx, ch in enumerate(rt_client.ft_header.labels):
    this_info = dict() 
    this_info['scanno'] = idx
    # extract numerical part of channel name
    this_info['logno'] = int(re.findall('[^\W\d_]+|\d+', ch)[-1])
    if ch.startswith('EEG'):
        this_info['kind'] = FIFF.FIFFV_EEG_CH
    elif ch.startswith('MEG'):
        this_info['kind'] = FIFF.FIFFV_MEG_CH
    elif ch.startswith('MCG'):
        this_info['kind'] = FIFF.FIFFV_MCG_CH
    elif ch.startswith('EOG'):
        this_info['kind'] = FIFF.FIFFV_EOG_CH
    elif ch.startswith('EMG'):
        this_info['kind'] = FIFF.FIFFV_EMG_CH
    elif ch.startswith('STI'):
        this_info['kind'] = FIFF.FIFFV_STIM_CH
    elif ch.startswith('ECG'):
        this_info['kind'] = FIFF.FIFFV_ECG_CH
    elif ch.startswith('MISC'):
        this_info['kind'] = FIFF.FIFFV_MISC_CH
    # Fieldtrip already does calibration
    this_info['range'] = 1.0
    this_info['cal'] = 1.0
    this_info['ch_name'] = ch
    this_info['loc'] = None
    if ch.startswith('EEG'):
        this_info['coord_frame'] = FIFF.FIFFV_COORD_HEAD
    elif ch.startswith('MEG'):
        this_info['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
    else:
        this_info['coord_frame'] = FIFF.FIFFV_COORD_UNKNOWN
    if ch.startswith('MEG') and ch.endswith('1'):
        this_info['unit'] = FIFF.FIFF_UNIT_T
    elif ch.startswith('MEG') and (ch.endswith('2') or
                                   ch.endswith('3')):
        this_info['unit'] = FIFF.FIFF_UNIT_T_M
    else:
        this_info['unit'] = FIFF.FIFF_UNIT_V
    this_info['unit_mul'] = 0
    #print info
    print idx
    info['chs'].append(this_info)
    print info['chs'][idx]['kind']
    info._update_redundant()
    info._check_consistency()    
    
 

for idx in range(136):
    print info['chs'][idx]['kind']
rt_client.__exit__(1,1,1)





event_id = 0
tmin,tmax = -0.5,0.5
rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax,
                         stim_channel='status', picks=picks,
                         reject= None,
                         decim=1, isi_max=10.0, proj=None)


'''
event_id, tmin, tmax = 1, -0.2, 0.5  # select the left-auditory condition
# user must provide list of bad channels because
# FieldTrip header object does not provide that
bads = ['MEG 2443', 'EEG 053']
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
'''