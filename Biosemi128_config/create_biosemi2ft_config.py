"""
Run this file under linux (my laptop or desktop)
"""
import numpy as np
file_dir = "/home/ying/Dropbox/RealTimeEEG/Biosemi128_config/"
# read the input file name and save the sensor name
input_fname = file_dir + "1020-system128+8+sens.cfg"


channel_names = list()
count = -1
with open(input_fname,"r") as fid:
    # count from label on, for every line, record the line number and channel name    
    tmp_count = 0
    for line in fid:
        tmp = line
        tmp_count+=1
        if tmp_count >400:
            break
        if tmp[0:8] == "[Labels]":
            count = 0
            print "start counting"
            continue   
        if tmp[0] != "[" and count >= 0:
            tmp1 = tmp.split("=")
            channel_names.append(tmp1[-1][0:-2]) 
            print tmp1
            count +=1
            print "count = %d" %count
        elif tmp[0] == "[" and count >0 :
            break

id1 = np.nonzero(np.array(channel_names) == "Ol1h" )[0][0]
id2 = np.nonzero(np.array(channel_names) == "Ol2h" )[0][0]
channel_names[id1] = "OI1h"
channel_names[id2] = "OI2h"


# add EEG as suffix
for i in range(256):
    channel_names[i] = "EEG_%s" % channel_names[i]
    
# modifie other channel names, HARD CODED!!! EXG1-EXG8, masteroids, EOGs and ECG
EXG_names = ['MISC_M1','MISC_M2','EOG_LO1','EOG_LO2','EOG_IO1', 'EOG_SO1', 'EOG_IO2', 'ECG',]
channel_names[256: (256+8)] = EXG_names
selected_channels = np.hstack([np.arange(128),np.arange(256,256+8)])
# Trigger or STI channel is saved in the FieldTrip buffer with type "TRIGGER"


# line endings in windows is "\r\n"
le = "\r\n"

downsample = 8
bandwidth = 50 # lowpass filter butterworth
bworder = 4   # order of the bw filter
statusrefresh = 4
batteryrefresh = 20

outdir = "/home/ying/Dropbox/RealTimeEEG/"
out_fname = outdir + "biosemi2ft_128_channel_config.txt"
with open(out_fname, "w") as fid:
    fid.write("[select]%s" %le)
    for i in range(len(selected_channels)):
        fid.write("%d=%s_%d%s" %(selected_channels[i]+1,
            channel_names[selected_channels[i]],selected_channels[i]+1,le))
    ##==debugging testing, understand all the channels
    ##==include the additional 280-256-8= 16 channels
    #for i in range(280):
    #    fid.write("%d=%s%d%s" %(i+1, "extra", i+1, le))
    #fid.write(le)
    #fid.write("downsample %d%s" %(downsample,le)) 
    #fid.write("bandwith %s" %(bandwidth,le)) 
    #fid.write("bworder %s" %(bworder,le)) 
    fid.write("statusrefresh %d%s" %(statusrefresh,le)) 
    fid.write("batteryrefresh %d%s" %(batteryrefresh,le)) 
    
                   
        
    


                       
                       
                       
                      