
# coding: utf-8

# In[7]:


import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.signal import stft
from scipy import signal
import pickle
import re
import datetime
import shutil


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
dir_name_to_change=os.path.dirname(dname)
os.chdir(dir_name_to_change)
np.set_printoptions(precision=4,suppress=True)


# In[2]:

def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

source_dir='./datasets'
target_dir_stft_pickles='./stft'
try:
    shutil.copytree(source_dir, target_dir_stft_pickles, ignore=ig_f)
except:
    pass


'''
To prepare for this code the full indented directory structure must be present for stft
If this is not present, run the following

def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]
    
shutil.copytree(datasets, stft, ignore=ig_f)
'''


# In[3]:


def get_stft(fileName,targetFileName=None):
    if targetFileName is None:
        targetFileName=re.sub('.wav','.pkl',re.sub('datasets','stft',fileName))
    print ("Processing filename"+str(fileName)+" to "+str(targetFileName))
    if not os.path.isfile(targetFileName):
        myAudio = fileName
        samplingFreq, mySound = wavfile.read(myAudio)
        mySound = mySound / (2.**15)
        signal_stft=signal.stft(mySound,samplingFreq,'hann')
        with open(targetFileName,'wb') as f:
            pickle.dump(signal_stft,f)
        return signal_stft
    else:
        return 0


# In[4]:


list_of_files=[]
for root, _, filenames in os.walk(source_dir):
     for filename in filenames:
         if filename.split('.')[-1]=='wav':
             list_of_files.append(os.path.join(root, filename))

if os.name=='nt':
    list_of_files=[re.sub('\\\\','/',x) for x in list_of_files] 
    
print (os.getcwd())

# In[5]:


len(list_of_files)


# In[8]:


counter=0
for each_file in list_of_files:
    if (counter%100==0):
        print ('file '+ str(counter)+' : '+each_file)
        print (datetime.datetime.now())
    try:
        _=get_stft(each_file)
    except Exception as e:
        print (str(e))
    counter=counter+1