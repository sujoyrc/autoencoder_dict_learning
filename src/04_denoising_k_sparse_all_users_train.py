import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.signal import stft,istft
from scipy import signal
import IPython
import pickle
import datetime
from math import ceil
import configparser as cp
import random
import glob
import os
import fnmatch
import re
import random
from math import ceil,floor
import warnings

import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug

warnings.filterwarnings("ignore")

class K_Sparse_AutoEncoder:
    '''
    This is a class to do sparse coding using back-propagation.
    This is based on theory in 
    Makhzani, Alireza, and Brendan Frey. "K-sparse autoencoders." arXiv preprint arXiv:1312.5663 (2013).
    available at https://arxiv.org/pdf/1312.5663.pdf 
    '''
    def __init__(self,padded_im,padded_mag,input_matrix_2D_list,sparse_length_list,k_sparse_list,train_batch_size=10,test_batch_size=10,N_epochs=250,validation_after_epochs=10,n_samples_train=50):
        self.sparse_length_list=sparse_length_list
        self.train_results={}
        self.test_results={}
        self.padded_im=padded_im
        self.padded_mag=padded_mag
        self.input_matrix_2D_list=input_matrix_2D_list
        self.train_set=[]
        self.val_set=[]
        self.N_epochs=N_epochs           # Early stopping not yet implemented
        self.VALIDATION_AFTER_EPOCHS=validation_after_epochs
        self.train_batch_size=train_batch_size
        self.test_batch_size=test_batch_size
        self.n_samples_train=n_samples_train
        self.n_samples_test=len(self.input_matrix_2D_list)-self.n_samples_train  # Whatever is not in train is in test
        self.model_results={}
        self.batch = 50
        self.signal_dim =  list(input_matrix_2D_list[0].shape)
        self.k_sparse_list = k_sparse_list

    def train_model(self,sparse_length,k_sparse,debug_level=0):
        print ("Running with sparse_length="+str(sparse_length)+" and k_sparse="+str(k_sparse))
        # This is a bit untidy. Can be cleaned up later
        batch = 50
        signal_dim =  self.signal_dim
        #
        if debug_level>0: print (signal_dim)
        model_components={}
        x = tf.placeholder(tf.float32, [None,signal_dim[0],signal_dim[1]])
        batch_size = tf.placeholder(tf.int32)
        if debug_level>0: print (x.shape)
        #
        W = tf.Variable(tf.truncated_normal([signal_dim[1],sparse_length],stddev=1e-1), name='weights')
        b = tf.Variable(tf.constant(0.0, shape=[sparse_length], dtype=tf.float32),trainable=True, name='biases')
        x_2d = tf.reshape(x,[-1,signal_dim[1]])
        z = tf.matmul(x_2d,W) + b
        if debug_level>0: print (W.shape,b.shape,x_2d.shape,z.shape)
        #
        tao,tao_indices = tf.nn.top_k(z,k=k_sparse,sorted=True)
        indices_range = tf.expand_dims(tf.range(0,batch*signal_dim[0] ), 1) 
        range_repeated = tf.tile(indices_range, [1, k_sparse]) 
        if debug_level>0:   print(tao,tao_indices,indices_range,range_repeated)  
        full_indices = tf.concat([tf.expand_dims(range_repeated, 2), tf.expand_dims(tao_indices, 2)], axis=2)
        full_indices = tf.reshape(full_indices, [-1, 2])
        mask=tf.ones(tf.shape(full_indices)[0])
        #mask = tf.SparseTensor(tf.ones(tf.shape(full_indices)[0]),dense_shape=tf.constant([signal_dim[0]*batch,sparse_length]))
        tao_mask = tf.sparse_to_dense(full_indices,tf.constant([signal_dim[0]*batch,sparse_length]), mask, validate_indices=False)
        #tao_mask = tf.sparse_to_dense(full_indices,tf.constant([signal_dim[0]*batch,sparse_length]), mask)
        #
        z_tao = tf.multiply(tao_mask,z)
        #
        b_dash = tf.Variable(tf.constant(0.0, shape=[signal_dim[1]], dtype=tf.float32),trainable=True, name='biases')
        x_recons = tf.matmul(z_tao,tf.transpose(W)) + b_dash
        #
        error = tf.losses.mean_squared_error(x_2d,x_recons)
        #
        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(error)
        #
        sess = tf.Session()
        #if debug_level>0:   sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())
        #
        for e in range(1,self.N_epochs):
            sess.run(train_step,feed_dict={x:self.train_set,batch_size:self.train_batch_size})   #### inp should be replaced by batch

            if e%self.VALIDATION_AFTER_EPOCHS == 0 :
                err = sess.run(error,feed_dict={x:self.val_set,batch_size:self.test_batch_size})   ### inp should be replaced by val_set
                print("Epoch ",e," has val error : ", err)
        #
        x_recons_data_train=sess.run(x_recons,feed_dict={x:self.train_set})
        x_recons_train_list=np.split(x_recons_data_train,self.n_samples_test,axis=0)
        #
        x_recons_data_val=sess.run(x_recons,feed_dict={x:self.val_set})
        x_recons_test_list=np.split(x_recons_data_val,self.n_samples_test,axis=0)
        #
        model_components['W']=sess.run(W)
        model_components['b']=sess.run(b)
        model_components['tao']=sess.run(tao,feed_dict={x:self.val_set})
        model_components['tao_indices']=sess.run(tao_indices,feed_dict={x:self.val_set})
        model_components['b_dash']=sess.run(b_dash)
        model_components['err']=err
        model_components['x_recons_train_list']=x_recons_train_list
        model_components['x_recons_test_list']=x_recons_test_list
        return model_components
        
    def train_test_split(self):
        train_indices = np.random.choice(len(self.input_matrix_2D_list), self.n_samples_train, replace=False).tolist()
        val_indices=[x for x in range(len(self.input_matrix_2D_list)) if x not in train_indices]
        indices_dict={}
        indices_dict['train_indices']=train_indices
        indices_dict['val_indices']=val_indices
        #
        self.train_set = np.array([self.input_matrix_2D_list[i] for i in train_indices])
        self.val_set = np.array([self.input_matrix_2D_list[i] for i in val_indices])
        return indices_dict

    def hyperparameter_search(self):
        for each_sparse_length in self.sparse_length_list:
            for each_k_sparse in self.k_sparse_list:
                self.model_results[(each_sparse_length,each_k_sparse)]=self.train_model(each_sparse_length,each_k_sparse,debug_level=1)
        return self.model_results

def get_input_matrix2D_per_user(userId,list_of_stft_files):
    #print (list_of_stft_files)
    #print (userId)
    files_this_user=[x for x in list_of_stft_files if (x.split('/')[-2])==userId][0:100]
    print (len(files_this_user))
    signal_stft_list=[]
    for each_file in files_this_user:
        with open(each_file,'rb') as f:
            signal_stft_info=pickle.load(f)
            #print (signal_stft_info[0].shape,signal_stft_info[1].shape)
        signal_stft_list.append(signal_stft_info[2])
    signal_stft_list_mag=[np.real(x) for x in signal_stft_list]
    signal_stft_im=[np.imag(x) for x in signal_stft_list]
    max_time=max([x.shape[1] for x in signal_stft_list])
    freq_bins=max([x.shape[0] for x in signal_stft_list])
    padded_mag=[np.pad(signal_stft_list_mag[i]\
                          ,((0,0),(0,max_time-signal_stft_list_mag[i].shape[1])),mode='constant',constant_values=0)\
                   for i in range(len(signal_stft_list_mag))]
    padded_im=[np.pad(signal_stft_im[i]\
                          ,((0,0),(0,max_time-signal_stft_im[i].shape[1])),mode='constant',constant_values=0)\
                   for i in range(len(signal_stft_im))]
    input_matrix=[padded_mag[i] for i in range(len(padded_mag))]
    return input_matrix,max_time,freq_bins,padded_mag,padded_im,signal_stft_list_mag,signal_stft_im,files_this_user

def reconstruct_signals(list_of_reconstructed,padded_im,signal_stft_list,indices):
    reconstructed_dict={}
    for i in range(len(indices)):
        position_of_signal=indices[i]
        real_component=list_of_reconstructed[i]
        imaginary_component=padded_im[position_of_signal]
        actual_signal_stft=signal_stft_list[position_of_signal]
        actual_signal_stft_cols=actual_signal_stft.shape[1]
        signal_stft_padded=real_component+1j*imaginary_component
        signal_stft_same_size=signal_stft_padded[:,0:actual_signal_stft_cols]
        original_signal=istft(actual_signal_stft,samplingFreq,'hann')
        reconstructed_signal=istft(signal_stft_same_size,samplingFreq,'hann')
        squared_error=np.sum((original_signal[1]-reconstructed_signal[1])**2)
        num_samples=original_signal[1].shape[0]
        mean_squared_error=squared_error/num_samples
        power_signal=sum([p**2 for p in original_signal[1]])/num_samples
        power_noise=sum([p**2 for p in reconstructed_signal[1]])/num_samples
        db=10*np.log10(power_signal/power_noise)
        reconstructed_dict[position_of_signal]={}
        reconstructed_dict[position_of_signal]['original_signal']=original_signal
        reconstructed_dict[position_of_signal]['reconstructed_signal']=reconstructed_signal
        reconstructed_dict[position_of_signal]['mse']=mean_squared_error
        reconstructed_dict[position_of_signal]['snr']=db
    return reconstructed_dict

#os.chdir('C:\\SujoyRc\\Personal\\JUEE\\twoUserModel')
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
dir_name_to_change=os.path.dirname(dname)
os.chdir(dir_name_to_change)
np.set_printoptions(precision=4,suppress=True)
samplingFreq=25000
#print (os.getcwd())
list_of_speakers=[str(x) for x in range(1,35)]


for each_user in list_of_speakers[6:]:
    user_model_results={}
    print ("############")
    print ("Processing user "+each_user)
    stft_directory_clean='stft\\single_speaker\\'+each_user
    #print(stft_directory_clean)
    list_of_stft_files=[]
    for root, subdirs, files in os.walk(stft_directory_clean):
        for each_file in files:
            if re.search('pkl',each_file):
                file_name=root+'/'+each_file
                if os.path.isfile(file_name):
                    list_of_stft_files.append(file_name)
                    
    #print (list_of_stft_files)
    if os.name=='nt':
        list_of_stft_files=[re.sub('\\\\','/',x) for x in list_of_stft_files] # Replace \\ in Windows by /
        
    signal_stft_list=[]
    for each_file in list_of_stft_files:
        with open(each_file,'rb') as f:
            signal_stft_info=pickle.load(f)
            #print (signal_stft_info[0].shape,signal_stft_info[1].shape)
        signal_stft_list.append(signal_stft_info[2])
    #print ([x.shape for x in signal_stft_list])    
    list_of_users=list(set([x.split('/')[-2] for x in list_of_stft_files]))
    #list_of_users=['s1']
    all_noisy_audio_dict={}
    all_clean_audio_dict={}
    all_denoised_audio_dict={}

    #for each_user in list_of_users:

    this_user=each_user
    input_matrix_2D_list,max_time,freq_bins,padded_mag,padded_im,signal_stft_list_mag,signal_stft_im,files_this_user=get_input_matrix2D_per_user(this_user,list_of_stft_files)
    #print (len(input_matrix_2D_list))
    start_sparse_length=(floor(max_time/50)+1)*50
    sparse_length_list=[start_sparse_length,start_sparse_length+50,start_sparse_length+100]
    k_sparse_list=[10,25,50,100]
    k_sparse_autoencoder=K_Sparse_AutoEncoder(padded_im,padded_mag,input_matrix_2D_list,sparse_length_list,k_sparse_list)
    indices_dict=k_sparse_autoencoder.train_test_split()
    user_model_results[each_user]=k_sparse_autoencoder.hyperparameter_search()    
    for each_model in list(user_model_results[each_user].keys()):
        signals_this_model=reconstruct_signals(user_model_results[each_user][each_model]['x_recons_test_list'],padded_im,signal_stft_list,indices_dict['val_indices'])
        user_model_results[each_user][each_model]['signals']=signals_this_model
    model_snr_mse={}
    list_of_models=list(user_model_results[each_user].keys())
    for each_model in list_of_models:
        average_mse=np.mean(np.array([user_model_results[each_user][each_model]['signals'][k]['mse'] for k in list(user_model_results[each_user][each_model]['signals'].keys())]))
        average_snr=np.mean(np.array([user_model_results[each_user][each_model]['signals'][k]['snr'] for k in list(user_model_results[each_user][each_model]['signals'].keys())]))
        model_snr_mse[each_model]=[average_mse,average_snr]

    model_directory='./models/s'+each_user+'/'
    model_snr_mse_file=model_directory+each_user+'_model_snr_mse.pkl'
    user_model_results_file=model_directory+each_user+'_user_model_results.pkl'
    with open(user_model_results_file,'wb') as f:
        pickle.dump(user_model_results[each_user],f)
    with open(model_snr_mse_file,'wb') as f:
        pickle.dump(model_snr_mse,f)

# user_model_results_all_file='./models/'+'user_model_results_all.pkl'
# with open(user_model_results_all_file,'wb') as f:
#     pickle.dump(user_model_results,f)