#!/usr/bin/env python3
 
from __future__ import division, print_function, absolute_import
from os import listdir
from os.path import isfile
from os.path import join
import numpy as np
import librosa
import tflearn
import tensorflow as tf
 
model_dir = 'model'
model_name = 'keyword.model'
data_dir = 'data'
test_dir = 'test'
keyword_dir = 'keyword'
not_keyword_dir = 'not-keyword'
length_ext = '.len'
 
learning_rate = 0.0001
num_features = 20
lstm_size = 16
lstm_dropout = 0.8

def load_length():
	len_file = join(model_dir, model_name + length_ext)
	if isfile(len_file):
		print("Found length file.")
		with open(len_file) as f:
			for line in f:
				if len(line) > 0:
					return int(line)
	return None

def save_length(length):
	len_file = join(model_dir, model_name + length_ext)
	with open(len_file, 'w') as f:
		f.write(str(length))

def save_model(model, length):
	model.save(join(model_dir, model_name))
	save_length(length)

def _load_mfcc(filename):
	wave, sr = librosa.load(filename, mono=True)
	return librosa.feature.mfcc(wave, sr, n_mfcc=num_features)
	
def load_mfccs(path):
	return [_load_mfcc(join(path, f)) for f in listdir(path)]
	
def max_length_mfccs(mfccs):
	max_length = 0
	for i in mfccs:
		width, length = i.shape
		if length > max_length:
			max_length = length
	return max_length

def normalize_mfccs(mfccs, max_length):
	for i in range(len(mfccs)):
		width, length = mfccs[i].shape
		if length <= max_length:
			padding_dim = ((0, 0), (0, max_length - length))
			mfccs[i] = np.pad(mfccs[i], padding_dim, mode='constant', constant_values=0)
		else:
			mfccs[i] = np.split(mfccs[i], [max_length], axis=1)[0]
	
def create_net(in_sx, in_sy, out_sx):
	"""
	Creates a tflearn neural network with the correct
	architecture for learning to hear the keyword
	"""
	net = tflearn.input_data([None, in_sx, in_sy])
	net = tflearn.lstm(net, lstm_size, dropout=lstm_dropout)
	net = tflearn.fully_connected(net, out_sx, activation='softmax')
	net = tflearn.regression(net, learning_rate=learning_rate, optimizer='adam', loss='categorical_crossentropy')
	return net

def create_model(net):
	return tflearn.DNN(net, tensorboard_verbose=0)

def train_model(model, inputs, outputs, n_epoch):
	model.fit(inputs, outputs, n_epoch=n_epoch, validation_set=0.1, show_metric=True, 
			batch_size=len(inputs))

def fix_version_errors():
	col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	for x in col:
		tf.add_to_collection(tf.GraphKeys.VARIABLES, x)

def try_load_into_model(model):
	"""
	Returns True if found and loaded model
	Returns False otherwise
	"""
	model_path = join(model_dir, model_name)
	if isfile(model_path + '.index'):
		print('Loading saved model...')
		model.load(model_path)
		return True
	return False

def _load_data(parent_dir, max_length=None):
	"""
	Provide max_length to force audio files to be chopped
	or extended in the case of an already trained network
	"""
	mfccs = []
	outputs = []
	
	def load_subdir(subdir, output):
		nonlocal mfccs, outputs
		path = join(parent_dir, subdir)
		subdir_mfccs = load_mfccs(path)
		
		mfccs += subdir_mfccs
		outputs += [np.array(output)] * len(subdir_mfccs)
	
	load_subdir(keyword_dir, [1, 0])
	load_subdir(not_keyword_dir, [0, 1])
	
	if max_length is None:
		max_length = max_length_mfccs(mfccs)
	normalize_mfccs(mfccs, max_length)
	
	return mfccs, outputs

def load_training_data(max_length=None):
	return _load_data(data_dir, max_length)

def load_test_data(max_length=None):
	return _load_data(join(data_dir, test_dir), max_length)

