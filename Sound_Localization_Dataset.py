import os,glob,json
import torch
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional
from PIL import Image
from random import randint
from tqdm import tqdm
import time
import random
from random import choice
import math
import pdb
import scipy
import scipy.io as sio



def localization_gt_loader(sample, annotation_path):
	video_path = sample.replace('\n','')
	words = [word.replace('\n','') for word in video_path.split('/')]
	video_name = words[-1][:-4]
	path = annotation_path+'/'+video_name+'.mat'
	worker_gt = np.zeros((1,400))
	w_a = 1
	w_m = 1
	if os.path.exists((path)):
		gt_file = sio.loadmat(path)
		gt = (gt_file['gt_box20'])
		n_workers = gt.shape[2]
		worker = randint(0,n_workers-1)
		worker_gt_val = gt[:,:,worker]
		worker_gt = np.reshape(worker_gt_val,(1,400))
	else:
		w_a = 0
	weights = np.zeros((2))
	weights[0] = w_a
	weights[1] = w_m
	worker_gt_t = torch.from_numpy(worker_gt).view(1,400).float()
	weights_t = torch.from_numpy(weights)
	return worker_gt_t, weights_t


def audio_loader(sample, neg_sample):
	# Get positive audio
	video_path = sample.replace('\n', '')
	audio_path = video_path + '.mat'
	pos_sound_file = sio.loadmat(audio_path)
#modify
	pos_sound = (pos_sound_file['a_feature'])
	pos_sound = np.asarray(pos_sound) # CHECK THE SIZE!
	pos_sound_tensor = torch.from_numpy(pos_sound).squeeze().float()
	# Get negative audio
	neg_video_path = neg_sample.replace('\n','')
	neg_audio_path = neg_video_path+'.mat'
	neg_sound_file = sio.loadmat(neg_audio_path)
	# modify
	neg_sound = (neg_sound_file['a_feature'])
	neg_sound = np.asarray(neg_sound)  # CHECK THE SIZE!
	neg_sound_tensor = torch.from_numpy(neg_sound).squeeze().float()

	return pos_sound_tensor, neg_sound_tensor

def image_loader(sample):
	video_path = sample.replace('\n','')
	all_frames = video_path+'.jpg'
	first_image = Image.open(all_frames).convert('RGB')
	return first_image


class Sound_Localization_Dataset(Dataset):
	def __init__(self, dataset_file):
		ds = open(dataset_file)
		lines = ds.readlines()
		self.data = lines
		self.preprocess = transforms.Compose([transforms.Resize((320,320)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

	def __getitem__(self,index):
		datum = self.data[index]
		# Get negative index and negative sample
		neg_index = choice([r for r in range(0,len(self.data)) if r not in [index]])
		neg_datum = self.data[neg_index]
		# Get video frames
		first_frame = image_loader(datum)
		first_frame_t = self.preprocess(first_frame).float()
		pos_audio_t,neg_audio_t = audio_loader(datum, neg_datum)
		# worker_gt_t, weigths_t = localization_gt_loader(datum,self.annotation_path)
		return first_frame_t, pos_audio_t, neg_audio_t, datum
		# return first_frame_t, pos_audio_t, neg_audio_t, worker_gt_t, weigths_t, datum

	def __len__(self):
		return len(self.data)



