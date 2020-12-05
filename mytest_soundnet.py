import torch
import torch.nn as nn
import numpy as np
from util import load_from_txt
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
local_config = {
	'batch_size': 1,
	'eps': 1e-5,
	'sample_rate': 22050,
	'load_size': 22050 * 20,
	'name_scope': 'SoundNet_TF',
	'phase': 'extract',
}
#load model and weights
from soundnet2 import SoundNet8_pytorch
model = SoundNet8_pytorch()
model.load_state_dict(torch.load("sound8.pth"))
#summary model
from torchsummaryX import summary
# summary(model, torch.zeros(1,1,22050 * 20,1))
#load data and extract feature
audio_txt = os.path.abspath(os.path.join(BASE_DIR,'mydata', "audio_files.txt"))#path of audio_files.txt
sound_samples, audio_paths = load_from_txt(audio_txt, config=local_config)
for idx, sound_sample in enumerate(sound_samples):
	print(audio_paths[idx])
	new_sample = torch.from_numpy(sound_sample)
	output = model.forward(new_sample)
	#classification
	softmax = nn.Softmax(dim =1)
	id_obj = torch.max(softmax(output[0]),1)
	id_scn = torch.max(softmax(output[1]),1)
	print('#####objects class: %s'%torch.squeeze(id_obj[1]))
	print('#####places class: %s'%torch.squeeze(id_scn[1]))
	#average poolling
	avgpool_layer = nn.AvgPool2d((4,1))
	avgpool_obj = avgpool_layer(softmax(output[0]))
	#tensor-->ndarry
	a_feature = avgpool_obj.detach()
	a_feature = np.array(a_feature)
	#save as '.mat'
	import scipy.io as io
	audio_name = audio_paths[idx].split('.')[0]
	io.savemat('%s.mat' % audio_name,{'a_feature':a_feature})
	print('#####a_feature is saved in %s.mat' % audio_name)

