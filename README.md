# soundnet_localize_sound_source
This is a Pytorch implementation of the paper 【Learning to Localize Sound Source in Visual Scenes】 and soundnet, which only includes the verification part and does not include the training process.

	1. run 'mytest_soundnet.py' to generate audio feature:
Load the pretrained model from "sound8.pth", and edit the audio data list in BASE_DIR/mydata/audio_files.txt. 

id_obj and id_scn is the classification result of the object and the scene.

After the softmax and average pooling on the output of "Object" branch of Conv8 layer, the output is saved as '.mat' file. This is the audio feature of the next architecture.

	2. run 'mytest_avnet.py' to localize sound source in visual scenes:
Load the pretrained model from "sound_localization_latest.pth", and edit the data list in BASE_DIR/mydata/mytest_list.txt. 

'Mydata' folder should contain frames as .jpg and audio features as .mat.

Result shows the attention map.
