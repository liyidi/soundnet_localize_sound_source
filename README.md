# soundnet_localize_sound_source
This is a Pytorch implementation of the paper 【Learning to Localize Sound Source in Visual Scenes】 and soundnet, which only includes the verification part and does not include the training process.

	1. download the model
sound8.pth:
https://drive.google.com/file/d/1-PhHutIYV9Oi2DhDZL2h1Myu84oGLI81/view

sound_localization_latest.pth:
https://drive.google.com/file/d/1JMD-LjHbfZ_yUy-l6tjbI46yYQfH8oS4/view

	2. run 'mytest_soundnet.py' to generate audio feature:
Load the pretrained model from "sound8.pth", and edit the audio data list in BASE_DIR/mydata/audio_files.txt. 

id_obj and id_scn is the classification result of the object and the scene.

After the softmax and average pooling on the output of "Object" branch of Conv8 layer, the output is saved as '.mat' file. This is the audio feature of the next architecture.

	3. run 'mytest_avnet.py' to localize sound source in visual scenes:
Load the pretrained model from "sound_localization_latest.pth", and edit the data list in BASE_DIR/mydata/mytest_list.txt. 

'Mydata' folder should contain frames as .jpg and audio features as .mat.

Result shows the attention map.

	4.refrence: 
Senocak A, Oh T H, Kim J, et al. Learning to localize sound source in visual scenes[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 4358-4366.

Yusuf Aytar, Carl Vondrick, and Antonio Torralba. "Soundnet: Learning sound representations from unlabeled video." Advances in Neural Information Processing Systems. 2016.

