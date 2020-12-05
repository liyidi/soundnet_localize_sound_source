from network import *
from Sound_Localization_Dataset import *
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
# load model
net = AVModel()
net.load_state_dict(torch.load("sound_localization_latest.pth"))
# load data
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
val_dataset_file = os.path.abspath(os.path.join(BASE_DIR,'mydata', "mytest_list.txt"))
dataset_test = Sound_Localization_Dataset(val_dataset_file)
dataloader_test = DataLoader(dataset_test, batch_size = 1, shuffle = False)
#validation
result = []
for i, mydata in enumerate(dataloader_test):
    print('Eval step:', i,mydata[3])
    frame_t_val = mydata[0]
    pos_audio_val = mydata[1]
    neg_audio_val = mydata[2]
    z_val, pos_audio_embedding_val, neg_audio_embedding_val, att_map_val = net.forward(frame_t_val, pos_audio_val, neg_audio_val)
    att_map = att_map_val.view(20,20)
    import cv2
    b = att_map.detach().numpy()
    norm_image = cv2.normalize(b, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    result.append(norm_image)
    from torchvision import transforms
    unloader = transforms.ToPILImage()
    image_map = unloader(norm_image)
    sns_plot = sns.heatmap(image_map, cmap='Spectral_r')
    plt.show()
    plt.pause(0.5)
    plt.close()

#show heatmap
for j in range(0,len(result)):
    ds = open(val_dataset_file)
    lines = ds.readlines()
    datum = lines[j]
    video_path = datum.replace('\n', '')
    frames_path = video_path + '.jpg'
    frame = cv2.imread(frames_path)
    map1 = result[j]
    remap1 = cv2.resize(map1,(256,256))
    heatmap = np.uint8(remap1)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    frame_map = frame*0.5+heatmap*0.5
    frame_map = frame_map.astype(np.uint8)
    plt.imshow(frame_map[:,:,::-1])
    plt.show()
    cv2.imwrite('demo.png',frame_map)
