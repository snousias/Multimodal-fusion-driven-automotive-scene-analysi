import os
import random

root="F:/_Datasets/"
root_dataset=root
kittipath=root_dataset+"KITTI/ObjectDetection/"

imagesPath=kittipath+"training/image_2"

fs=os.listdir(imagesPath)
for i,f in enumerate(fs):
    fs[i]=os.path.splitext(f)[0]

f=open('image_sets/images.txt','w')
for ele in fs:
    f.write(kittipath+'training/image_2/'+ele+'\n')
f.close()

f=open('image_sets/labels.txt','w')
for ele in fs:
    f.write(kittipath+'training/label_2/'+ele+'\n')
f.close()





print("ok")