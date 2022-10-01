import os
import random

root="F:/_Datasets/"
root_dataset=root
kittipath=root_dataset+"KITTI/ObjectDetection/"
imagesPath=kittipath+"training/image_2"


fs=os.listdir(imagesPath)
for i,f in enumerate(fs):
    fs[i]=os.path.splitext(f)[0]




f=open('./image_sets/trainVal.txt','w')
for ele in fs:
    f.write(imagesPath +"/"+ele+'\n')
f.close()


fs=random.sample(fs, len(fs))

f=open('./image_sets/train.txt','w')
for ele in fs[:int(0.8*len(fs))]:
    f.write(ele+'\n')
f.close()



f=open('./image_sets/val.txt','w')
for ele in fs[int(0.8*len(fs)):]:
    f.write(ele+'\n')
f.close()


print("ok")