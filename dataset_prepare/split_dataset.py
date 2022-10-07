import os
import random
import argparse


def check_if_exists_and_create_if_not(folder):
    isExist = os.path.exists(folder)
    if not isExist:
        os.makedirs(folder)
    return


datasetdir = ''
parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--dataset_dir', type=str, default='', help='specify the data directory')
parser.add_argument('--ratio', type=float, default=0.8, help='splitting ratio')
args = parser.parse_args()

ratio = args.ratio
datasetdir = args.dataset_dir

if not os.path.exists(datasetdir):
    print("Error: Directory does not exist")
    exit(2)

check_if_exists_and_create_if_not(datasetdir + 'ImageSets')

imagesPath = datasetdir + "/training/image_2"
fs = os.listdir(imagesPath)
fs.sort()
for i, f in enumerate(fs):
    fs[i] = os.path.splitext(f)[0]

f = open(datasetdir + '/ImageSets/trainVal.txt', 'w')
for ele in fs:
    f.write(ele + '\n')
f.close()
fs = random.sample(fs, len(fs))
f = open(datasetdir + '/ImageSets/train.txt', 'w')
for ele in fs[:int(ratio * len(fs))]:
    f.write(ele + '\n')
f.close()
f = open(datasetdir + '/ImageSets/val.txt', 'w')
for ele in fs[int(ratio * len(fs)):]:
    f.write(ele + '\n')
f.close()

f = open(datasetdir + '/ImageSets/test.txt', 'w')
for ele in fs[int(ratio * len(fs)):]:
    f.write(ele + '\n')
f.close()

print("Complete")