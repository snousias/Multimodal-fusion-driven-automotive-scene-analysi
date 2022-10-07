import numpy as np
import os
import glob
import random
from random import sample
import argparse

datasetdir = ''

parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--dataset_dir', type=str, default='', help='specify the data directory')

args = parser.parse_args()

datasetdir=args.dataset_dir

if not os.path.exists(datasetdir):
    print("Error: Directory does not exist")
    exit(2)

random.seed(10)

os.chdir(datasetdir + "/training/label_2")
g = glob.glob("*.txt")
g.sort()
for gfile in g:
    fpath = datasetdir + "/training/label_2/" + gfile
    mname = gfile.split(sep=".")[0]
    gf = os.path.getsize(fpath)
    if gf == 0:
        filePath = datasetdir + "/training/" + "calib/" + mname + ".txt"
        if os.path.exists(filePath):
            os.remove(filePath)
        filePath = datasetdir + "/training/" + "image_2/" + mname + ".png"
        if os.path.exists(filePath):
            os.remove(filePath)
        filePath = datasetdir + "/training/" + "label_2/" + mname + ".txt"
        if os.path.exists(filePath):
            os.remove(filePath)
        filePath = datasetdir + "/training/" + "planes/" + mname + ".txt"
        if os.path.exists(filePath):
            os.remove(filePath)
        dfilePath = datasetdir + "/training/" + "velodyne/" + mname + ".bin"
        if os.path.exists(filePath):
            os.remove(filePath)
        print("ok")

os.chdir(datasetdir + "/training/planes")
g = glob.glob("*.txt")
g.sort()
for idx, gfile in enumerate(g):
    fpath = datasetdir + "/training/velodyne/" + gfile
    mname = gfile.split(sep=".")[0]

    if True:
        filePathCalib = datasetdir + "/training/" + "calib/" + mname + ".txt"
        filePathImage = datasetdir + "/training/" + "image_2/" + mname + ".png"
        filePathLabel = datasetdir + "/training/" + "label_2/" + mname + ".txt"
        filePathVelo = datasetdir + "/training/" + "velodyne/" + mname + ".bin"
        filePathPlanes = datasetdir + "/training/" + "planes/" + mname + ".txt"

        # Read in the file
        # with open(filePathLabel, 'r') as file:
        #     filedata = file.read()
        # # Replace the target string
        # filedata = filedata.replace('Bicycle', 'Cyclist')
        # # Write the file out again
        # with open(filePathLabel, 'w') as file:
        #     file.write(filedata)

        with open(filePathLabel, "r") as f:
            lines = f.readlines()
        with open(filePathLabel, "w") as f:
            for line in lines:
                if "None" in line:
                    print("None found")
                else:
                    f.write(line)

        doDelete = True
        # with open(filePathLabel) as f:
        #     if ('Cyclist' in f.read()):
        #         doDelete = False
        #         f.seek(0)
        #         f.close()
        with open(filePathLabel) as f:
            if ('Car' in f.read()):
                doDelete = False
                f.seek(0)
                f.close()
        with open(filePathLabel) as f:
            if ('Pedestrian' in f.read()):
                doDelete = False
                f.seek(0)
                f.close()

        if doDelete:
            if os.path.exists(filePathCalib):
                os.remove(filePathCalib)

            if os.path.exists(filePathImage):
                os.remove(filePathImage)

            if os.path.exists(filePathLabel):
                os.remove(filePathLabel)

            if os.path.exists(filePathVelo):
                os.remove(filePathVelo)

            if os.path.exists(filePathPlanes):
                os.remove(filePathPlanes)

        if ((not os.path.exists(filePathCalib)) | (not os.path.exists(filePathImage)) | (
                not os.path.exists(filePathLabel)) | \
                (not os.path.exists(filePathVelo)) | (not os.path.exists(filePathPlanes))):

            if os.path.exists(filePathCalib):
                os.remove(filePathCalib)

            if os.path.exists(filePathImage):
                os.remove(filePathImage)

            if os.path.exists(filePathLabel):
                os.remove(filePathLabel)

            if os.path.exists(filePathVelo):
                os.remove(filePathVelo)

            if os.path.exists(filePathPlanes):
                os.remove(filePathPlanes)

g = glob.glob("*.txt")
g.sort()

for idx, gfile in enumerate(g):
    fpath = datasetdir + "/training/velodyne/" + gfile
    mname = gfile.split(sep=".")[0]
    filePathCalib = datasetdir + "/training/" + "calib/" + mname + ".txt"
    filePathImage = datasetdir + "/training/" + "image_2/" + mname + ".png"
    filePathLabel = datasetdir + "/training/" + "label_2/" + mname + ".txt"
    filePathVelo = datasetdir + "/training/" + "velodyne/" + mname + ".bin"
    filePathPlanes = datasetdir + "/training/" + "planes/" + mname + ".txt"

    filePathCalibNew = datasetdir + "/training/" + "calib/" + '{0:06d}'.format(idx) + ".txt"
    filePathImageNew = datasetdir + "/training/" + "image_2/" + '{0:06d}'.format(idx) + ".png"
    filePathLabelNew = datasetdir + "/training/" + "label_2/" + '{0:06d}'.format(idx) + ".txt"
    filePathVeloNew = datasetdir + "/training/" + "velodyne/" + '{0:06d}'.format(idx) + ".bin"
    filePathPlanesNew = datasetdir + "/training/" + "planes/" + '{0:06d}'.format(idx) + ".txt"

    os.rename(filePathCalib, filePathCalibNew)
    os.rename(filePathImage, filePathImageNew)
    os.rename(filePathLabel, filePathLabelNew)
    os.rename(filePathVelo, filePathVeloNew)
    os.rename(filePathPlanes, filePathPlanesNew)
