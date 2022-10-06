import subprocess
import os
import glob
import json
from pathlib import Path

config = json.load(open('config.json', ))
config = config['multimodalv2']
for i, k in enumerate(config):
    if k not in ['root',
                 'path_to_image',
                 'path_to_image_right',
                 'path_to_lidar',
                 'path_to_labels',
                 'path_to_calibration_for_tracking',
                 'path_to_groundtruth_for_tracking'] and isinstance(config[k], str):
        config[k] = config['root'] + config[k]

kitti_eval_tool_path = os.path.join(Path(config['root']),'fusion/kitti-eval/cpp/evaluate_object')
dump_folder_path=config['save_path_root']


gtlist=glob.glob(os.path.join(dump_folder_path, 'groundtruth','label_2')+"/*.txt")
gtlist=[v.split('/')[-1] for v in gtlist]
gtlist=[v.split('.')[0] for v in gtlist]

for prediction_folder in ['prediction_image','prediction_fusion','prediction_lidar']:

    predlist=glob.glob(os.path.join(dump_folder_path, prediction_folder,'data')+"/*.txt")
    predlist=[v.split('/')[-1] for v in predlist]
    predlist=[v.split('.')[0] for v in predlist]




    with open(dump_folder_path+'/'+'val.txt',mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(predlist))

    plotfolderpath=os.path.join(dump_folder_path, prediction_folder,'plot')
    if os.path.exists(plotfolderpath):
        for file in os.scandir(plotfolderpath):
            os.remove(file.path)
        os.rmdir(plotfolderpath)

    cmd = '{} {} {} {} {}'.format(kitti_eval_tool_path,
                                  os.path.join(dump_folder_path, 'groundtruth'),
                                  dump_folder_path+'/'+'val.txt',
                                  os.path.join(dump_folder_path, prediction_folder),
                                  len(predlist))

    status = subprocess.call(cmd, shell=True)