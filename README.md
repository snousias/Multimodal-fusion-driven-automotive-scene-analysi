# Multimodal fusion driven scene understanding
> This repository provides a plugin for the OpenPCDet object detection framework that facilitates 
> fusion of 2D and 3D object detection.

## Instructions

1. Please download, and install all the requirements of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) 
2. Download folder named **"fusion"** into the root directory of OpenPCDet
3. Structure should be the following
```
    |
    |
    |-- data/
    |
    |-- docker/
    |
    |-- docs/
    |
    |-- pcdet/
    |
    |-- fusion/
    |
    |-- tools/
    |
    |-- LICENCE
    |
    |-- README.md
    |
    |-- requirements.txt
    |
    |-- setup.py

```

## Limitations

1. Non reported
2. Successfully tested with commit [b345b08c5d3e49ff82a5374d033ddd2b5e66253e](https://github.com/open-mmlab/OpenPCDet/commit/b345b08c5d3e49ff82a5374d033ddd2b5e66253e) [2022-09-25]


## Requirements

Extra requirements for cropping pdf report in evalution script:
> sudo apt-get install texlive-extra-utils  

## Usage
Configuration file:
> fusion/cfgs_custom/multimodal/config.json

```
{
  "multimodalv2": {
    "root": "/home/<HOME_DIR>/Workspace/Automotive/OpenPCDet/",
    "path_to_data": "data/kitti/training/",
    "path_to_calibration_for_tracking": "calib.txt",
    "path_to_groundtruth_for_tracking": "groundtruth.txt",
    "path_to_image": "image_2/",
    "path_to_image_right": "image_3/",
    "path_to_lidar": "velodyne/",
    "path_to_labels": "label_2/",
    "deeplab_root": "",
    "save_path_root": "fusion/results/dump/",
    "save_path_came": "fusion/results/dump/image/",
    "save_path_image_from_lidar": "fusion/results/dump/image_lidar/",
    "save_path_meta_data": "fusion/results/dump/meta_data/",
    "save_path_lidar": "fusion/results/dump/lidar/",
    "cut_off_percentage": 0.8,
    "cut_off_2D": 0.8,
    "nms_fusion_threshold": 0.5,
    "segmentation_model": "",
    "image_detection_model": "fusion/imagedet/models/squeezedet_kitti_epoch280.pth",
    "lidar_detection_cfg": "fusion/cfgs/kitti_models/pv_rcnn.yaml",
    "lidar_detection_model": "fusion/trained/pv_rcnn_8369.pth",
    "start_frame": 0,
    "denoise": 0,
    "meta_data": 0
  },
  "comment": {

  }
}
```

### Run fusion:
> cd fusion
> 
> python runFusion.py

### Evaluate fusion outcomes,
The script compares fusion with image-only and LIDAR-only detection:
> cd fusion
> 
> python runEvaluate.py

## Results



 
<!--
## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## General Information
- Provide general information about your project here.
- What problem does it (intend to) solve?
- What is the purpose of your project?
- Why did you undertake it?


## Technologies Used
- Tech 1 - version 1.0
- Tech 2 - version 2.0
- Tech 3 - version 3.0


## Features
List the ready features here:
- Awesome feature 1
- Awesome feature 2
- Awesome feature 3


## Screenshots
![Example screenshot](./img/screenshot.png)

## Setup
What are the project requirements/dependencies? Where are they listed? A requirements.txt or a Pipfile.lock file perhaps? Where is it located?

Proceed to describe how to install / setup one's local environment / get started with the project.


## Usage
How does one go about using it?
Provide various use cases and code examples here.

`write-your-code-here`


## Project Status
Project is: _in progress_ / _complete_ / _no longer being worked on_. If you are no longer working on it, provide reasons why.


## Room for Improvement
Include areas you believe need improvement / could be improved. Also add TODOs for future development.

Room for improvement:
- Improvement to be done 1
- Improvement to be done 2

To do:
- Feature to be added 1
- Feature to be added 2


## Acknowledgements
Give credit here.
- This project was inspired by...
- This project was based on [this tutorial](https://www.example.com).
- Many thanks to...


## Contact
Created by [@flynerdpl](https://www.flynerd.pl/) - feel free to contact me!

-->