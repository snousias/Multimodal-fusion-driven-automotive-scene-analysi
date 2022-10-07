## CarlaSimulatorKitti

> Pygame CARLA Simulator to produce KITTI 2D/3D object detection for dataset generation to be deployed to OpenPCDet

> Modified for compatibility with OpenPCDet

### Disclaimer
This repository was initially available at https://github.com/mesakh123/CarlaSimulatorKitti

The code was mofidied for compatibility with OpenPCDet to include planes and folder structure Velodyne 64 LIDAR Properties

**Source Codes**
```
https://github.com/mesakh123/CarlaSimulatorKitti 
https://github.com/mmmmaomao/DataGenerator
```

### Specifics


**Folder Format**

```
|-- dataset
    |-- training
    |   |-- calib/ # camera and lidar coeff
    |   |-- image_2/ # RGB image
    |   |-- label_2/ # KITTI format image information
    |   |-- velodyne/ 
    |   |-- planes/
    |   |-- train.txt
    |   |-- trainval.txt
    |   |-- val.txt

```

**KITTI label format**

```
 Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car','Pedestrian',
   					 'TrafficSigns', etc.
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
```

**label type**
Two types of labels:

1.  Actors : Car, Pedestrian
2.  Environment : None，Buildings，Fences，Other，Pedestrians，Poles，RoadLine，Roads，Sidewalks，TrafficSigns，Vegetation，Vehicles，Walls，Sky，Ground，Bridge，RailTrack，GuardRail，TrafficLight，Static，Dynamic，Water，Terrain

**Usage**

Carla Version：carla 0.9.13

Collecting Data

```
python3 generator.py
```

Only show pygame

```
python3 inference.py --loop
```

