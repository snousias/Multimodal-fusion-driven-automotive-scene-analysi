import mayavi
import numpy as np
import mayavi.mlab

def viz_mayavi(points, vals="distance"):
    x = points[:, 0]  # x position of point
    y = points[:, 1]  # y position of point
    z = points[:, 2]  # z position of point
    r = points[:, 3]  # reflectance value of point
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
    # Plot using mayavi -Much faster and smoother than matplotlib
    if vals == "height":
        col = z
    else:
        col = d
    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    mayavi.mlab.points3d(x, y, z,
                         col,  # Values used for Color
                         mode="point",
                         colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                         # color=(0, 0, 0),   # Used a fixed (r,g,b) instead
                         figure=fig,
                         )
    mayavi.mlab.show()


fnm = '/home/stavros/Workspace/Automotive/OpenPCDet/dataset_generate/CarlaSimulatorKitti/dataset/training/velodyne/000000.bin'
points = np.fromfile(fnm, dtype=np.float32).reshape((-1, 4)).astype(np.float32)
print(np.shape(points)[0])
viz_mayavi(points)
print("OK")
