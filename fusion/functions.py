import argparse
import glob
from pathlib import Path
import os
import numpy as np
import cv2
import json
import os
import tqdm
import skimage.io
import torch
import torch.utils.data
from torch.autograd import Variable
from torchvision.ops import nms
import sys
import random
from skimage.metrics import structural_similarity
# from skimage.measure import compare_ssim

import matplotlib.pyplot as plt
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.utils.calibration_kitti import Calibration
from tools.visual_utils import visualize_utils as V
import mayavi.mlab as mlab
import shutil




def generateAnnotationPrediction(metric,classes):
    annotations_pred = {}
    annotations_pred['name'] = np.array([classes[int(x)] for x in metric['classes']])
    annotations_pred['truncated'] = np.array([-1 for x in metric['classes']])
    annotations_pred['occluded'] = np.array([-1 for x in metric['classes']])
    annotations_pred['alpha'] = np.array([0 for x in metric['classes']])
    annotations_pred['bbox'] = np.array(
        [[float(info) for info in x] for x in metric['boxes']]).reshape(-1, 4)
    annotations_pred['dimensions'] = np.array(
        [[0.0, 0.0, 0.0] for x in metric['classes']]).reshape(
        -1, 3)[:, [2, 0, 1]]
    annotations_pred['location'] = np.array(
        [[0.0, 0.0, 0.0] for x in metric['classes']]).reshape(-1, 3)
    annotations_pred['rotation_y'] = np.array(
        [0 for x in metric['classes']]).reshape(-1)
    annotations_pred['score'] = np.array([float(x) for x in metric['scores']])
    return annotations_pred



def writePredictionsToList(input):
    filecontent = []
    for rowindex in range(np.shape(input['name'])[0]):
        line = []
        for key, val in input.items():
            if not isinstance(val[rowindex], (list, tuple, np.ndarray)):
                if isinstance(val[rowindex], str):
                    line.append(str(val[rowindex]).lower())
                elif (key == 'truncated'):
                    line.append(str(int(val[rowindex])))
                elif (key == 'occluded'):
                    line.append(str(int(val[rowindex])))
                elif (key == 'score'):
                    line.append('{:.2f}'.format(val[rowindex]))
                else:
                    line.append(str(int(val[rowindex])))
            else:
                for element in val[rowindex]:
                    line.append('{:.2f}'.format(element))
        lineout = ' '.join(line)
        filecontent.append(lineout)
    return filecontent



def projection_matrix(calibration_file):
    C = Calibration(calibration_file)
    R0_ext = np.hstack((C.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
    R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
    R0_ext[3, 3] = 1
    V2C_ext = np.vstack((C.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
    V2C_ext[3, 3] = 1
    P2_ext = np.vstack((C.P2, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
    P2_ext[3, 3] = 1
    return np.dot(np.dot(P2_ext, R0_ext), V2C_ext)


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


class_colors = (255. * np.array(
    [0.850, 0.325, 0.098,
     0.466, 0.674, 0.188,
     0.098, 0.325, 0.850,
     0.301, 0.745, 0.933,
     0.635, 0.078, 0.184,
     0.300, 0.300, 0.300,
     0.600, 0.600, 0.600,
     1.000, 0.000, 0.000,
     1.000, 0.500, 0.000,
     0.749, 0.749, 0.000,
     0.000, 1.000, 0.000,
     0.000, 0.000, 1.000,
     0.667, 0.000, 1.000,
     0.333, 0.333, 0.000,
     0.333, 0.667, 0.000,
     0.333, 1.000, 0.000,
     0.667, 0.333, 0.000,
     0.667, 0.667, 0.000,
     0.667, 1.000, 0.000,
     1.000, 0.333, 0.000,
     1.000, 0.667, 0.000,
     1.000, 1.000, 0.000,
     0.000, 0.333, 0.500,
     0.000, 0.667, 0.500,
     0.000, 1.000, 0.500]
)).astype(np.uint8).reshape((-1, 3))


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2
    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]
    return corners3d.numpy() if is_numpy else corners3d


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, path_conf=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training,
            root_path=Path(path_conf["path_to_data"] + path_conf["path_to_lidar"]), logger=logger
        )
        self.root_path = Path(path_conf["path_to_data"] + path_conf["path_to_lidar"])
        self.ext = ext
        data_file_list = glob.glob(str(self.root_path / f'*{self.ext}')) if self.root_path.is_dir() else [
            self.root_path]
        data_file_list.sort()
        # keep from frame starting frame to end
        self.sample_file_list = data_file_list[path_conf["start_frame"]:]
        self.image_path = path_conf["path_to_data"] + path_conf["path_to_image"]
        # self.image_path_right = path_conf["path_to_data"] + path_conf["path_to_image_right"]
        # self.at_image_path = path_conf["path_to_data"] + path_conf["path_to_attacked_image"]
        # self.de_image_path = path_conf["path_to_data"] + path_conf["save_denoised_image"]
        # self.lane_segmentation_path = path_conf["path_to_data"] +"laneSegmentation/"
        self.names = os.listdir(self.image_path)
        self.names.sort()
        self.names = self.names[path_conf["start_frame"]:]

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        lidar_name = \
            (self.sample_file_list[index].split("/")[len(self.sample_file_list[index].split("/")) - 1]).split('.')[0]
        img_pth_left = self.image_path + lidar_name + '.png'
        img_left = cv2.imread(img_pth_left, -1)
        # img_right = cv2.imread(self.image_path_right + lidar_name + '.png', -1)
        # img_left_lane_segmentation_path=self.lane_segmentation_path+ lidar_name + "_laneSegmentation"+'.png'
        # img_left_lane_segmentation=cv2.imread(img_left_lane_segmentation_path, -1)
        # cimg = skimage.io.imread(img_pth).astype(np.float32)
        # at_image = cv2.imread(self.at_image_path + lidar_name + '.png', -1)
        # de_image = cv2.imread(self.de_image_path + lidar_name + '.png', -1)

        input_dict = {
            'points': points,
            'frame_id': index,
            'image': img_left,
            # 'image_lane_segmentation':img_left_lane_segmentation,
            # 'image_right': img_right
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict




def detect_in_lidar(data_dict, model):
    load_data_to_gpu(data_dict)
    pred_dicts, _ = model.forward(data_dict)
    return pred_dicts


def get3DBBoxProjection(pred_dicts, points_with_reflectance, projection_mat, cutoff=None):
    ref_boxes = pred_dicts[0]['pred_boxes']
    ref_scores = pred_dicts[0]['pred_scores']
    ref_labels = pred_dicts[0]['pred_labels']
    ref_corners3d = []

    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = ref_boxes.cpu().numpy()
    if ref_boxes is not None and len(ref_boxes) > 0:
        ref_corners3d = boxes_to_corners_3d(ref_boxes)
    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        ref_scores = ref_scores.cpu().numpy()
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.cpu().numpy()

    predictions = []
    for i, box in enumerate(ref_corners3d):
        if len(ref_scores) > 0:
            if ref_scores[i] < cutoff: continue
            corner_lines = []
            skip = 0

            """
                7 -------- 4
               /|         /|
              6 -------- 5 .
              | |        | |
              . 3 -------- 0
              |/         |/
              2 -------- 1
            """
            boxcenter_3D = np.mean(box, axis=0)
            boxcenter_projected = np.dot(projection_mat, np.append(boxcenter_3D, 1))
            boxcenter_projected[0] = int(boxcenter_projected[0] / boxcenter_projected[2])
            boxcenter_projected[1] = int(boxcenter_projected[1] / boxcenter_projected[2])
            boxcenter_projected = boxcenter_projected.astype(int)
            boxcenter_projected = boxcenter_projected[:2]
            for corner in box:
                cr = np.append(corner, 1)
                image_points = np.dot(projection_mat, cr)
                image_points[0] = image_points[0] / image_points[2]
                image_points[1] = image_points[1] / image_points[2]
                if image_points[2] < 2:
                    skip = 1
                center_coordinates = (int(image_points[0]), int(image_points[1]))
                corner_lines.append(center_coordinates)
            if skip == 1:
                continue
            x, y = np.array(corner_lines).T
            corner_lines = corner_lines
            x = np.array(x)
            y = np.array(y)
            box2d = {'minx': min(x), 'miny': min(y), 'maxx': max(x), 'maxy': max(y)}
            ref_boxes_projected = np.asarray([np.asarray([line[0], line[1]]) for line in corner_lines])
            ref_boxes_projected = [tuple([line[0], line[1]]) for line in corner_lines]

            minbox = np.min(box, axis=0)
            maxbox = np.max(box, axis=0)
            points3dcin = points_with_reflectance[
                          (((points_with_reflectance[:, 0] > minbox[0]) &
                            (points_with_reflectance[:, 0] < maxbox[0])) &
                           ((points_with_reflectance[:, 1] > minbox[1]) &
                            (points_with_reflectance[:, 1] < maxbox[1])) &
                           ((points_with_reflectance[:, 2] > minbox[2]) & (
                                   points_with_reflectance[:, 2] <
                                   maxbox[2]))), :]
            points3dcin = np.c_[points3dcin, np.ones(np.shape(points3dcin)[0])]

            points3dinprojected = np.zeros((np.shape(points3dcin)[0], 2))
            for row in range(np.shape(points3dcin)[0]):
                cpointp = np.dot(projection_mat, points3dcin[row, :])
                if cpointp[2] != 0:
                    cpointp[0] = int(cpointp[0] / cpointp[2])
                    cpointp[1] = int(cpointp[1] / cpointp[2])
                else:
                    cpointp[0] = int(cpointp[0])
                    cpointp[1] = int(cpointp[1])
                cpointp = cpointp.astype(int)
                points3dinprojected[row, :] = np.asarray([cpointp[0], cpointp[1]])
            points3dinprojected = points3dinprojected.astype(int)

            prediction = {"bbox3D": box,
                          "bbox3DCenter": boxcenter_3D,
                          "bbox3DProjected": ref_boxes_projected,
                          "bbox3DCenterProjected": boxcenter_projected,
                          "bbox2D": box2d,
                          "label": ref_labels[i],
                          "score": ref_scores[i],
                          "points3D": points3dcin,
                          "points3DProjected": points3dinprojected
                          }
            predictions.append(prediction)
    return predictions


def get3DBoundingBox(A):
    bbox = {"minx": 0,
            "maxx": 0,
            "miny": 0,
            "maxy": 0,
            "minz": 0,
            "maxz": 0
            }

    if np.shape(A)[0] > 0:
        bbox = {"minx": np.min(A[:, 0]),
                "maxx": np.max(A[:, 0]),
                "miny": np.min(A[:, 1]),
                "maxy": np.max(A[:, 1]),
                "minz": np.min(A[:, 2]),
                "maxz": np.max(A[:, 2])
                }
    return bbox


def draw_bbox(predictions, image, showBoxCenter):
    for prediction in predictions:
        label = prediction["label"]
        projectedPoints = prediction['bbox3DProjected']
        if label == 1:
            color = (0, 0, 142)
            color_vis = (0, 255, 0)
        elif label == 2 or label == 3:
            color = (220, 20, 60)
            color_vis = (255, 255, 0)
        else:
            continue
        thickness = 2

        if showBoxCenter:
            image = cv2.circle(image, (prediction['bbox3DCenterProjected'][0], prediction['bbox3DCenterProjected'][1]),
                               radius=6, color=(0, 0, 255), thickness=5)

        image = cv2.line(image, projectedPoints[0], projectedPoints[1],
                         (int(color_vis[0]), int(color_vis[1]), int(color_vis[2])), thickness)
        image = cv2.line(image, projectedPoints[0], projectedPoints[3],
                         (int(color_vis[0]), int(color_vis[1]), int(color_vis[2])), thickness)
        image = cv2.line(image, projectedPoints[0], projectedPoints[4],
                         (int(color_vis[0]), int(color_vis[1]), int(color_vis[2])), thickness)
        image = cv2.line(image, projectedPoints[1], projectedPoints[2],
                         (int(color_vis[0]), int(color_vis[1]), int(color_vis[2])), thickness)
        image = cv2.line(image, projectedPoints[1], projectedPoints[5],
                         (int(color_vis[0]), int(color_vis[1]), int(color_vis[2])), thickness)
        image = cv2.line(image, projectedPoints[2], projectedPoints[3],
                         (int(color_vis[0]), int(color_vis[1]), int(color_vis[2])), thickness)
        image = cv2.line(image, projectedPoints[2], projectedPoints[6],
                         (int(color_vis[0]), int(color_vis[1]), int(color_vis[2])), thickness)
        image = cv2.line(image, projectedPoints[3], projectedPoints[7],
                         (int(color_vis[0]), int(color_vis[1]), int(color_vis[2])), thickness)
        image = cv2.line(image, projectedPoints[4], projectedPoints[7],
                         (int(color_vis[0]), int(color_vis[1]), int(color_vis[2])), thickness)
        image = cv2.line(image, projectedPoints[4], projectedPoints[5],
                         (int(color_vis[0]), int(color_vis[1]), int(color_vis[2])), thickness)
        image = cv2.line(image, projectedPoints[5], projectedPoints[6],
                         (int(color_vis[0]), int(color_vis[1]), int(color_vis[2])), thickness)
        image = cv2.line(image, projectedPoints[7], projectedPoints[6],
                         (int(color_vis[0]), int(color_vis[1]), int(color_vis[2])), thickness)

        # tuple = {'image': black_img, 'label': ref_labels[i], '2d_corder': box2d}
        # black_imgs.append(tuple)

    return image


def visualize_2Dboxes(image, class_ids, boxes, scores=None, class_names=None, save_path=None, show=False, cutoff=None,
                      isgt=False,color=None,offset=0):
    image = image.astype(np.uint8)
    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        if scores is not None and cutoff is not None:
            if scores[i] > cutoff:
                class_id = class_ids[i]
                if color==None:
                    color = class_colors[class_id].tolist()
                bbox = boxes[i].astype(np.uint32).tolist()
                # if ((bbox[0] <= bbox[2]) & (bbox[1] <= bbox[3])):

                if ((bbox[0] > 0) & (bbox[0] < np.shape(image)[1])):
                    if ((bbox[2] > 0) & (bbox[2] < np.shape(image)[1])):
                        if ((bbox[1] > 0) & (bbox[1] < np.shape(image)[0])):
                            if ((bbox[3] > 0) & (bbox[3] < np.shape(image)[0])):
                                bbox[0] = np.max(np.asarray([0, bbox[0]]))
                                bbox[1] = np.max(np.asarray([0, bbox[1]]))
                                bbox[2] = np.min(np.asarray([np.shape(image)[1] - 1, bbox[2]]))
                                bbox[3] = np.min(np.asarray([np.shape(image)[0] - 1, bbox[3]]))

                                if not isgt:
                                    image = cv2.rectangle(image, (bbox[0]+offset, bbox[1]+offset), (bbox[2]+offset, bbox[3]+offset), color, 2)
                                if isgt:
                                    image = cv2.rectangle(image,
                                                          (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                                          [0, 255, 255], 2)

                                # print(class_id)
                                class_name = class_names[class_id] if ((class_names is not None) and (class_id<len(class_names))) else 'class_{}'.format(
                                    class_id)

                                text = '{} {:.2f}'.format(class_name, scores[i]) if scores is not None else class_name
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                text_size = cv2.getTextSize(text, font, fontScale=.5, thickness=1)[0]
                                if not isgt:
                                    image = cv2.rectangle(image,
                                                          (bbox[0], bbox[1] - text_size[1] - 8),
                                                          (bbox[0] + text_size[0] + 8, bbox[1]),
                                                          color, -1)
                                    image = cv2.putText(image, text, (bbox[0] + 4, bbox[1] - 4), font,
                                                        fontScale=.5, color=(255, 255, 255), thickness=1,
                                                        lineType=cv2.LINE_AA)

    return image


def project3Dto2D(A, projection_mat):
    cpointp = np.dot(projection_mat, A)
    if (np.isnan(cpointp).any()):
        return np.asarray([np.nan, np.nan])
    if cpointp[2] != 0:
        cpointp[0] = int(cpointp[0] / cpointp[2])
        cpointp[1] = int(cpointp[1] / cpointp[2])
    else:
        cpointp[0] = int(cpointp[0])
        cpointp[1] = int(cpointp[1])
    cpointp = cpointp.astype(int)
    return cpointp


def label_img_to_color(img):
    label_to_color = {
        0: [10, 0, 0],
        1: [20, 0, 0],
        2: [30, 0, 0],
        3: [50, 0, 0],
        4: [80, 0, 0],
        5: [90, 0, 0],
        6: [0, 10, 0],
        7: [0, 20, 0],
        8: [0, 60, 0],
        9: [0, 80, 0],
        10: [0, 120, 0],
        11: [220, 20, 60],  # person
        12: [220, 20, 60],  # riders
        13: [0, 0, 142],  # car
        14: [0, 20, 200],
        15: [0, 10, 60],
        16: [0, 20, 30],
        17: [10, 0, 60],
        18: [255, 0, 0],
        19: [0, 255, 255]
    }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]

            img_color[row, col] = np.array(label_to_color[label])

    return img_color


def segment_image(img, network):
    img = img / 255.0
    img = img - np.array([0.485, 0.456, 0.406])
    img = img / np.array([0.229, 0.224, 0.225])  # (shape: (512, 1024, 3))
    img = np.transpose(img, (2, 0, 1))  # (shape: (3, 512, 1024))
    img = img.astype(np.float32)
    imgs = torch.from_numpy(img)  # (shape: (3, 512, 1024))
    imgsori = torch.from_numpy(img)  # (shape: (3, 512, 1024))

    with torch.no_grad():
        imgs = imgs.unsqueeze(0)
        imgs = Variable(imgs).cuda()  # (shape: (batch_size, 3, img_h, img_w))

        outputs = network(imgs)  # (shape: (batch_size, num_classes, img_h, img_w))
        outputs = outputs.data.cpu().numpy()  # (shape: (batch_size, num_classes, img_h, img_w))
        pred_label_imgs = np.argmax(outputs, axis=1)  # (shape: (batch_size, img_h, img_w))
        pred_label_imgs = pred_label_imgs.astype(np.uint8)

        pred_label_img = pred_label_imgs[0]  # (shape: (img_h, img_w))
        img = imgsori  # (shape: (3, img_h, img_w))

        img = img.data.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # (shape: (img_h, img_w, 3))
        img = img * np.array([0.229, 0.224, 0.225])
        img = img + np.array([0.485, 0.456, 0.406])
        img = img * 255.0
        img = img.astype(np.uint8)

        pred_label_img_color = label_img_to_color(pred_label_img)
        # overlayed_img = 0.35 * img + 0.65 * pred_label_img_color
        overlayed_img = 0. * img + 1.0 * pred_label_img_color
        colour_overlayed_img = 0.7 * img + 0.3 * pred_label_img_color
        overlayed_img = overlayed_img.astype(np.uint8)
        colour_overlayed_img = colour_overlayed_img.astype(np.uint8)

        return colour_overlayed_img, overlayed_img
