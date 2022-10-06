import glob
import json
import sys

# Add OpenPCDet directory to path =======================================================================
config = json.load(open('config.json', ))
config = config['multimodalv2']
sys.path.insert(0, config['root'])

from definitions import *
from pcdet.config import cfg, cfg_from_yaml_file


def parse_config(external_config=None):
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')

    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')

    parser.add_argument('--data_path', type=str,
                        default='',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()
    if external_config is not None:
        args.cfg_file = external_config['lidar_detection_cfg']
        args.ckpt = external_config['lidar_detection_model']
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    return args, cfg


def detect2D(detector, imagein, idx, cutoff, device, preprocess_func=None):
    imagein = imagein[0, :, :, :]
    if np.shape(imagein)[2] == 4:
        imagein = imagein[:, :, :3]
    imagein_meta = {'image_id': str(idx), 'orig_size': np.array(imagein.shape, dtype=np.int32)}
    imagein, imagein_meta, _ = preprocess_func(imagein, imagein_meta)
    imagein = np.moveaxis(imagein, -1, 0)
    imagein = torch.from_numpy(imagein).unsqueeze(0).to(device)
    imagein_meta = {k: torch.from_numpy(v).unsqueeze(0).to(device) if isinstance(v, np.ndarray) else [v]
                    for k, v in imagein_meta.items()}
    inp = {'image': imagein, 'image_meta': imagein_meta}
    imagedets, _ = detector.detect(inp, cutoff)
    return imagedets


def late_fusion_2D_3D(pred_dicts_2D, pred_dicts_3D_to_2D, points_with_reflectance, projection_mat, cutoff2D,
                      nms_fusion_threshold):
    # ===== Get projected points of ============================================
    pointsinc = np.c_[points_with_reflectance, np.ones(np.shape(points_with_reflectance)[0])]
    pointsprojected = np.zeros((np.shape(points_with_reflectance)[0], 2))
    for row in range(np.shape(pointsinc)[0]):
        cpointp = project3Dto2D(pointsinc[row, :], projection_mat)
        if not np.isnan(cpointp).any():
            pointsprojected[row, :] = np.asarray([cpointp[0], cpointp[1]])
    pointsprojected = pointsprojected.astype(int)

    # ========== Get projected points within boxes ================================
    if 'boxes' in pred_dicts_2D[0]:
        correspondingProjectedPointsList = []
        corresponding3DPointsList = []
        bbox3DList = []
        bbox3DCenterList = []
        for i, bbox in enumerate(pred_dicts_2D[0]['boxes']):
            B = (((pointsprojected[:, 0] > bbox[0]) & (pointsprojected[:, 0] < bbox[2])) &
                 ((pointsprojected[:, 1] > bbox[1]) & (pointsprojected[:, 1] < bbox[3])))
            indices = np.asarray(np.where(B))
            correspondingProjectedPointsList.append(pointsprojected[B, :])
            corresponding3DPointsList.append(pointsinc[B, :])
            pointstocalc = pointsinc[B, :3]
            corresponding3DBbox = get3DBoundingBox(pointstocalc)
            corresponding3DBboxCentroid = np.mean(pointstocalc, axis=0)
            corresponding3DBboxProcessed = np.asarray([
                corresponding3DBboxCentroid[0],
                corresponding3DBboxCentroid[1],
                corresponding3DBboxCentroid[2],
                corresponding3DBbox['maxx'] - corresponding3DBbox['minx'],
                corresponding3DBbox['maxy'] - corresponding3DBbox['miny'],
                corresponding3DBbox['maxz'] - corresponding3DBbox['minz'],
                0
            ])
            bbox3DCenterList.append(corresponding3DBboxCentroid)
            bbox3DList.append(corresponding3DBboxProcessed)
        pred_dicts_2D[0]['correspondingProjectedPoints'] = np.asarray(correspondingProjectedPointsList)
        pred_dicts_2D[0]['corresponding3DPoints'] = np.asarray(corresponding3DPointsList)
        pred_dicts_2D[0]['corresponding3DCentroid'] = bbox3DCenterList
        pred_dicts_2D[0]['corresponding3DBBox'] = bbox3DList

    # ========= Fusion 2D ==========================================
    imgdetcls = None
    imgdetsbox = None
    imgdetsscores = None
    if len(pred_dicts_2D) > 0:
        if (('class_ids' in pred_dicts_2D[0]) & ('boxes' in pred_dicts_2D[0]) & ('scores' in pred_dicts_2D[0])):
            imgdetcls = pred_dicts_2D[0]['class_ids']
            imgdetsbox = pred_dicts_2D[0]['boxes']
            imgdetsscores = pred_dicts_2D[0]['scores']

    if len(pred_dicts_3D_to_2D) > 0:
        det_boxes_3D_2D = np.asarray(
            [np.asarray([
                det['bbox2D']['minx'],
                det['bbox2D']['miny'],
                det['bbox2D']['maxx'],
                det['bbox2D']['maxy']]) for det in pred_dicts_3D_to_2D])
        det_class_ids_3D_2D = np.asarray([det['label'] for det in pred_dicts_3D_to_2D]) - 1
        det_scores_3D_2D = np.asarray([det['score'] for det in pred_dicts_3D_to_2D])

        if (imgdetcls is not None) & (imgdetsbox is not None) & (imgdetsscores is not None):
            class_2D_fused = np.concatenate((det_class_ids_3D_2D, imgdetcls), axis=0)
            boxes_2D_fused = np.concatenate((det_boxes_3D_2D, imgdetsbox), axis=0)
            scores_2D_fused = np.concatenate((det_scores_3D_2D, imgdetsscores), axis=0)
        else:
            class_2D_fused = det_class_ids_3D_2D
            boxes_2D_fused = det_boxes_3D_2D
            scores_2D_fused = det_scores_3D_2D

    else:
        if (imgdetcls is not None) & (imgdetsbox is not None) & (imgdetsscores is not None):
            class_2D_fused = imgdetcls
            boxes_2D_fused = imgdetsbox
            scores_2D_fused = imgdetsscores
        else:
            class_2D_fused = np.asarray([])
            boxes_2D_fused = np.asarray([])
            scores_2D_fused = np.asarray([])

    ## LIDAR to IMAGE
    if np.shape(boxes_2D_fused)[0] > 0:
        boxes = torch.from_numpy(boxes_2D_fused).type(torch.float32)
        scores = torch.from_numpy(scores_2D_fused).type(torch.float32)
        bboxSelection = nms(boxes=boxes, scores=scores, iou_threshold=nms_fusion_threshold)
        bboxSelection = bboxSelection.cpu().detach().numpy()
        class_2D_fused = class_2D_fused[bboxSelection]
        boxes_2D_fused = boxes_2D_fused[bboxSelection]
        scores_2D_fused = scores_2D_fused[bboxSelection]

    ## IMAGE to LIDAR
    pointsToVisualize = []
    centroids_2D_to_3D = []
    if len(pred_dicts_2D) > 0:
        if (('class_ids' in pred_dicts_2D[0]) & ('boxes' in pred_dicts_2D[0]) & ('scores' in pred_dicts_2D[0])):
            for i, bbox in enumerate(pred_dicts_2D[0]['boxes']):
                if pred_dicts_2D[0]['scores'][i] > cutoff2D:
                    for row in range(np.shape(pred_dicts_2D[0]['correspondingProjectedPoints'][i])[0]):
                        point = pred_dicts_2D[0]['correspondingProjectedPoints'][i][row]
                        pointsToVisualize.append(point)
                        # images_all_outputs = cv2.circle(images_all_outputs,(point[0],point[1]), radius=2,
                        #                                           color=(0, 255, 0), thickness=3)
                    p = pred_dicts_2D[0]['corresponding3DCentroid'][i].tolist()
                    p.append(0.0)
                    p = np.asarray(p)
                    ctemp = project3Dto2D(p, projection_mat)
                    if not np.isnan(ctemp).any():
                        centroids_2D_to_3D.append(ctemp)
    return class_2D_fused, boxes_2D_fused, scores_2D_fused, centroids_2D_to_3D, pointsToVisualize


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'y1', 'x2', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'y1', 'x2', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    iou = intersection_area / float(bb1_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def pipeline():
    # Config & groundtruth =============================================================================================

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
    args, cfg = parse_config(config)

    if 'tracking' in config['path_to_data']:
        calibration_file_path = config['path_to_data'] + config['path_to_calibration_for_tracking']
        projection_mat = projection_matrix(calibration_file_path)
        gtinput = np.genfromtxt(config['path_to_data'] + config['path_to_groundtruth_for_tracking'], dtype=str)

    else:
        calibration_file_path = config['path_to_data'] + 'calib/000000.txt'
        projection_mat = projection_matrix(calibration_file_path)
        A = glob.glob(config['path_to_data'] + config['path_to_labels'] + "*.txt")
        A.sort()
        gtinput = []
        for idx, gtfile in enumerate(A):
            gtdata = np.genfromtxt(gtfile, dtype=str)
            if (len(np.shape(gtdata)) == 1):
                gtdata = np.expand_dims(gtdata, axis=0)
            for object_index in range(np.shape(gtdata)[0]):
                gtline = list(gtdata[object_index])
                gtline.insert(0, str(object_index))
                gtline.insert(0, str(idx))
                gtinput.append(gtline)
        gtinput = np.asarray(gtinput)

    classes = 'Car Pedestrian Cyclist Van Truck Person Tram Misc DontCare Person_sitting'.split()
    for idx, cls in enumerate(classes):
        gtinput[gtinput == cls] = idx
    gtinput = gtinput.astype(float)
    gtinput = np.append(gtinput, np.zeros((np.shape(gtinput)[0], 1)), axis=1)

    # 2D detector ======================================================================================================
    cfgimg = Config().parse()
    init_env(cfgimg)
    # prepare configurations
    cfgimg.load_model = config["image_detection_model"]
    # cfgimg.load_model = '/home/stavros/Workspace/SqueezeDet-PyTorch/exp/carla_test/model_last.pth'
    cfgimg.gpus = [-1]  # -1 to use CPU
    cfgimg.debug = 2  # to visualize detection boxes
    dataset = KITTI('val', cfgimg)
    cfgimg = Config().update_dataset_info(cfgimg, dataset)
    # preprocess image to match model's input resolution
    preprocess_func = dataset.preprocess
    del dataset



    # prepare model & detector
    model = SqueezeDet(cfgimg)
    model = load_model(model, cfgimg.load_model)
    detector = Detector(model.to(cfgimg.device), cfgimg)

    # Local paths
    if os.path.isdir(config["save_path_root"]):
        shutil.rmtree(config["save_path_root"], ignore_errors=False)
    os.mkdir(config["save_path_root"])
    os.mkdir(config["save_path_came"])

    os.mkdir(config["save_path_image_from_lidar"])
    os.mkdir(config["save_path_lidar"])

    os.mkdir(config["save_path_root"] + "/groundtruth")
    os.mkdir(config["save_path_root"] + "/groundtruth/label_2")
    os.mkdir(config["save_path_root"] + "/prediction_image")
    os.mkdir(config["save_path_root"] + "/prediction_image/data")
    os.mkdir(config["save_path_root"] + "/prediction_fusion")
    os.mkdir(config["save_path_root"] + "/prediction_fusion/data")
    os.mkdir(config["save_path_root"] + "/prediction_lidar")
    os.mkdir(config["save_path_root"] + "/prediction_lidar/data")

    # os.mkdir(config["save_path_attacked_came"])
    # os.mkdir(config["save_path_meta_data"])
    # os.mkdir(config["save_path_de_meta_data"])

    logger = common_utils.create_logger()

    # load dataset for the demo lidar/image/attacked image
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=['Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck', 'Person', 'Tram', 'Misc', 'DontCare',
                     'Person_sitting'],
        training=False,
        path_conf=config, ext=args.ext, logger=logger
    )

    dist_train = False
    writePoints = True

    # load model for 3D object detection
    model_lidar = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model_lidar.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model_lidar.cuda()
    model_lidar.eval()
    doVisualizeAndSave = True

    with torch.no_grad():
        close = 0  # for closing the plot after the screnshot
        for idx, data_dict in enumerate(demo_dataset):

            if doVisualizeAndSave:
                if close == 1: mlab.close()

            data_dict = demo_dataset.collate_batch([data_dict])

            # ====================== Input to be read from messages======================================================


            if np.shape(data_dict['image'][0])[2]==4:
                input_image_data= data_dict['image'][0][:,:,:3]
            else:
                input_image_data = data_dict['image'][0]

            images_all_outputs = copy.deepcopy(input_image_data)
            processed_image = copy.deepcopy(input_image_data)
            # Input 3D
            points = copy.deepcopy(data_dict['points'])
            points_with_reflectance = points[:, 1:4]
            # Detect 2D=================================================
            pred_dicts_2D = detect2D(detector, data_dict['image'], idx, config['cut_off_2D'], cfgimg.device,
                                     preprocess_func)
            # Detect 3D=================================================
            pred_dicts_3D = detect_in_lidar(data_dict, model_lidar)
            pred_dicts_3D_to_2D = get3DBBoxProjection(pred_dicts_3D, points_with_reflectance, projection_mat,
                                                      cutoff=config["cut_off_percentage"])

            # Make sure that projections are within the image ===================================

            dimX = data_dict['image'].size(dim=1)
            dimY = data_dict['image'].size(dim=2)
            pred_dicts_3D_to_2D = [p for p in pred_dicts_3D_to_2D if
                                   ((p['bbox2D']['minx'] > 0) and (p['bbox2D']['maxx'] < dimY) and
                                    (p['bbox2D']['miny'] > 0) and (p['bbox2D']['maxy'] < dimX))]

            # ================= Groundtruth ================================================================
            df = gtinput[gtinput[:, 0] == data_dict['frame_id'][0], :]
            df = df[df[:, 2] <= 2, :]
            groundtruth_per_frame = []
            groundtruth_per_frame_class = []
            groundtruth_per_frame_score = []
            groundtruth_difficulty = []
            groundtruth_all = []
            for row in range(np.shape(df)[0]):
                groundtruth_all.append(df[row, :])
                groundtruth_per_frame.append([df[row, 6], df[row, 7], df[row, 8], df[row, 9]])
                groundtruth_per_frame_class.append(df[row, 2])
                groundtruth_per_frame_score.append(1)
                groundtruth_difficulty.append(df[row, 17])
            groundtruth_per_frame = np.asarray(groundtruth_per_frame)
            groundtruth_per_frame_class = np.asarray(groundtruth_per_frame_class).astype(int)
            groundtruth_per_frame_score = np.asarray(groundtruth_per_frame_score)
            groundtruth_difficulty = np.asarray(groundtruth_difficulty)
            groundtruth_all = np.asarray(groundtruth_all)
            # dontcareareas = groundtruth_per_frame[groundtruth_per_frame_class == 8, :]
            # S = (groundtruth_per_frame_class == 0) | (groundtruth_per_frame_class == 1) | (
            #             groundtruth_per_frame_class == 2)
            # groundtruth_per_frame = groundtruth_per_frame[S, :]
            # groundtruth_per_frame_class = groundtruth_per_frame_class[S]
            # groundtruth_per_frame_score = groundtruth_per_frame_score[S]
            # groundtruth_difficulty = groundtruth_difficulty[S]
            # groundtruth_all = groundtruth_all[S, :]

            annotations_gt = {}
            content = groundtruth_all[:, 2:]
            annotations_gt['name'] = np.array([classes[int(x[0])] for x in content])
            annotations_gt['truncated'] = np.array([float(x[1]) for x in content])
            annotations_gt['occluded'] = np.array([int(x[2]) for x in content])
            annotations_gt['alpha'] = np.array([float(x[3]) for x in content])
            annotations_gt['bbox'] = np.array(
                [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
            # dimensions will convert hwl format to standard lhw(camera) format.
            annotations_gt['dimensions'] = np.array(
                [[float(info) for info in x[8:11]] for x in content]).reshape(
                -1, 3)[:, [2, 0, 1]]
            annotations_gt['location'] = np.array(
                [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
            annotations_gt['rotation_y'] = np.array(
                [float(x[14]) for x in content]).reshape(-1)
            annotations_gt['score'] = np.zeros([len(annotations_gt['bbox'])])

            filecontent = []
            input = annotations_gt
            for rowindex in range(np.shape(input['name'])[0]):
                line = []
                for key, val in input.items():
                    if not isinstance(val[rowindex], (list, tuple, np.ndarray)):
                        if isinstance(val[rowindex], str):
                            line.append(str(val[rowindex]))
                        elif (key == 'occluded'):
                            line.append(str(int(val[rowindex])))
                        else:
                            line.append('{:.2f}'.format(val[rowindex]))
                    else:
                        for element in val[rowindex]:
                            line.append('{:.2f}'.format(element))
                lineout = ' '.join(line)
                filecontent.append(lineout)
            with open(config["save_path_root"] + "/groundtruth/label_2/" + "{:06d}".format(
                    data_dict['frame_id'][0]) + ".txt", mode='wt', encoding='utf-8') as myfile:
                myfile.write('\n'.join(filecontent))

            # ======================== Fusion ==========================================

            class_2D_fused, boxes_2D_fused, scores_2D_fused, centroids_2D_to_3D, pointsToVisualize = \
                late_fusion_2D_3D(pred_dicts_2D, pred_dicts_3D_to_2D, points_with_reflectance, projection_mat,
                                  config['cut_off_2D'], config['nms_fusion_threshold'])

            # ======================== Compare with dont care areas  ==========================================

            # ========================  Modalities  ==========================================

            if 'class_ids' in pred_dicts_2D[0]:
                metric = {}
                metric['boxes'] = pred_dicts_2D[0]['boxes']
                metric['classes'] = pred_dicts_2D[0]['class_ids']
                metric['scores'] = pred_dicts_2D[0]['scores']
                filecontent = writePredictionsToList(generateAnnotationPrediction(metric, classes))
                with open(
                        config["save_path_root"] + "/prediction_image/data/" + "{:06d}".format(
                            data_dict['frame_id'][0]) + ".txt",
                        mode='wt', encoding='utf-8') as myfile:
                    myfile.write('\n'.join(filecontent))

                if len(class_2D_fused) > 0:
                    metric = {}
                    metric['boxes'] = boxes_2D_fused
                    metric['classes'] = class_2D_fused

                    metric['scores'] = scores_2D_fused
                    filecontent = writePredictionsToList(generateAnnotationPrediction(metric, classes))
                    with open(config["save_path_root"] + "/prediction_fusion/data/" + "{:06d}".format(
                            data_dict['frame_id'][0]) + ".txt", mode='wt', encoding='utf-8') as myfile:
                        myfile.write('\n'.join(filecontent))

                if len(pred_dicts_3D_to_2D) > 0:
                    det_boxes_3D_2D = np.asarray(
                        [np.asarray([
                            det['bbox2D']['minx'],
                            det['bbox2D']['miny'],
                            det['bbox2D']['maxx'],
                            det['bbox2D']['maxy']]) for det in pred_dicts_3D_to_2D])
                    det_class_ids_3D_2D = np.asarray([det['label'] for det in pred_dicts_3D_to_2D]) - 1
                    det_scores_3D_2D = np.asarray([det['score'] for det in pred_dicts_3D_to_2D])
                    metric = {}
                    metric['boxes'] = det_boxes_3D_2D
                    metric['classes'] = det_class_ids_3D_2D
                    metric['scores'] = det_scores_3D_2D
                    filecontent = writePredictionsToList(generateAnnotationPrediction(metric, classes))
                    with open(config["save_path_root"] + "/prediction_lidar/data/" + "{:06d}".format(
                            data_dict['frame_id'][0]) + ".txt", mode='wt', encoding='utf-8') as myfile:
                        myfile.write('\n'.join(filecontent))

            # ======================== Compute MAP  ==========================================

            # ======================== Compute MAP  ==========================================

            # ========================Visualization ==========================================

            # Visualize Fusion 2D ==========================================
            class_names = ('Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck', 'Person_sitting', 'Tram', 'Misc', 'DontCare')
            if True:

                if True:
                    if 'boxes' in pred_dicts_2D[0]:
                        images_all_outputs = visualize_2Dboxes(images_all_outputs,
                                                               pred_dicts_2D[0]['class_ids'],
                                                               pred_dicts_2D[0]['boxes'],
                                                               pred_dicts_2D[0]['scores'],
                                                               class_names=class_names,
                                                               save_path=None, cutoff=config['cut_off_2D'],
                                                               color=[0, 0, 255], offset=4)

                    images_all_outputs = visualize_2Dboxes(images_all_outputs,
                                                           class_2D_fused,
                                                           boxes_2D_fused,
                                                           scores_2D_fused,
                                                           class_names=class_names,
                                                           save_path=None, cutoff=config['cut_off_2D'])

                if False:
                    for point in pointsToVisualize:
                        images_all_outputs = cv2.circle(images_all_outputs, (point[0], point[1]), radius=2,
                                                        color=(0, 255, 0), thickness=3)

                for centroid in centroids_2D_to_3D:
                    if np.all(np.isfinite(centroid)):
                        images_all_outputs = cv2.circle(images_all_outputs,
                                                        (centroid[0],
                                                         centroid[1]),
                                                        radius=4,
                                                        color=(255, 0, 0), thickness=4)

                # Draw LIDAR / 3D =================================================
                images_all_outputs = draw_bbox(pred_dicts_3D_to_2D, images_all_outputs, showBoxCenter=True)

                # Draw Metrics =================================================

                # Draw legends =================================================

                predType = 'Prediction LIDAR'
                images_all_outputs = cv2.rectangle(images_all_outputs, (870, 220), (890, 240), [0, 255, 0], -1)
                cv2.putText(images_all_outputs, text=predType, org=(900, 240),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 255, 0),
                            thickness=2, lineType=cv2.LINE_AA)

                predType = 'Prediction Image-based'
                images_all_outputs = cv2.rectangle(images_all_outputs, (870, 260), (890, 280), [0, 0, 255], -1)
                cv2.putText(images_all_outputs, text=predType, org=(900, 280),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 255),
                            thickness=2, lineType=cv2.LINE_AA)

                predType = 'Prediction Fusion'
                images_all_outputs = cv2.rectangle(images_all_outputs, (870, 300), (890, 320), [255, 0, 0], -1)
                cv2.putText(images_all_outputs, text=predType, org=(900, 320),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 0, 0),
                            thickness=2, lineType=cv2.LINE_AA)

                images_all_outputs = cv2.rectangle(images_all_outputs, (870, 340), (890, 360), [0, 255, 255], -1)
                cv2.putText(images_all_outputs, text='Groundtruth', org=(900, 360),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 255, 255),
                            thickness=2, lineType=cv2.LINE_AA)

                images_all_outputs = visualize_2Dboxes(images_all_outputs,
                                                       groundtruth_per_frame_class,
                                                       groundtruth_per_frame,
                                                       groundtruth_per_frame_score,
                                                       class_names=class_names,
                                                       save_path=None, cutoff=config['cut_off_2D'], isgt=True)

            # metadata ========================================================================================
            fused_images = []
            meta_data = []
            # we find 3d bbox we project the to the image as 2d bbox we find the area and crop the respective area
            # in the semgented images. After that we combare the 2 areas
            for prediction in pred_dicts_3D_to_2D:
                min_x = prediction['bbox2D']['minx']
                min_y = prediction['bbox2D']['miny']
                max_x = prediction['bbox2D']['maxx']
                max_y = prediction['bbox2D']['maxy']
                crop_img = processed_image[min_y:max_y, min_x:max_x]
                if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                    continue
                if prediction["label"] == 1:
                    crop_lid = np.ones((crop_img.shape[0], crop_img.shape[1], 3), dtype=np.float32) * [0, 0, 142]
                if prediction["label"] == 2 or prediction["label"] == 3:
                    crop_lid = np.ones((crop_img.shape[0], crop_img.shape[1], 3), dtype=np.uint8) * [220, 20, 60]
                if crop_lid.shape[0] >= 3 and crop_lid.shape[1] >= 3:
                    fused_images.append(
                        {"crop_camera": crop_img, "crop_lidar": crop_lid}
                    )
                    # data for video
                    meta_data.append({'minx': min_x, 'miny': min_y, 'maxx': max_x, 'maxy': max_y})

            # save the results ========================================================================================

            frame_idx = demo_dataset.names[idx].split('.')[0]
            cv2.imwrite(config['save_path_came'] + frame_idx + "_" + str(3) + "_" + str(3) + '.png', processed_image)
            cv2.imwrite(config['save_path_image_from_lidar'] + frame_idx + "_" + str(3) + "_" + str(3) + '.png',
                        images_all_outputs)
            VP = data_dict['points'][:, 1:].cpu().numpy()
            VB = pred_dicts_3D[0]['pred_boxes'].cpu().numpy()
            VS = pred_dicts_3D[0]['pred_scores'].cpu().numpy()
            VL = pred_dicts_3D[0]['pred_labels'].cpu().numpy()

            if doVisualizeAndSave:
                V.draw_scenes(
                    points=VP,
                    ref_boxes=VB,
                    ref_scores=VS,
                    ref_labels=VL
                )
                mlab.savefig(config['save_path_lidar'] + frame_idx + ".png")
                close = 1

    logger.info('Demo done.')
    pass


pipeline()
