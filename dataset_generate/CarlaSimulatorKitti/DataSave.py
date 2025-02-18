from config import config_to_trans
from carlautils.export_utils import *
import time


class DataSave:
    def __init__(self, cfg,player, args = None):
        self.cfg = cfg
        self.OUTPUT_FOLDER = None
        self.LIDAR_PATH = None
        self.KITTI_LABEL_PATH = None
        self.IMAGE_PATH = None
        self.CALIBRATION_PATH = None
        self.RADAR_PATH = None
        self.IMU_PATH = None
        self.GNSS_PATH = None
        self.VELO_PATH = None
        self.GROUNDPLANE_PATH = None
        self.player=player
        self.IMAGE_EXT = self._get_image_ext(args)
        self.kitti_only = True if args and args.kitti_only else False
        
        self._generate_path(self.cfg["SAVE_CONFIG"]["ROOT_PATH"])
        self.captured_frame_no = self._current_captured_frame_num()

    def _get_image_ext(self, args = None):
        image_filter = ['jpg','jpeg','png']

        if args :
            try:
                if str(args.image_type).lower() in image_filter:
                    return str(args.image_type).lower()
                else:
                    raise Exception
            except:
                return "png"
        return "png"

    def _generate_path(self, root_path):
        """Save path of generated data"""
        PHASE = "training"
        self.OUTPUT_FOLDER = os.path.join(root_path, PHASE)
        folders = ["calib", "image_2", "label_2", "velodyne","radar", "imu","gnss","velo","planes"]
        if self.kitti_only:
            folders = ["calib","image_2", "label_2","velodyne","planes"]

        for folder in folders:
            directory = os.path.join(self.OUTPUT_FOLDER, folder)
            if not os.path.exists(directory):
                os.makedirs(directory)

        self.GROUNDPLANE_PATH = os.path.join(self.OUTPUT_FOLDER, "planes/{0:06}.txt")
        self.LIDAR_PATH = os.path.join(self.OUTPUT_FOLDER, "velodyne/{0:06}.bin")
        self.KITTI_LABEL_PATH = os.path.join(self.OUTPUT_FOLDER, "label_2/{0:06}.txt")
        self.IMAGE_PATH = os.path.join(self.OUTPUT_FOLDER, "image_2/{0:06}."+self.IMAGE_EXT)
        self.CALIBRATION_PATH = os.path.join(self.OUTPUT_FOLDER, "calib/{0:06}.txt")
        self.RADAR_PATH = os.path.join(self.OUTPUT_FOLDER, "radar/{0:06}.txt")
        self.IMU_PATH = os.path.join(self.OUTPUT_FOLDER, "imu/{0:06}.txt")
        self.GNSS_PATH = os.path.join(self.OUTPUT_FOLDER, "gnss/{0:06}.txt")
        self.VELO_PATH = os.path.join(self.OUTPUT_FOLDER,"velo/{0:06}.txt")
    def _current_captured_frame_num(self):
        """获取文件夹中存在的数据量"""
        label_path = os.path.join(self.OUTPUT_FOLDER, "label_2/")
        num_existing_data_files = len(
            [name for name in os.listdir(label_path) if name.endswith(".txt")]
        )
        print("Currently there are {} data exist.".format(num_existing_data_files))
        if num_existing_data_files == 0:
            return 0
        answer = input(
            "There already exists a dataset in {}. Would you like to (O)verwrite or (A)ppend the dataset? (O/A)".format(
                self.OUTPUT_FOLDER
            )
        )
        if answer.upper() == "O":
            # logging.info(
            #    "Resetting frame number to 0 and overwriting existing")
            return 0
        # logging.info("Continuing recording data on frame number {}".format(
        #    num_existing_data_files))
        return num_existing_data_files

    def save_training_files(self, data):

        groundtruth_plane_fname = self.GROUNDPLANE_PATH.format(self.captured_frame_no)
        lidar_fname = self.LIDAR_PATH.format(self.captured_frame_no)
        kitti_label_fname = self.KITTI_LABEL_PATH.format(self.captured_frame_no)
        img_fname = self.IMAGE_PATH.format(self.captured_frame_no)
        calib_filename = self.CALIBRATION_PATH.format(self.captured_frame_no)
        radar_file_name = self.RADAR_PATH.format(self.captured_frame_no)
        imu_file_name = self.IMU_PATH.format(self.captured_frame_no)
        gnss_file_name = self.GNSS_PATH.format(self.captured_frame_no)
        velo_file_name = self.VELO_PATH.format(self.captured_frame_no)
        for agent, dt in data["agents_data"].items():

            camera_transform = config_to_trans(
                self.cfg["SENSOR_CONFIG"]["RGB"]["TRANSFORM"]
            )
            lidar_transform = config_to_trans(
                self.cfg["SENSOR_CONFIG"]["LIDAR"]["TRANSFORM"]
            )

            save_ref_files(self.OUTPUT_FOLDER, self.captured_frame_no)
            save_image_data(img_fname, dt["sensor_data"][0])
            save_label_data(kitti_label_fname, dt["kitti_datapoints"])
            # save_label_data(kitti_label_fname, dt["carla_datapoints"])
            if self.kitti_only:
                continue
            save_calibration_matrices(
                [camera_transform, lidar_transform], calib_filename, dt["intrinsic"]
            )
            save_lidar_data(lidar_fname, dt["sensor_data"][2])

            save_radar_data(radar_file_name,dt["radar_datapoints"])
            save_imu_data(imu_file_name, dt['imu_data'])
            save_gnss_data(gnss_file_name, dt['gnss_data'])
            save_velo_data(velo_file_name, dt['velo_data'])


            save_groundplanes(groundtruth_plane_fname, self.player, 1.6)

        self.captured_frame_no += 1
