#-*- coding: UTF-8 -*- 
import os
import pickle
import random
import time
from copy import copy

from geometry import get_track_length
from utils.dataset_reader import read_trajectory, read_tracks, read_pedestrian

# large_scale dataset
class DatasetLoader():
    # get all files in the directory according to the scenario setting
    def __init__(self, gt_tracks_dir, ego_max_length=5.5, ego_num=1, eval=False):
        # self._track_loader = DatasetLoader(groundtruth_tracks_dir, eval=self._eval)
        # groundtruth_tracks_dir = {dataset_dir}/recorded_trackfiles/{map_name} 
        self._gt_tracks_dir = gt_tracks_dir # {dataset_dir}/recorded_trackfiles/{map_name}
        self._gt_tracks = os.listdir(self._gt_tracks_dir) # get all files in the directory
        self._gt_csv_index = None
        
        self._eval = eval

        self._ego_max_length = ego_max_length
        self._ego_num = ego_num
        self._possible_ego_dict = dict()
        self._possible_ego_id = []

        # remove troubled vehicles which can not be properly controlled
        self._cont_be_used_as_ego_dict = {0: [], 1: [], 2: [11], 3: [81], 4:[], 5: [57], 6:[], 7:[15, 67]} # 存储不能作为ego的车辆id {csv索引:ego_id}
        # remove small set data from big set data
        small_set_ego_dict = {5: [29, 30, 33, 36, 37, 39, 40, 41]} # small_set_ego_dict = {5: [29, 30, 33, 36, 37, 39, 40, 41]}
        for csv, value in small_set_ego_dict.items():
            self._cont_be_used_as_ego_dict[csv].extend(value) # self._cont_be_used_as_ego_dict = {0: [], 1: [], 2: [11], 3: [81], 4: [], 5: [57, 29, 30, 33, 36, 37, 39, 40, 41], 6: [], 7: [15, 67]} 
                                                                   
        # print("======cont_be_used_as_ego_dict: ======",self._cont_be_used_as_ego_dict)
        # collect tracks that can be used as ego
        self._set_possible_ego_id_dict()
    
    @property
    def track_id(self):
        return self._gt_csv_index
    
    # dataset模式下，选择可能的ego车辆id
    def _set_possible_ego_id_dict(self):
        tracks_num = len(self._gt_tracks) # groundtruth track 目录列表的长度(即{dataset_dir}/recorded_trackfiles/{map_name}/下的文件数)=8
        print("+++++++++++++tracks_num+++++++++++: ",tracks_num)
        for csv_index in range(tracks_num): # 8
            track_file_path = os.path.join(self._gt_tracks_dir, "vehicle_tracks_" + str(csv_index).zfill(3) + ".csv")
            # track_file_path = dataset/recorded_trackfiles/DR_USA_Intersection_EP0/vehicle_tracks_000.csv、vehicle_tracks_001.csv，......
            track_dict = read_tracks(track_file_path) # 读取csv文件vehicle_tracks_000.csv，返回track字典
            # print("+++++++++++++track_dict+++++++++++: ",track_dict)
            vehicle_id_list = track_dict.keys()
            # print("+++++++++++++vehicle_id_list+++++++++++: ",vehicle_id_list)
            possible_ego_id = []
            # some vehicles are not appropriate to use as ego
            for vehicle_id in vehicle_id_list:
                length_condition = track_dict[vehicle_id].length <= self._ego_max_length # ego should have a proper vehicle length
                # length_condition = track_dict[vehicle_id].length <= 5.5  true or false # csv文件中的length列的数据
                time_condition = (track_dict[vehicle_id].time_stamp_ms_last - track_dict[vehicle_id].time_stamp_ms_first)/1000 >= 5 # ego should exist at least 5 seconds
                # time_condition = (track_dict[vehicle_id].time_stamp_ms_last - track_dict[vehicle_id].time_stamp_ms_first)/1000 >= 5 true or false # csv文件中的time_stamp_ms列的数据
                track_condition = get_track_length(track_dict[vehicle_id]) >= 20 # ego route should be longer than 20 meters
                # track_condition = get_track_length(track_dict[vehicle_id]) >= 20 true or false
                if length_condition and time_condition and track_condition: # 三个条件都满足
                    if vehicle_id not in self._cont_be_used_as_ego_dict[csv_index]: # 并且车辆id不在不能作为ego的车辆id列表中
                        possible_ego_id.append(vehicle_id) # 将车辆id添加到可能的ego车辆id列表中
            print('csv {} possible ego num: {}'.format(csv_index, len(possible_ego_id)))
            # 当前的possible_ego_id是csv文件中，首先去除了不符合以上三个条件的、然后去除了不能作为ego的车辆id，剩下得到的才是possible_ego_id
            self._possible_ego_dict[csv_index] = possible_ego_id
            # print("+++++++++++++self._possible_ego_dict+++++++++++: ",self._possible_ego_dict)

    # dataset模式下，从文件中提取数据
    # if mode is train, then use the former 3/4 files, else the latter 1/4 files
    def _extract_data_from_file(self, random=False):
        tracks_num = len(self._gt_tracks)  # 8
        train_num = int(tracks_num * 0.75) # 6
        # eval_num = tracks_num - train_num
        # print("============in _extract_data_from_file def,slef._eval: ===========",self._eval)
        if not self._eval: # train mode
            # choose randomly from all tracks
            # print("========== in _extract_data_from_file def if branch =========")
            if random:  # 随机选择
                self._gt_csv_index = random.randint(0, train_num - 1)
            # choose in turn
            else:       # 按顺序选择
                # print("self._gt_csv_index：",self._gt_csv_index) # None 
                if self._gt_csv_index is None: # 初始值为None,第一次进入, self._gt_csv_index设置为0
                    self._gt_csv_index = 0 
                else: # 后面的_gt_csv_index的值不为None
                    if not self._possible_ego_id: 
                        self._gt_csv_index = 0 if self._gt_csv_index == train_num - 1 else int(self._gt_csv_index) + 1
                        # 如果可能的ego车辆id列表为空，且_gt_csv_index的值为5，那么_gt_csv_index的值设置为0，否则_gt_csv_index的值加1 

        else:  # eval mode
            # print("=========== in _extract_data_from_file def else branch =========")
            if self._gt_csv_index is None:
                self._gt_csv_index = train_num # 6
                # print("=========================== in  _extract_data_from_file def else branch ===============================，self._gt_csv_index：",self._gt_csv_index)
            else:
                if not self._possible_ego_id:
                    self._gt_csv_index = train_num if self._gt_csv_index == tracks_num - 1 else int(self._gt_csv_index) + 1


    def _get_possible_ego_id(self):
        return copy(self._possible_ego_dict[self._gt_csv_index])
            
    # change track file randomly in train mode, and in turn in eval mode
    # 切换轨迹文件(.csv)返回csv文件索引
    def change_track_file(self):
        self._extract_data_from_file()
        # print("=========================== in change_track_file def self._gt_csv_index: ===============================,",self._gt_csv_index)
        return self._gt_csv_index

    # dataset模式下读取轨迹文件
    def read_track_file(self, vdi_type, route_type):
        # print("=========================in read_track_file_def, vdi_type: =========================",vdi_type)
        # print("=========================in read_track_file_def, route_type: =========================",route_type)
        if vdi_type == 'react':
            self._gt_csv_index = '5'
            track_file_path = os.path.join(self._gt_tracks_dir, "vehicle_tracks_" + str(self._gt_csv_index).zfill(3) + ".csv")
            track_dict = read_tracks(track_file_path)
            self._possible_ego_id = [41]
        elif vdi_type == 'record' and route_type == 'ground_truth': # 
            track_file_path = os.path.join(self._gt_tracks_dir, "vehicle_tracks_" + str(self._gt_csv_index).zfill(3) + ".csv")
            # track_file_path = dataset/recorded_trackfiles/DR_USA_Intersection_EP0/vehicle_tracks_xxx.csv
            track_dict = read_tracks(track_file_path)
            if not self._eval:
                # random choose
                # self._possible_ego_id = self._get_possible_ego_id()
                # choose in turn
                if not self._possible_ego_id:
                    self._possible_ego_id = self._get_possible_ego_id() # 获取可能的ego车辆id
                    # print("========self._get_possible_ego_id():========",self._get_possible_ego_id()) # 当前csv文件中可能的ego车辆id(去除不合理的车辆id和)
            else:
                if not self._possible_ego_id:
                    self._possible_ego_id = self._get_possible_ego_id()
        else:
            print('please check if the vdi_type and route_type are correct')
        
        # self._possible_ego_id = [21]

        return track_dict
    
    def select_ego(self):
        ego_id_list = []
        if not self._eval:  # 训练模式
            # random choose
            # ego_id_list = [random.choice(self._possible_ego_id) for _ in range(self._ego_num)]
            # choose in turn
            print('csv {} remain {} train ego id: {}'.format(self._gt_csv_index, len(self._possible_ego_id), self._possible_ego_id))
            ego_id_list.append(self._possible_ego_id[0]) # 选择第一个可能的ego id
            self._possible_ego_id.pop(0) # 移除已选择的id
        else:               # 评估模式
            for _ in range(self._ego_num): # 循环选择指定数量的ego
                print('csv {} remain {} eval ego id: {}'.format(self._gt_csv_index, len(self._possible_ego_id), self._possible_ego_id))
                ego_id_list.append(self._possible_ego_id[0])
                self._possible_ego_id.pop(0)
        
        # ego_id_list = [11]
        print('csv:', self._gt_csv_index, 'ego:', ego_id_list)
        return ego_id_list

    def get_start_timestamp(self):
        return None

        
# small_sclae dataset     
class PredictionLoader():
    def __init__(self, prediction_tracks_dir, gt_tracks_dir, ego_num, ego_max_length=5.5, eval=False):
        self._prediction_tracks_dir = prediction_tracks_dir
        self._gt_tracks_dir = gt_tracks_dir
        self._gt_csv_index = None
        self._eval = eval
        
        self._ego_max_length = ego_max_length
        self._ego_num = ego_num
        self._possible_ego_dict = dict()
        self._possible_ego_id = []
        
        # 初始化数据相关属性
        self._data = None
        self.data_file = None
        
        # 定义小数据集
        self.small_set_dict = {5: [29, 30, 33, 36, 37, 39, 40, 41]}
        
        # 初始化可能的ego车辆
        self._set_possible_ego_id_dict()

    @property
    def track_id(self):
        return self._gt_csv_index
    
    def _set_possible_ego_id_dict(self):
        for csv_index, vehicle_ids in self.small_set_dict.items():
            possible_ego_id = []
            # 使用 csv_index 而不是 self._gt_csv_index
            track_file_path = os.path.join(
                self._gt_tracks_dir, 
                "vehicle_tracks_" + str(csv_index).zfill(3) + ".csv"
            )
            track_dict = read_tracks(track_file_path)
            
            for vehicle_id in vehicle_ids:
                if vehicle_id in track_dict:
                    length_condition = track_dict[vehicle_id].length <= self._ego_max_length
                    time_condition = (track_dict[vehicle_id].time_stamp_ms_last - track_dict[vehicle_id].time_stamp_ms_first)/1000 >= 5
                    track_condition = get_track_length(track_dict[vehicle_id]) >= 20
                    
                    if length_condition and time_condition and track_condition:
                        possible_ego_id.append(vehicle_id)
                        
            self._possible_ego_dict[csv_index] = possible_ego_id
            # 初始化 self._gt_csv_index
            if self._gt_csv_index is None:
                self._gt_csv_index = csv_index

    def _get_possible_ego_id(self, track_dict):
        vehicle_id_list = track_dict.keys()     # 车辆id  track key
        possible_ego_id = []                    # 可能的ego车辆id
        for vehicle_id in vehicle_id_list:      # 遍历车辆id
            possible_ego_id.append(vehicle_id)  # ego id即为车辆id
        return possible_ego_id
    
    # train模式下，选择数据
    def _select_data_train(self, file_name=None):
        # extract data from prediction file
        if not file_name:                                       # 如果未指定文件名
            file_name = random.choice(self._prediciton_tracks)  # 随机选择文件，就是.pkl文件中随机选择一个
        data_file = os.path.join(self._prediction_tracks_dir, file_name) # 构建文件路径 {root_dir}/prediction/{map_name}/xxxxx.pkl
        with open(data_file, 'rb') as f:                        # 以二进制读取模式打开
            data = pickle.load(f)                               # 使用pickle加载数据 data为转换后的pkl数据
        self._possible_ego_id = self._get_possible_ego_id(data['egos_track']) # 获取ego车辆id
        return data_file, data

    # eval模式下，选择数据
    def _select_data_eval(self, data_file, data, file_name=None):
        if self._only_trouble: # false by default
            # if current data exist and still have trouble cars
            if self._ego_num == 1 and self._possible_ego_id: 
                pass
            # find next file which has trouble cars
            else:
                while True:
                    if not file_name:
                        file_name = self._prediciton_tracks[self._file_index]
                    data_file = os.path.join(self._prediction_tracks_dir, file_name)
                    with open(data_file, 'rb') as f:
                        data = pickle.load(f)
                    if data['gt_of_trouble']:
                        self._possible_ego_id = self._get_possible_ego_id(data['gt_of_trouble'])
                        break
                    else:
                        self._file_index += 1
        elif self._eval: # false by default
            if not data:
                file_name = '5_45800_55800_33P0.pkl'
                data_file = os.path.join(self._prediction_tracks_dir, file_name)
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
            # select every ego in turn when evaluate policy
            if not self._possible_ego_id:
                self._possible_ego_id = self._get_possible_ego_id(data['egos_track'])

                # self._possible_ego_id = [33,37,41]
                # self._possible_ego_id = [36, 41]
        
        return data_file, data

    # 获取ground truth数据的csv文件索引
    def _get_gt_csv_index(self, data_file):
        # get corresponding ground truth data from dataset
        gt_csv_index = data_file[data_file.find('DR_USA_Intersection_EP0') + len('DR_USA_Intersection_EP0') + 1]
        # 找到'DR_USA_Intersection_EP0'字符串在data_file的起始位置，然后加上字符串长度22，再加1，即为csv文件索引，换句话说就是得到DR_USA_Intersection_EP0_x中的x这个数字
        return gt_csv_index
    
    # 从文件中提取数据
    def _extract_data_from_files(self, file_name=None):
        # when evaluate policy, select every possible (only trouble or all) ego in turn
        if self._eval or self._only_trouble: # self._eval = False, self._only_trouble = False
            self.data_file, self._data = self._select_data_eval(self.data_file, self._data, file_name)
        else: # by default, train mode
            self.data_file, self._data = self._select_data_train(file_name) # 选择训练数据 self._select_data_train(file_name='5_45800_55800_33P0.pkl')
        # get ground truth data csv file index corresponding to the file, if we want to use gt data
        self._gt_csv_index = self._get_gt_csv_index(self.data_file)

    def change_track_file(self):
        if not self._eval:
            if self._gt_csv_index is None:
                self._gt_csv_index = list(self.small_set_dict.keys())[0]
            else:
                self._gt_csv_index = list(self.small_set_dict.keys())[0]
        else:
            if self._gt_csv_index is None:
                self._gt_csv_index = list(self.small_set_dict.keys())[0]
            else:
                self._gt_csv_index = list(self.small_set_dict.keys())[0]
                
        return self._gt_csv_index

    def read_track_file(self, vdi_type, route_type): # vdi_type = settings['vdi_type'], route_type = settings['route_type']
        if vdi_type == 'react':
            self._gt_csv_index = '5'
            track_file_path = os.path.join(self._gt_tracks_dir, "vehicle_tracks_" + str(self._gt_csv_index).zfill(3) + ".csv") # 
            # track_file_path = {self._gt_tracks_dir}/"vehicle_tracks_" + "005" + ".csv" = dataset/recorded_trackfiles/DR_USA_Intersection_EP0/vehicle_tracks_005.csv
            track_dict = read_tracks(track_file_path)
            self._possible_ego_id = [41] # force it to be 41
        
        elif vdi_type == 'record' and route_type == 'ground_truth':
            track_file_path = os.path.join(self._gt_tracks_dir, "vehicle_tracks_" + str(self._gt_csv_index).zfill(3) + ".csv")
            # track_file_path = {self._gt_tracks_dir}/"vehicle_tracks_" + "005" + ".csv" = dataset/recorded_trackfiles/DR_USA_Intersection_EP0/vehicle_tracks_005.csv
            track_dict = read_tracks(track_file_path)
            # if self._load_mode == 'both' or self._load_mode == 'pedestrian':
            #     self._pedestrian_dict = read_pedestrian(self._pedestrian_file_name)
        
        elif vdi_type == 'record' and route_type == 'predict':
            track_dict = read_trajectory(self.data_file)
            # track_dict = read_trajectory(self.data_file) = {track_id: track}
        
        print("vdi_type",vdi_type)
        print("route_type",route_type)
        print("in read_track_file_def, track_file_path: ",track_file_path) # /home/developer/workspace/interaction_gym/dataset/recorded_trackfiles/DR_USA_Intersection_EP0/vehicle_tracks_005.csv
        # print("in read_track_file_def, track_dick: ",track_dict)
        return track_dict

    def select_ego(self):
        ego_id_list = []
        if not self._eval:
            possible_egos = self._possible_ego_dict[self._gt_csv_index]
            ego_id_list.append(possible_egos[0])
            self._possible_ego_id = possible_egos[1:]
        else:
            possible_egos = self._possible_ego_dict[self._gt_csv_index]
            ego_id_list.append(possible_egos[0])
            self._possible_ego_id = possible_egos[1:]
            
        print("vehicle_tracks_" + str(self._gt_csv_index).zfill(3) + ".csv")
        return ego_id_list

    def get_start_timestamp(self):
        return None

    def get_ego_routes(self):
        ego_route_dict = dict()
        for ego_id in self.ego_id_list:
            ego_info = self._data['egos_track'][ego_id]
            ego_route_dict[ego_id] = ego_info[1:]
            print("self._data : ",self._data)
        return ego_route_dict
