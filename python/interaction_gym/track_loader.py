import os
import pickle
import random
import time
from copy import copy

from geometry import get_track_length
from utils.dataset_reader import read_trajectory, read_tracks, read_pedestrian

class DatasetLoader():
    # get all files in the directory according to the scenario setting
    def __init__(self, gt_tracks_dir, ego_max_length=5.5, ego_num=1, eval=False):
        self._gt_tracks_dir = gt_tracks_dir
        self._gt_tracks = os.listdir(self._gt_tracks_dir)
        self._gt_csv_index = None
        
        self._eval = eval

        self._ego_max_length = ego_max_length
        self._ego_num = ego_num
        self._possible_ego_dict = dict()
        self._possible_ego_id = []

        # remove troubled vehicles which can not be properly controlled
        self._cont_be_used_as_ego_dict = {0: [], 1: [], 2: [11], 3: [81], 4:[], 5: [57], 6:[], 7:[15, 67]}
        # remove small set data from big set data
        small_set_ego_dict = {5: [29, 30, 33, 36, 37, 39, 40, 41]}
        for csv, value in small_set_ego_dict.items():
            self._cont_be_used_as_ego_dict[csv].extend(value)

        # collect tracks that can be used as ego
        self._set_possible_ego_id_dict()
    
    @property
    def track_id(self):
        return self._gt_csv_index
    
    def _set_possible_ego_id_dict(self):
        tracks_num = len(self._gt_tracks)
        for csv_index in range(tracks_num):
            track_file_path = os.path.join(self._gt_tracks_dir, "vehicle_tracks_" + str(csv_index).zfill(3) + ".csv")
            track_dict = read_tracks(track_file_path)
            vehicle_id_list = track_dict.keys()
            possible_ego_id = []
            # some vehicles are not appropriate to use as ego
            for vehicle_id in vehicle_id_list:
                length_condition = track_dict[vehicle_id].length <= self._ego_max_length # ego should have a proper vehicle length
                time_condition = (track_dict[vehicle_id].time_stamp_ms_last - track_dict[vehicle_id].time_stamp_ms_first)/1000 >= 5 # ego should exist at least 5 seconds
                track_condition = get_track_length(track_dict[vehicle_id]) >= 20 # ego route should be longer than 20 meters
                if length_condition and time_condition and track_condition:
                    if vehicle_id not in self._cont_be_used_as_ego_dict[csv_index]:
                        possible_ego_id.append(vehicle_id)
            print('csv {} possible ego num: {}'.format(csv_index, len(possible_ego_id)))
            self._possible_ego_dict[csv_index] = possible_ego_id


    # if mode is train, then use the former 3/4 files, else the latter 1/4 files
    def _extract_data_from_file(self, random=False):
        tracks_num = len(self._gt_tracks)
        train_num = int(tracks_num * 0.75)
        # eval_num = tracks_num - train_num
        if not self._eval:
            # choose randomly from all tracks
            if random:
                self._gt_csv_index = random.randint(0, train_num - 1)
            # choose in turn
            else:
                if self._gt_csv_index is None:
                    self._gt_csv_index = 0
                else:
                    if not self._possible_ego_id:
                        self._gt_csv_index = 0 if self._gt_csv_index == train_num - 1 else int(self._gt_csv_index) + 1
        else: 
            if self._gt_csv_index is None:
                self._gt_csv_index = train_num
            else:
                if not self._possible_ego_id:
                    self._gt_csv_index = train_num if self._gt_csv_index == tracks_num - 1 else int(self._gt_csv_index) + 1
        

    def _get_possible_ego_id(self):
        return copy(self._possible_ego_dict[self._gt_csv_index])
            
    # change track file randomly in train mode, and in turn in eval mode
    def change_track_file(self):
        self._extract_data_from_file()
        return self._gt_csv_index

    def read_track_file(self, vdi_type, route_type):
        if vdi_type == 'react':
            self._gt_csv_index = '5'
            track_file_path = os.path.join(self._gt_tracks_dir, "vehicle_tracks_" + str(self._gt_csv_index).zfill(3) + ".csv")
            track_dict = read_tracks(track_file_path)
            self._possible_ego_id = [41]
        elif vdi_type == 'record' and route_type == 'ground_truth':
            track_file_path = os.path.join(self._gt_tracks_dir, "vehicle_tracks_" + str(self._gt_csv_index).zfill(3) + ".csv")
            track_dict = read_tracks(track_file_path)
            if not self._eval:
                # random choose
                # self._possible_ego_id = self._get_possible_ego_id()
                # choose in turn
                if not self._possible_ego_id:
                    self._possible_ego_id = self._get_possible_ego_id()
            else:
                if not self._possible_ego_id:
                    self._possible_ego_id = self._get_possible_ego_id()
        else:
            print('please check if the vdi_type and route_type are correct')
        
        # self._possible_ego_id = [21]

        return track_dict
    
    def select_ego(self):
        ego_id_list = []
        if not self._eval:
            # random choose
            # ego_id_list = [random.choice(self._possible_ego_id) for _ in range(self._ego_num)]
            # choose in turn
            print('csv {} remain {} train ego id: {}'.format(self._gt_csv_index, len(self._possible_ego_id), self._possible_ego_id))
            ego_id_list.append(self._possible_ego_id[0])
            self._possible_ego_id.pop(0)
        else:
            for _ in range(self._ego_num):
                print('csv {} remain {} eval ego id: {}'.format(self._gt_csv_index, len(self._possible_ego_id), self._possible_ego_id))
                ego_id_list.append(self._possible_ego_id[0])
                self._possible_ego_id.pop(0)
        
        # ego_id_list = [11]
        print('csv:', self._gt_csv_index, 'ego:', ego_id_list)
        return ego_id_list

    def get_start_timestamp(self):
        return None
    
class PredictionLoader():
    def __init__(self, prediction_tracks_dir, gt_tracks_dir, ego_num, ego_max_length=5.5, only_trouble=False, eval=False):
        # the directory path of prediction files
        self._prediction_tracks_dir = prediction_tracks_dir
        self._prediciton_tracks = os.listdir(self._prediction_tracks_dir)
        self._gt_tracks_dir = gt_tracks_dir
        # if only select trouble ones as ego in prediction files
        self._only_trouble = only_trouble 
        self._eval = eval

        self._data = {}
        self.data_file = None
        self._gt_csv_index = None

        self._file_index = 0

        self._ego_num = ego_num # how many vehicles are controlled by policy
        self._ego_max_length = ego_max_length
        self._possible_ego_id = []
        self.ego_id_list = []

    @property
    def track_id(self):
        return self._gt_csv_index
    
    def _get_possible_ego_id(self, track_dict):
        vehicle_id_list = track_dict.keys()
        possible_ego_id = []
        for vehicle_id in vehicle_id_list:
            possible_ego_id.append(vehicle_id)
        return possible_ego_id
    
    def _select_data_train(self, file_name=None):
        # extract data from prediction file
        if not file_name: # random select
            file_name = random.choice(self._prediciton_tracks)
        data_file = os.path.join(self._prediction_tracks_dir, file_name)
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        self._possible_ego_id = self._get_possible_ego_id(data['egos_track'])
        return data_file, data

    def _select_data_eval(self, data_file, data, file_name=None):
        if self._only_trouble:
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
        elif self._eval:
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

    def _get_gt_csv_index(self, data_file):
        # get corresponding ground truth data from dataset
        gt_csv_index = data_file[data_file.find('DR_USA_Intersection_EP0') + len('DR_USA_Intersection_EP0') + 1]
        return gt_csv_index
    
    def _extract_data_from_files(self, file_name=None):
        # when evaluate policy, select every possible (only trouble or all) ego in turn
        if self._eval or self._only_trouble:
            self.data_file, self._data = self._select_data_eval(self.data_file, self._data, file_name)
        else:
            self.data_file, self._data = self._select_data_train(file_name)
        # get ground truth data csv file index corresponding to the file, if we want to use gt data
        self._gt_csv_index = self._get_gt_csv_index(self.data_file)
    

    def change_track_file(self, file_name=None):
        self._extract_data_from_files(file_name)
        return self._gt_csv_index

    def read_track_file(self, vdi_type, route_type):
        if vdi_type == 'react':
            self._gt_csv_index = '5'
            track_file_path = os.path.join(self._gt_tracks_dir, "vehicle_tracks_" + str(self._gt_csv_index).zfill(3) + ".csv")
            track_dict = read_tracks(track_file_path)
            self._possible_ego_id = [41] # force it to be 41
        elif vdi_type == 'record' and route_type == 'ground_truth':
            track_file_path = os.path.join(self._gt_tracks_dir, "vehicle_tracks_" + str(self._gt_csv_index).zfill(3) + ".csv")
            track_dict = read_tracks(track_file_path)
            # if self._load_mode == 'both' or self._load_mode == 'pedestrian':
            #     self._pedestrian_dict = read_pedestrian(self._pedestrian_file_name)
        elif vdi_type == 'record' and route_type == 'predict':
            track_dict = read_trajectory(self.data_file)
        return track_dict

    def select_ego(self):
        self.ego_id_list = []
        # for Test, we either select troubled(i.e. collision/rotate/acc exceed) vehicles
        # or select all possible ego vehicles in turn
        if self._only_trouble:
            while True:
                self.ego_id_list.append(self._possible_ego_id[0])
                self._possible_ego_id.pop(0)
                if not self._possible_ego_id or len(self.ego_id_list) == self._ego_num:
                    break
        elif self._eval:
            print('possible ego id list:', self._possible_ego_id)
            self.ego_id_list.append(self._possible_ego_id[0])
            self._possible_ego_id.pop(0)
        # for Training, we random select 1 vehicle within all vehicles
        else:
            self.ego_id_list.append(random.choice(self._possible_ego_id))
        print('csv:', self._gt_csv_index, 'ego:', self.ego_id_list)
        return self.ego_id_list

    def get_start_timestamp(self):
        for info in self._data['egos_track'].values():
            ego_start_timestamp_list = [info[0][0]]
        return ego_start_timestamp_list

    def get_ego_routes(self):
        ego_route_dict = dict()
        for ego_id in self.ego_id_list:
            ego_info = self._data['egos_track'][ego_id]
            ego_route_dict[ego_id] = ego_info[1:]

        return ego_route_dict
