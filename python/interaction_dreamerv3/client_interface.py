import sys
import pathlib
import os
import zmq
import numpy as np
import random
import gym
import time
from datetime import datetime
import pickle
import collections

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

#将字典转换为对象
class Dict2Class(object):

    def __init__(self, dict):
        for key in dict:
            setattr(self, key, dict[key])

class ClientInterface(object):

    def __init__(self, args, record=True):
        # env args
        self.args = Dict2Class(args) if isinstance(args, dict) else args # 这句之后，后面就可以用self.args.xxx来访问参数了
        self.discrete_action_num = 4
        self.action_space = gym.spaces.Discrete(self.discrete_action_num)

        # connection with I-SIM server
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        print("connecting to interaction gym...")
        url = ':'.join(["tcp://localhost", str(self.args.port)]) # "tcp://localhost:5561”
        self._socket.connect(url)
        
        # simulator statue flags
        self.env_init_flag = False
        self.can_change_track_file_flag = False
        self.scen_init_flag = False
        self.env_reset_flag = False

        self._gt_csv_index = None
        self.ego_id_list = None

        time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # record episode data if needed to calculate metrics
        self._record = record
        self.run_data_dict = dict()
        self._run_filename = "run " + time_string + '.pkl'
        # record prediction data if possible to calculate ade
        self.prediction_data_dict = dict()
        self._prediction_filename = "prediction " + time_string + '.pkl'

    def __del__(self):
        self._socket.close()

    def _env_initialize(self, args):
        # settings in dict format
        settings = vars(args)
        # send to env
        message_send = {'command': 'env_init', 'content': settings}
        self._socket.send_string(str(message_send))
        # recieve from env
        message_recv = self._socket.recv_string()

        if message_recv == 'env_init_done':
            self.env_init_flag = True
            print('env init done')

    def _change_track_file(self):
        # send to env
        # for simple react scenarios, the track file is fixed
        message_send = {'command': 'track_init'}
        self._socket.send_string(str(message_send))
        # recieve from env
        message_recv = self._socket.recv()
        str_message = bytes.decode(message_recv)
        self._gt_csv_index = eval(str_message)
        self.can_change_track_file_flag = False

    def _scenario_initialize(self):
        # send to env
        message_send = {'command': 'scen_init'}
        self._socket.send_string(str(message_send))

        # recieve from env
        message_recv = self._socket.recv()
        str_message = bytes.decode(message_recv)

        if str_message == 'wrong_num':
            print(message_recv)
        else:
            self.ego_id_list = eval(str_message)
            self.scen_init_flag = True
    
    def _reset_prepare(self):
        # init env
        if not self.env_init_flag:
            self._env_initialize(self.args)
        else:
            # change track file number
            if self.can_change_track_file_flag and not self.scen_init_flag:
                self._change_track_file()
            # choose ego vehicle and map init
            elif not self.scen_init_flag:
                self._scenario_initialize()
            # several depandecies have been checked, can reset environment
            elif self.scen_init_flag and not self.env_reset_flag:
                return True

        return False

    def _observation_to_ego_dict_array(self, observation, ego_id_list, npc_id_dict, other_id_dict):
        if self.args.state_frame == 'ego':
            needed_state = ['vector_past_vehicle_state_ego_frame'] # ['vector_map']
        elif self.args.state_frame == 'global':
            needed_state = ['vector_past_vehicle_state'] # ['vector_map']

        # initialize ego state dict
        ego_state_dict = dict()
        for ego_id in ego_id_list:
            ego_state_dict[ego_id] = dict()
        # for every ego vehicle(since there may be multiple ego vehicles)
        for ego_id in ego_id_list:
            # npc_1 ~ npc_n, and mask_npc indicates their existence
            npc_index_set = set('npc_' + str(i) for i in range(1, self.args.npc_num + 1))
            mask_npc = np.zeros(len(npc_index_set), dtype=np.int32)
            is_new_npc = np.zeros(len(npc_index_set), dtype=np.int32)
            id_npc = np.zeros(len(npc_index_set), dtype=np.int32)
            # other_1 ~ other_n, also has mask
            other_index_set = set('other_' + str(i) for i in range(1, self.args.other_num + 1))
            mask_other = np.zeros(len(other_index_set), dtype=np.int32)
            is_new_other = np.zeros(len(other_index_set), dtype=np.int32)
            id_other = np.zeros(len(other_index_set), dtype=np.int32)
            
            # current surrounding vehicles' id, from closet to farest (distance), which is devided into 2 parts
            surrounding_vehicles_id = observation['surrounding_vehicles_id'][ego_id]
            # print(surrounding_vehicles_id)
            npc_id_distance = list(surrounding_vehicles_id)[:self.args.npc_num]
            other_id_distance = list(surrounding_vehicles_id)[self.args.npc_num:self.args.npc_num + self.args.other_num]
            # print(npc_id_distance, other_id_distance)
                
            # get state for every ego vehicle
            for state_name in needed_state:
                state = observation[state_name]
                # vehicle state
                if state_name.startswith('vector_past_vehicle_state'):
                    # 1. remove or transfer(actually is also remove) vehicles' id from npc_id_dict(id: npc_*) and other_id_dict(id: other_*)
                    remove_npc_id_set = set(npc_id_dict.keys()) - set(npc_id_distance)
                    for remove_npc_id in remove_npc_id_set:
                        npc_id_dict.pop(remove_npc_id)
                    remove_other_id_set = set(other_id_dict.keys()) - set(other_id_distance)
                    for remove_other_id in remove_other_id_set:
                        other_id_dict.pop(remove_other_id)

                    # 2. update ego_state_dict
                    for vehicle_id, traj_state in state.items():
                        vector_state = np.reshape(traj_state, [19, 5])
                        # ego state
                        if vehicle_id == ego_id: 
                            ego_state_dict[ego_id]['ego'] = vector_state
                        # npc and other vehicle state
                        else: 
                            # if this vehicle has been in the state dict in last step and in ego's dectect range
                            if vehicle_id in npc_id_dict.keys():
                                npc_index = npc_id_dict[vehicle_id]
                                ego_state_dict[ego_id][npc_index] = vector_state
                            elif vehicle_id in other_id_dict.keys():
                                other_index = other_id_dict[vehicle_id]
                                ego_state_dict[ego_id][other_index] = vector_state
                            # if this vehicle is new for state dict, then put it in the state dict if there is a room
                            else:
                                if len(npc_id_dict.keys()) < self.args.npc_num and vehicle_id in npc_id_distance:
                                    feasible_npc_index = random.choice(list(npc_index_set - set(npc_id_dict.values())))
                                    is_new_npc[int(feasible_npc_index[-1]) - 1] = 1
                                    npc_id_dict[vehicle_id] = feasible_npc_index
                                    ego_state_dict[ego_id][feasible_npc_index] = vector_state
                                elif len(other_id_dict.keys()) < self.args.other_num and vehicle_id in other_id_distance:
                                    feasible_other_index = random.choice(list(other_index_set - set(other_id_dict.values())))
                                    is_new_other[int(feasible_other_index[-1]) - 1] = 1
                                    other_id_dict[vehicle_id] = feasible_other_index
                                    ego_state_dict[ego_id][feasible_other_index] = vector_state

                    # 3. update npc_mask and other_mask
                    for vehicle_id, npc_index in npc_id_dict.items():
                        mask_npc[int(npc_index[-1]) - 1] = 1
                        id_npc[int(npc_index[-1]) - 1] = vehicle_id
                    for vehicle_id, other_index in other_id_dict.items():
                        mask_other[int(other_index[-1]) - 1] = 1
                        id_other[int(other_index[-1]) - 1] = vehicle_id

                    # 4. complete state dict if there are not enough vehicles around
                    while len(ego_state_dict[ego_id].keys()) < self.args.npc_num + self.args.other_num + 1:
                        current_npc_index_set = set([key for key in ego_state_dict[ego_id].keys() if 'npc' in key])
                        current_other_index_set = set([key for key in ego_state_dict[ego_id].keys() if 'other' in key])
                        if len(current_npc_index_set) < self.args.npc_num:
                            padding_npc_index = list(npc_index_set - current_npc_index_set)[0]
                            ego_state_dict[ego_id][padding_npc_index] = np.zeros([19, 5])
                        elif len(current_other_index_set) < self.args.other_num:
                            padding_other_index = list(other_index_set - current_other_index_set)[0]
                            ego_state_dict[ego_id][padding_other_index] = np.zeros([19, 5])

                    # 5. plus mask and other npc padding related state in state dict
                    ego_state_dict[ego_id]['mask_npc'] = mask_npc
                    ego_state_dict[ego_id]['id_npc'] = id_npc
                    ego_state_dict[ego_id]['mask_other'] = mask_other
                    ego_state_dict[ego_id]['id_other'] = id_other

                    ego_state_dict[ego_id]['should_init_npc'] = []
                    for i in range(len(mask_npc)):
                        if not mask_npc[i] or is_new_npc[i]:
                            ego_state_dict[ego_id]['should_init_npc'].append(1)
                        else:
                            ego_state_dict[ego_id]['should_init_npc'].append(0)

                    ego_state_dict[ego_id]['should_init_other'] = []
                    for i in range(len(mask_other)):
                        if not mask_other[i] or is_new_other[i]:
                            ego_state_dict[ego_id]['should_init_other'].append(1)
                        else:
                            ego_state_dict[ego_id]['should_init_other'].append(0)

                    # 6. from closet to farest (distance) of npc/other vehicles index
                    ego_state_dict[ego_id]['index_npc'] = []
                    for vehicle_id in npc_id_distance:
                        ego_state_dict[ego_id]['index_npc'].append(npc_id_dict[vehicle_id])
                    ego_state_dict[ego_id]['index_other'] = []
                    for vehicle_id in other_id_distance:
                        ego_state_dict[ego_id]['index_other'].append(other_id_dict[vehicle_id])
                    
                    # print('mask_npc:', mask_npc)
                    # print('id_npc:', id_npc)
                    # print('is_new_npc:', is_new_npc)
                    # print('should_init_npc:', ego_state_dict[ego_id]['should_init_npc'])
                    # print(npc_id_distance, npc_id_dict, ego_state_dict[ego_id]['index_npc'])

                    # print('mask_other:', mask_other)
                    # print('id_other:', id_other)
                    # print('is_new_other:', is_new_other)
                    # print('should_init_other:', ego_state_dict[ego_id]['should_init_other'])
                    # print(other_id_distance, other_id_dict, ego_state_dict[ego_id]['index_other'])

                    # for i in range(len(npc_id_distance)):
                    #     if not npc_id_dict[npc_id_distance[i]] == ego_state_dict[ego_id]['index_npc'][i]:
                    #         print('oops')
                    # for i in range(len(other_id_distance)):
                    #     if not other_id_dict[other_id_distance[i]] == ego_state_dict[ego_id]['index_other'][i]:
                    #         print('ooooops')
                    # print('__________________________________________________')

                # map state
                # TODO: didnt consider map for now
                elif state_name == 'vector_map':
                    ego_state_dict[ego_id]['map'] = np.reshape(state, [-1, 4])


            # print('mask:', mask_npc)
            # print('id:',id_npc)
            # print('id-order dict:', npc_id_dict)

            # print('id list close to far: ', observation['interaction_vehicles_id'][ego_id])
            # print('order list close to far: ', ego_state_dict[ego_id]['order_npc'])
            # print(ego_state_dict[ego_id].keys())
                
        return ego_state_dict, npc_id_dict, other_id_dict
    
    def reset(self):
        # reset flags
        self.can_change_track_file_flag = True  # this is used for multi-track-file random selection
        self.scen_init_flag = False
        self.env_reset_flag = False

        while not self._reset_prepare():
            self._reset_prepare()

        # send to env
        message_send = {'command': 'reset'}
        self._socket.send_string(str(message_send))
        # recieve from env
        message_recv = self._socket.recv()
        message_recv = eval(bytes.decode(message_recv))
        if isinstance(message_recv, dict):
            self.env_reset_flag = True
            observation = message_recv['observation']
        else:
            self.scen_init_flag = False
        
        # record npc id and index
        self._npc_id_dict = {}
        self._other_id_dict = {}
        state_dict_array, self._npc_id_dict, self._other_id_dict = self._observation_to_ego_dict_array(observation, self.ego_id_list, self._npc_id_dict, self._other_id_dict)

        # record prediction data
        self.ep_prediction_data = {}
        self.ep_prediction_data.update({'ego_gt': {}})

        # record run data for analyse
        if self._record:
            # initialize the record
            self.run_data_dict[self._gt_csv_index] = dict() if self._gt_csv_index not in self.run_data_dict.keys() else self.run_data_dict[self._gt_csv_index]
            self.run_data_dict[self._gt_csv_index][self.ego_id_list[0]] = list() if self.ego_id_list[0] not in self.run_data_dict[self._gt_csv_index].keys() else self.run_data_dict[self._gt_csv_index][self.ego_id_list[0]]
            self.run_data_dict[self._gt_csv_index][self.ego_id_list[0]].append(collections.defaultdict(list))
            # fill the record
            self.run_record = self.run_data_dict[self._gt_csv_index][self.ego_id_list[0]][-1]
            self.run_record['speed'].append(0.)
            self.run_record['completion_rate'].append(0.)
            self.run_record['gt_distance'].append(0.)
            self.run_record['collision'].append(0)
            self.run_record['success'].append(0)

        return state_dict_array

    def step(self, action_dict, prediction=None):
        # send current action to env, consider visualize vehicle trajectory prediction
        content_dict = action_dict
        content_dict.update({'prediction': prediction.tolist() if prediction is not None else prediction})
        message_send = {'command': 'step', 'content': content_dict}
        self._socket.send_string(str(message_send))
        # recieve next observation, reward, done and aux info from env
        message_recv = self._socket.recv()
        message_recv = eval(bytes.decode(message_recv))

        observation = message_recv['observation']
        reward_dict = message_recv['reward']
        done_dict = message_recv['done']
        aux_info_dict = message_recv['aux_info']

        # record prediction data for ade analyse
        if prediction is not None:
            for ego_id in self.ego_id_list:
                # print(self._npc_id_dict)
                # pridciton in global frame
                prediction_global_index = aux_info_dict[ego_id].pop('prediction_global')
                time = list(prediction_global_index.keys())[0]
                # print(time)
                prediction_global_id = {ego_id: prediction_global_index[time][0]} # ego's index is always 0
                # if prediction_global_index[time][0] is None:
                #     print('find it')
                #     print(0)
                #     print(prediction_global_index[time][0])
                #     print(prediction)
                for npc_id in self._npc_id_dict:
                    npc_index = int(self._npc_id_dict[npc_id][-1])
                    prediction_global_id.update({npc_id: prediction_global_index[time][npc_index]})
                    # if prediction_global_index[time][npc_index] is None:
                    #     print('find it')
                    #     print(npc_index)
                    #     print(prediction_global_index[time][npc_index])
                    #     print(prediction)
                self.ep_prediction_data.update({time: prediction_global_id})
                # ego ground truth location
                ego_gt_loc = {time: aux_info_dict[ego_id].pop('ego_loc')}
                self.ep_prediction_data['ego_gt'].update(ego_gt_loc)

                # print(self.ep_prediction_data.keys())
                # print(len(self.ep_prediction_data['ego_gt']), self.ep_prediction_data['ego_gt'])
                # print(aux_info_dict[ego_id]['track_id'])
            # save file to disk
            all_done = False not in done_dict.values()
            if all_done:
                track_id = aux_info_dict[ego_id].pop('track_id')
                if track_id not in self.prediction_data_dict.keys():
                    self.prediction_data_dict[track_id] = {}
                data_index = len((self.prediction_data_dict[track_id].keys()))
                self.prediction_data_dict[track_id].update({data_index: self.ep_prediction_data})
                with open(self._prediction_filename, 'wb') as f:
                    pickle.dump(self.prediction_data_dict, f)
        
        # record episode data for analyse
        if self._record:        
            # fill the record
            ego_id = self.ego_id_list[0]
            aux_info = aux_info_dict[ego_id]
            self.run_record['speed'].append(aux_info['speed'])
            self.run_record['completion_rate'].append(aux_info['completion_rate'])
            self.run_record['gt_distance'].append(aux_info['distance_to_gt'])
            self.run_record['collision'].append(aux_info['result'] == 'collision')
            self.run_record['success'].append(aux_info['result'] == 'success')
            # save file to disk
            all_done = False not in done_dict.values()
            if all_done:
                with open(self._run_filename, 'wb') as f:
                    pickle.dump(self.run_data_dict, f)
                    print('file saved')
                
        # record npc id
        state_dict_array, self._npc_id_dict, self._other_id_dict = self._observation_to_ego_dict_array(observation, self.ego_id_list, self._npc_id_dict, self._other_id_dict)
        
        return state_dict_array, reward_dict, done_dict, aux_info_dict

