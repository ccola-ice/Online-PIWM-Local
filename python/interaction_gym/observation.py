#-*- coding: UTF-8 -*- 
import math
import matplotlib
import numpy as np
import heapq
from collections import defaultdict

try:
    import lanelet2
    import lanelet2_matching
    print("Lanelet2_matching import")
except:
    import warnings
    string = "Could not import lanelet2_matching"
    warnings.warn(string)
import geometry

class Observation:
    def __init__(self, settings, interaction_map, map_centerline_list, controlled_vehicles_dict, normalization=False):
        self._map = interaction_map
        self._map_centerline_list = map_centerline_list
        self._ego_vehicles_dict = controlled_vehicles_dict['ego']
        self._react_npc_vehicle_dict = controlled_vehicles_dict['npc']

        self._npc_num = settings['npc_num'] + settings['other_num']
        self._control_steering = settings['control_steering']

        # routes related info
        self.ego_route_dict = dict()
        self.ego_route_lanelet_dict = dict()
        self.ego_closet_bound_points = dict()

        self.react_npc_route_dict = dict()

        # termination judgement
        self.reach_goal = dict()
        self.collision = dict()
        self.deflection = dict()

        # observation terms
        self.trajectory_distance = dict()
        self.trajectory_location = dict()
        self.trajectory_speed = dict()
        
        self.distance_from_bound = dict()
        self.lane_observation = dict()

        self.future_route_points = dict()

        self.ego_shape = dict()
        self.react_npc_shape = dict()

        # self.ego_route_points = dict()
        # self.ego_route_target_speed = dict()
        self.ego_route_left_bound_points = dict()
        self.ego_route_right_bound_points = dict()

        self.react_npc_route_points = dict()

        self.ego_current_speed = dict()
        self.target_speed = dict()
        self.ego_next_loc = dict()

        self.vector_map = list()
        # self.vector_past_vehicle_state = dict() # list of [x_t-1, y_t-1, x_t, y_t, theta_t] in global frame, for now
        # self.vector_past_vehicle_state_ego_frame = dict() # list of [x_t-1, y_t-1, x_t, y_t, theta_t] in ego frame

        self.interaction_vehicles_id = dict()
        self.interaction_vehicles_observation = dict()
        self.attention_mask = dict()

        self.observation_dict = dict()

        # init
        for ego_id in self._ego_vehicles_dict.keys():
            self.reach_goal[ego_id] = False
            self.collision[ego_id] = False
            self.deflection[ego_id] = False

        self.virtual_route_bound = True
        self.normalization = normalization

        self.min_interval_distance = 2 # minmum interval distance of waypoint (meter)
        

    def reset(self, route_type, ego_route_dict, react_npc_route_dict, route_lanelet_dict=None):
        self.route_type = route_type
        self.ego_route_dict = ego_route_dict
        self.react_npc_route_dict = react_npc_route_dict

        self.ego_route_lanelet_dict = route_lanelet_dict

        for ego_id in self._ego_vehicles_dict.keys():
            # self.reach_goal_lanelet[k] = False
            self.reach_goal[ego_id] = False
            self.collision[ego_id] = False
            self.deflection[ego_id] = False

        self.trajectory_distance.clear()
        self.trajectory_location.clear()
        self.trajectory_speed.clear()

        self.ego_shape.clear()
        self.react_npc_shape.clear()

        # self.ego_route_points.clear()
        # self.ego_route_target_speed.clear()
        self.ego_route_left_bound_points.clear()
        self.ego_route_right_bound_points.clear()
        self.ego_closet_bound_points.clear()

        self.react_npc_route_points.clear()

        self.distance_from_bound.clear()
        self.lane_observation.clear()
        self.future_route_points.clear()
        self.ego_next_loc.clear()

        self.ego_current_speed.clear()
        self.target_speed.clear()

        # vecterize location point list
        self.vector_map = []
        for centerline_list in self._map_centerline_list:
            self.vector_map.append(geometry.vectorize_point_list(centerline_list, has_heading=False))
        # self.vector_past_vehicle_state.clear()
        # self.vector_past_vehicle_state_ego_frame.clear() 

        self.interaction_vehicles_id.clear()
        self.interaction_vehicles_observation.clear()
        self.attention_mask.clear()

        self.observation_dict.clear()

    def get_interaction_vehicles_id_and_observation(self, ego_state_dict, vehicles_state_dict, detective_range=50):
        loc = ego_state_dict['loc']
        speed = ego_state_dict['speed']
        heading = ego_state_dict['heading']

        surrounding_list = []
        # 1. check if this vehicle within ego's detective range, and put them together
        for veh_id, veh_state_dict in vehicles_state_dict.items():
            # motion state
            veh_loc = veh_state_dict['loc']
            veh_speed = veh_state_dict['speed']
            veh_heading = veh_state_dict['heading']

            distance = math.sqrt((veh_loc[0] - loc[0])**2 + (veh_loc[1] - loc[1])**2)
            y_relative = (veh_loc[1] - loc[1])*np.sin(heading) + (veh_loc[0] - loc[0])*np.cos(heading)
            # TODO: detective range should be a hyperameter
            if distance <= detective_range and y_relative > -12:
                add_dict = {'vehicle_id': veh_id, 'distance': distance, 'loc': veh_loc, 'speed': veh_speed, 'heading': veh_heading}
                surrounding_list.append(add_dict)

        # 2. get interaction vehicles and their basic observation
        interaction_list = heapq.nsmallest(self._npc_num, surrounding_list, key=lambda s: s['distance'])

        # 3. get their ids and full observation
        interaction_id_list = []
        interaction_observation_list = []
        for vehicle_dict in interaction_list:
            # id
            interaction_id_list.append(vehicle_dict['vehicle_id'])
            # basic observation
            # shape
            vehicle_polygan = vehicles_state_dict[vehicle_dict['vehicle_id']]['polygon']
            poly_01 = [i - j for i, j in zip(vehicle_polygan[0], vehicle_polygan[1])]
            poly_12 = [i - j for i, j in zip(vehicle_polygan[1], vehicle_polygan[2])]
            vehicle_length = math.sqrt(poly_01[0]**2 + poly_01[1]**2)
            vehicle_width = math.sqrt(poly_12[0]**2 + poly_12[1]**2)
            # motion state
            vehicle_loc = vehicle_dict['loc']
            vehicle_speed = vehicle_dict['speed']
            vehicle_heading = vehicle_dict['heading']

            # ture observation
            x_in_ego_frame = (vehicle_loc[1] - loc[1])*np.cos(heading) - (vehicle_loc[0] - loc[0])*np.sin(heading)
            y_in_ego_frame = (vehicle_loc[1] - loc[1])*np.sin(heading) + (vehicle_loc[0] - loc[0])*np.cos(heading)
            heading_error_with_ego = vehicle_heading - heading

            single_observation = [vehicle_length, vehicle_width, x_in_ego_frame, y_in_ego_frame, vehicle_speed, np.cos(heading_error_with_ego), np.sin(heading_error_with_ego)]
            # loc_loc, loc_heading = geometry.localize_transform(loc, heading, vehicle_loc, vehicle_heading)
            # global_loc, global_heading = geometry.delocalize_transform(loc, heading, loc_loc, loc_heading)
            # print(loc_loc == single_observation[2:4], loc_heading == heading_error_with_ego)
            # print(global_loc == vehicle_loc, round(global_heading, 2) == round(vehicle_heading,2))
            interaction_observation_list += single_observation

        # 4. zero padding and attention mask
        # TODO: the len of single ob should be a hyperparameter
        attention_mask = list(np.ones(self._npc_num + 1))
        npc_obs_size = self._npc_num * 7 # len(single_observation)
        if len(interaction_observation_list) < npc_obs_size:
            zero_padding_num = int((npc_obs_size - len(interaction_observation_list)) / 7) # len(single_observation))
            for _ in range(zero_padding_num):
                attention_mask.pop()
            for _ in range(zero_padding_num):
                attention_mask.append(0)
            while len(interaction_observation_list) < npc_obs_size:
                interaction_observation_list.append(0)

        return interaction_id_list, interaction_observation_list, attention_mask

    def get_surrounding_vehicle_id(self, observation_dict):
        intersection_vehicle_id = []
        for ego_id in self._ego_vehicles_dict.keys():
            intersection_vehicle_id += observation_dict['surrounding_vehicles_id'][ego_id]

        return intersection_vehicle_id

    def get_future_route_points(self, observation_dict):
        ego_future_route_points_dict = dict()
        react_ego_future_route_points_dict = dict()
        for ego_id in self._ego_vehicles_dict.keys():
            ego_future_route_points_dict[ego_id] = observation_dict['future_route_points'][ego_id]
        for react_npc_id in self._react_npc_vehicle_dict.keys():
            react_ego_future_route_points_dict[react_npc_id] = observation_dict['future_route_points'][react_npc_id]

        return ego_future_route_points_dict, react_ego_future_route_points_dict

    def get_current_bound_points(self, observation_dict):
        current_bound_points = []
        for ego_id in self._ego_vehicles_dict.keys():
            current_bound_points += observation_dict['current_bound_points'][ego_id]

        return current_bound_points

    def check_reach_goal(self, state_dict, goal_point):
        loc_x = state_dict['loc'][0]
        loc_y = state_dict['loc'][1]

        goal_loc_x = goal_point[0]
        goal_loc_y = goal_point[1]

        goal_distance = math.sqrt((loc_x - goal_loc_x)**2 + (loc_y - goal_loc_y)**2)
        return goal_distance < 2

    def check_collision(self, ego_state_dict, vehicles_state_dict, interaction_vehicles_id):
        for veh_id, veh_state_dict in vehicles_state_dict.items():
            if veh_id in interaction_vehicles_id:
                distance, collision = geometry.ego_other_distance_and_collision(ego_state_dict, veh_state_dict)
                if collision:
                    return True
        return False

    def check_ego_deflection(self, virtual_route_bound, limitation, distance_bound=None, distance_to_center=None, ego_y_in_point_axis=None):
        deflection = False
        if virtual_route_bound:
            if distance_to_center > limitation or ego_y_in_point_axis > 0:
                deflection = True
        else:
            if distance_bound < limitation:
                deflection = True
        return deflection

    # TODO: the needed observation terms should be set in the beginning, avoid wasting calculation
    # TODO: code below only consider one ego
    # TODO: detection_range should be a hyper parameter which is seted at the beginning
    def get_scalar_observation(self, current_time, detection_range=True):
        vector_past_vehicle_state = dict()
        vector_past_vehicle_state_ego_frame = dict()
        # ego observation
        for ego_id, ego_state in self._ego_vehicles_dict.items():
            # get ego shape, polygon and motion states
            ego_state_dict = dict()
            ego_state_dict['loc'] = [ego_state._current_state.x, ego_state._current_state.y]
            ego_state_dict['speed'] = math.sqrt(ego_state._current_state.vx ** 2 + ego_state._current_state.vy ** 2)
            ego_state_dict['heading'] = ego_state._current_state.psi_rad
            ego_state_dict['polygon'] = self._map.ego_polygon_dict[ego_id]

            # ego shape
            self.ego_shape[ego_id] = [ego_state._length, ego_state._width]

            # ego current speed value
            self.ego_current_speed[ego_id] = [ego_state_dict['speed']]
            
            # get other vehicles' state, including other egos, react npcs, and log/record npcs
            other_vehicles_state_dict = dict()
            for other_ego_id, other_ego_state in self._ego_vehicles_dict.items(): # other egos
                if other_ego_id == ego_id:
                    continue
                else:
                    other_vehicles_state_dict[other_ego_id] = dict()
                    other_vehicles_state_dict[other_ego_id]['loc'] = [other_ego_state._current_state.x, other_ego_state._current_state.y]
                    other_vehicles_state_dict[other_ego_id]['speed'] = math.sqrt(other_ego_state._current_state.vx ** 2 + other_ego_state._current_state.vy ** 2)
                    other_vehicles_state_dict[other_ego_id]['heading'] = other_ego_state._current_state.psi_rad
                    other_vehicles_state_dict[other_ego_id]['polygon'] = self._map.ego_polygon_dict[other_ego_id]
            for react_npc_id, react_npc_state in self._react_npc_vehicle_dict.items(): # react npcs
                other_vehicles_state_dict[react_npc_id] = dict()
                other_vehicles_state_dict[react_npc_id]['loc'] = [react_npc_state._current_state.x, react_npc_state._current_state.y]
                other_vehicles_state_dict[react_npc_id]['speed'] = math.sqrt(react_npc_state._current_state.vx ** 2 + react_npc_state._current_state.vy ** 2)
                other_vehicles_state_dict[react_npc_id]['heading'] = react_npc_state._current_state.psi_rad
                other_vehicles_state_dict[react_npc_id]['polygon'] = self._map.react_npc_polygon_dict[react_npc_id]
            for record_npc_id, record_npc_motion_state in self._map.record_npc_motion_state_dict.items(): # log/record npcs
                other_vehicles_state_dict[record_npc_id] = dict()
                other_vehicles_state_dict[record_npc_id]['loc'] = [record_npc_motion_state.x, record_npc_motion_state.y]
                other_vehicles_state_dict[record_npc_id]['speed'] = math.sqrt(record_npc_motion_state.vx ** 2 + record_npc_motion_state.vy ** 2)
                other_vehicles_state_dict[record_npc_id]['heading'] = record_npc_motion_state.psi_rad
                other_vehicles_state_dict[record_npc_id]['polygon'] = self._map.record_npc_polygon_dict[record_npc_id]

            # get current ego route point and target speed
            ego_route_points = geometry.get_route_point_with_heading_from_point_list(self.ego_route_dict[ego_id], self.min_interval_distance)
            ego_route_target_speed = geometry.get_target_speed_from_point_list(self.ego_route_dict[ego_id])
            if self.route_type == 'predict':
                # do not need lane bound and distance if use predict route (for now)
                self.ego_route_left_bound_points[ego_id], self.ego_route_right_bound_points[ego_id] = None, None
                self.ego_closet_bound_points[ego_id], self.distance_from_bound[ego_id] = None, None
                pass
            elif self.route_type == 'ground_truth':
                # get current lane bound and distance from lanelet
                # self.ego_route_left_bound_points[ego_id], self.ego_route_right_bound_points[ego_id] = geometry.get_route_bounds_points(self.ego_route_lanelet_dict[ego_id], self.min_interval_distance)
                # self.ego_closet_bound_points[ego_id], self.distance_from_bound[ego_id] = geometry.get_closet_bound_point(ego_state_dict['loc'], self.ego_route_left_bound_points[ego_id], self.ego_route_right_bound_points[ego_id])
                pass
            elif self.route_type == 'centerline':
                # ego_route_points = geometry.get_ego_route_point_with_heading_from_lanelet(route_lanelet, self.min_interval_distance)
                
                # get current lane bound and distance
                # self.ego_route_left_bound_points[ego_id], self.ego_route_right_bound_points[ego_id] = geometry.get_route_bounds_points(route_lanelet, self.min_interval_distance)
                # self.ego_closet_bound_points[ego_id], self.distance_from_bound[ego_id] = geometry.get_closet_bound_point(ego_state_dict['loc'], self.ego_route_left_bound_points[ego_id], self.ego_route_right_bound_points[ego_id])
                pass
            
            # ego observation
            # get ego's distance with ground truth trajectory (for ade and fde calculation)
            ego_trajectory = self._map.ego_track_dict[ego_id]
            ego_trajectory_location = [ego_trajectory.motion_states[current_time].x, ego_trajectory.motion_states[current_time].y]
            ego_trajectory_velocity = [ego_trajectory.motion_states[current_time].vx, ego_trajectory.motion_states[current_time].vy] 
            trajectory_distance = geometry.get_trajectory_distance(ego_state_dict['loc'], ego_trajectory_location)
            trajectory_location = geometry.get_trajectory_location(ego_state_dict, ego_trajectory_location)
            trajectory_speed = geometry.get_trajectory_speed(ego_trajectory_velocity)

            self.trajectory_distance[ego_id] = [trajectory_distance]
            self.trajectory_location[ego_id] = trajectory_location
            self.trajectory_speed[ego_id] = trajectory_speed
            
            # TODO: localize centerline map points
            
            # TODO: since the location frame change is processed in "replay.py" in dreamer, and the observation is in global frame, 
            # the state can only contain current timestep infomation

            # ego's past trajectory in global frame
            ego_past_state = self._map.past_ego_vehicle_state[ego_id] # x,y,heading in global frame
            vector_past_vehicle_state[ego_id] = geometry.vectorize_point_list(ego_past_state, has_heading=True)
            # ego's past location from global frame to ego frame
            get_location = lambda x: [state[0:2] for state in x]
            get_heading = lambda x: [state[2] for state in x]
            past_location = get_location(self._map.past_ego_vehicle_state[ego_id])
            past_heading = get_heading(self._map.past_ego_vehicle_state[ego_id])
            past_location, past_heading = geometry.localize_transform_list(ego_state_dict['loc'], ego_state_dict['heading'], past_location, past_heading)
            # vecterize location point list plus heading
            vector_location_ego_frame = geometry.vectorize_point_list(past_location, has_heading=False)
            vector_past_vehicle_state_ego_frame[ego_id] = [vector_location_ego_frame[i] + [past_heading[i+1]] for i in range(len(vector_location_ego_frame))]
            
            # print(ego_state_dict['loc'], ego_state_dict['heading'], vector_past_vehicle_state_ego_frame[ego_id][-1][0:2])
            # print(ego_state_dict['loc'], ego_state_dict['heading'])
            

            # ego distance, heading errors and velocity from route center
            target_speed, future_route_points = geometry.get_target_speed_and_future_route_points(ego_state_dict, ego_route_points, ego_route_target_speed)
            lane_observation = geometry.get_lane_observation(ego_state_dict, future_route_points, self._control_steering, self.normalization)

            self.lane_observation[ego_id] = lane_observation
            self.target_speed[ego_id] = [target_speed]
            self.future_route_points[ego_id] = future_route_points
            
            # ego's next locition raletive to current
            self.ego_next_loc[ego_id] = geometry.get_ego_next_loc(ego_state_dict)

            # get record interaction social vehicles' id and observation in classic style
            interaction_vehicles_id, interaction_vehicles_observation, attention_mask = self.get_interaction_vehicles_id_and_observation(ego_state_dict, other_vehicles_state_dict, detective_range=50)
            
            self.interaction_vehicles_id[ego_id] = interaction_vehicles_id
            self.interaction_vehicles_observation[ego_id] = interaction_vehicles_observation
            self.attention_mask[ego_id] = attention_mask

            # Finish judgement 1: reach goal
            goal_point = ego_route_points[-1]
            reach_goal = self.check_reach_goal(ego_state_dict, goal_point)
            self.reach_goal[ego_id] = reach_goal
            
            # Finish judgement 2: collision with other vehicles
            ego_collision = self.check_collision(ego_state_dict, other_vehicles_state_dict, interaction_vehicles_id)
            self.collision[ego_id] = ego_collision

            # Finish judgement 3: deflection from current route/road
            if self._control_steering:
                if self.virtual_route_bound:
                    ego_x_in_route_axis = self.lane_observation[ego_id][0]
                    limitation = 3
                    ego_deflection = self.check_ego_deflection(virtual_route_bound=True, limitation=limitation, distance_to_center=abs(ego_x_in_route_axis), ego_y_in_point_axis=lane_observation[1])
                else: # actual route bound
                    ego_min_bound_distance = min(self.distance_from_bound[ego_id])
                    limitation = 0.25
                    ego_deflection = self.check_ego_deflection(virtual_route_bound=False, limitation=limitation, distance_bound=ego_min_bound_distance)
            else:
                ego_deflection = False
            self.deflection[ego_id] = ego_deflection
        
        surrounding_react_npc_list = []
        # react npc observation (part of ego)
        for react_npc_id, react_npc_state in self._react_npc_vehicle_dict.items():
            # get npc shape, polygon and motion states
            react_npc_state_dict = dict()
            react_npc_state_dict['polygon'] = self._map.react_npc_polygon_dict[react_npc_id]
            react_npc_state_dict['loc'] = [react_npc_state._current_state.x, react_npc_state._current_state.y]
            react_npc_state_dict['speed'] = math.sqrt(react_npc_state._current_state.vx ** 2 + react_npc_state._current_state.vy ** 2)
            react_npc_state_dict['heading'] = react_npc_state._current_state.psi_rad
            # react npc shape
            self.react_npc_shape[react_npc_id] = [react_npc_state._length, react_npc_state._width]

            # TODO: origin point(i.e. ego state['loc'] and ['heading']) should be a global viriable
            distance = math.sqrt((react_npc_state_dict['loc'][0] - ego_state_dict['loc'][0])**2 + (react_npc_state_dict['loc'][1] - ego_state_dict['loc'][1])**2)
            if detection_range:
                y_relative = (react_npc_state_dict['loc'][1] - ego_state_dict['loc'][1])*np.sin(ego_state_dict['heading']) + (react_npc_state_dict['loc'][0] - ego_state_dict['loc'][0])*np.cos(ego_state_dict['heading'])
                # TODO: range of detection should be a hyperparameter
                if distance < 60 and y_relative > -30:
                    # add it to the list
                    add_dict = {'vehicle_id': react_npc_id, 'distance': distance}
                    surrounding_react_npc_list.append(add_dict)
            else:
                add_dict = {'vehicle_id': react_npc_id, 'distance': distance}
                surrounding_react_npc_list.append(add_dict)
                
            # get npc future route points
            react_npc_route_points = geometry.get_route_point_with_heading_from_point_list(self.react_npc_route_dict[react_npc_id], self.min_interval_distance)
            react_npc_route_target_speed = geometry.get_target_speed_from_point_list(self.react_npc_route_dict[react_npc_id])
            react_npc_target_speed, react_npc_future_route_points = geometry.get_target_speed_and_future_route_points(react_npc_state_dict, react_npc_route_points, react_npc_route_target_speed)
            self.future_route_points[react_npc_id] = react_npc_future_route_points

            # check if npc reach goal
            react_npc_goal_point = react_npc_route_points[-1]
            react_npc_reach_goal = self.check_reach_goal(react_npc_state_dict, react_npc_goal_point)
            self.reach_goal[react_npc_id] = react_npc_reach_goal
        
        # record npc state in trajectory style
        surrounding_record_npc_list = []
        for record_npc_id, record_npc_motion_state in self._map.record_npc_motion_state_dict.items():
            record_npc_loc = [record_npc_motion_state.x, record_npc_motion_state.y]
            distance = math.sqrt((record_npc_loc[0] - ego_state_dict['loc'][0])**2 + (record_npc_loc[1] - ego_state_dict['loc'][1])**2)
            # TODO: range of detection should be a hyperparameter
            if detection_range:
                y_relative = (record_npc_loc[1] - ego_state_dict['loc'][1])*np.sin(ego_state_dict['heading']) + (record_npc_loc[0] - ego_state_dict['loc'][0])*np.cos(ego_state_dict['heading'])
                if distance < 60 and y_relative > -30:
                    # add it to the list
                    add_dict = {'vehicle_id': record_npc_id, 'distance': distance}
                    surrounding_record_npc_list.append(add_dict)
            else:
                add_dict = {'vehicle_id': record_npc_id, 'distance': distance}
                surrounding_record_npc_list.append(add_dict)

        # find closet self._npc_num vehicles, get order and state of them, by their distance to ego
        surrounding_npc_list = surrounding_react_npc_list + surrounding_record_npc_list
        surrounding_npc_list = heapq.nsmallest(self._npc_num, surrounding_npc_list, key=lambda s: s['distance'])
        surrounding_id_list = []
        for npc_info in surrounding_npc_list:
            npc_id = npc_info['vehicle_id']
            surrounding_id_list.append(npc_id)
            if npc_id in self._map.past_record_npc_vehicle_state.keys():
                # state in gloabal frame
                npc_past_state = self._map.past_record_npc_vehicle_state[npc_id] # x,y,heading in global frame
                vector_past_vehicle_state[npc_id] = geometry.vectorize_point_list(npc_past_state, has_heading=True)
                # state in ego frame
                past_location, past_heading = get_location(self._map.past_record_npc_vehicle_state[npc_id]), get_heading(self._map.past_record_npc_vehicle_state[npc_id])
                past_location_ego_frame, past_heading_ego_frame = geometry.localize_transform_list(ego_state_dict['loc'], ego_state_dict['heading'], past_location, past_heading)
                vector_location_ego_frame = geometry.vectorize_point_list(past_location_ego_frame, has_heading=False)
                vector_past_vehicle_state_ego_frame[npc_id] = [vector_location_ego_frame[i] + [past_heading_ego_frame[i+1]] for i in range(len(vector_location_ego_frame))]
            elif npc_id in self._map.past_react_npc_vehicle_state.keys():
                # state in gloabal frame
                npc_past_state = self._map.past_react_npc_vehicle_state[npc_id] # x,y,heading in global frame
                vector_past_vehicle_state[npc_id] = geometry.vectorize_point_list(npc_past_state, has_heading=True)
                # state in ego frame
                past_location, past_heading = get_location(self._map.past_react_npc_vehicle_state[npc_id]), get_heading(self._map.past_react_npc_vehicle_state[npc_id])
                past_location_ego_frame, past_heading_ego_frame = geometry.localize_transform_list(ego_state_dict['loc'], ego_state_dict['heading'], past_location, past_heading)
                vector_location_ego_frame = geometry.vectorize_point_list(past_location_ego_frame, has_heading=False)
                vector_past_vehicle_state_ego_frame[npc_id] = [vector_location_ego_frame[i] + [past_heading_ego_frame[i+1]] for i in range(len(vector_location_ego_frame))]
        surrounding_id_dict = {ego_id: surrounding_id_list}
                

        # Finish judgements
        self.observation_dict['reach_goal'] = self.reach_goal
        self.observation_dict['collision'] = self.collision
        self.observation_dict['deflection'] = self.deflection

        # Observations - ego state
        self.observation_dict['ego_shape'] = self.ego_shape              # 2-D
        self.observation_dict['current_speed'] = self.ego_current_speed  # 1-D
        self.observation_dict['ego_next_loc'] = self.ego_next_loc        # 2-D

        # Observations - others state
        self.observation_dict['interaction_vehicles_observation'] = self.interaction_vehicles_observation  # 35-D

        # Observations - vector state
        self.observation_dict['vector_map'] = self.vector_map
        self.observation_dict['vector_past_vehicle_state'] = vector_past_vehicle_state
        self.observation_dict['vector_past_vehicle_state_ego_frame'] = vector_past_vehicle_state_ego_frame
        
        # Observations - route tracking
        self.observation_dict['trajectory_location'] = self.trajectory_location    # 2-D
        self.observation_dict['trajectory_speed'] = self.trajectory_speed          # 1-D
        self.observation_dict['trajectory_distance'] = self.trajectory_distance    # 1-D
        self.observation_dict['target_speed'] = self.target_speed                  # 1-D
        self.observation_dict['distance_from_bound'] = self.distance_from_bound    # 2-D
        self.observation_dict['lane_observation'] = self.lane_observation          # 8-D
        
        # Observations - attention mask
        self.observation_dict['attention_mask'] = self.attention_mask # 6-D

        # Observations - render
        self.observation_dict['surrounding_vehicles_id'] = surrounding_id_dict # self.surrounding_vehicles_id  # use for render
        self.observation_dict['current_bound_points'] = self.ego_closet_bound_points     # use for render
        self.observation_dict['future_route_points'] = self.future_route_points  # use for render
        
        return self.observation_dict
    
    # def save_scene_image(self, ego_id, current_time):
    #     # print(self._map._fig.canvas())
    #     # image_array = np.asarray(self._map._fig.canvas.buffer_rgba())
    #     # print(image_array.shape)
    #     # matplotlib.image.imsave('ego_'+ str(ego_id) + '_' + str(current_time) +'.png', image_array.tolist())
    #     self._map._fig.savefig('ego_'+ str(ego_id) + '_' + str(current_time) +'.png')

    # def get_ego_center_image(self, current_time, ego_state_dict):
    #     image_array = np.asarray(self._map._fig.canvas.buffer_rgba())
    #     # print(image_array.shape)

    #     image_array = image_array.reshape(self._map.fig_height, self._map.fig_width, 4)
    #     for k,v in ego_state_dict.items():
    #         ego_center_pixel_x = int((v.x - self._map.map_x_bound[0]) / self._map.map_width_ratio)
    #         ego_center_pixel_y = int((v.y - self._map.map_y_bound[0]) / self._map.map_height_ratio)
    #         image_min_x = ego_center_pixel_x - self._map.image_half_width
    #         image_max_x = ego_center_pixel_x + self._map.image_half_width
    #         image_min_y = ego_center_pixel_y - self._map.image_half_height
    #         image_max_y = ego_center_pixel_y + self._map.image_half_height
    #         ego_center_image = image_array[self._map.fig_height - image_max_y:self._map.fig_height- image_min_y,image_min_x:image_max_x,:].tolist()
    #         # matplotlib.image.imsave('ego_'+ str(k) + '_' + str(current_time) +'.png', ego_center_image)
    #     matplotlib.image.imsave('ego_'+ str(k) + '_' + str(current_time) +'.png', image_array.tolist())
        # matplotlib.image.imsave('map.png', image_array.tolist())

    # def get_visualization_observation(self, current_time, ego_state_dict):
    #     self.get_ego_center_image(current_time, ego_state_dict)

    # def get_vector_observation(self, current_time):
    #     for ego_id, ego_state in self._ego_vehicles_dict.items():
    #         # get ego shape, polygon and motion states
    #         ego_state_dict = dict()
    #         self.ego_shape[ego_id] = [ego_state._length, ego_state._width]
    #         ego_state_dict['polygon'] = self._map.ego_polygon_dict[ego_id]
    #         ego_state_dict['loc'] = [ego_state._current_state.x, ego_state._current_state.y]
    #         ego_state_dict['speed'] = math.sqrt(ego_state._current_state.vx ** 2 + ego_state._current_state.vy ** 2)
    #         ego_state_dict['heading'] = ego_state._current_state.psi_rad

    #         # get others' vector(2s) (xy_start, xy_end, attributre, id)
    #         other_vehicles_state_dict = dict()
    #         for other_ego_id, other_ego_state in self._ego_vehicles_dict.items():
    #             if other_ego_id == ego_id:
    #                 continue
    #             else:
    #                 other_vehicles_state_dict[other_ego_id] = dict()
    #                 other_vehicles_state_dict[other_ego_id]['polygon'] = self._map.ego_polygon_dict[other_ego_id]
    #                 other_vehicles_state_dict[other_ego_id]['loc'] = [other_ego_state._current_state.x, other_ego_state._current_state.y]
    #                 other_vehicles_state_dict[other_ego_id]['speed'] = math.sqrt(other_ego_state._current_state.vx ** 2 + other_ego_state._current_state.vy ** 2)
    #                 other_vehicles_state_dict[other_ego_id]['heading'] = other_ego_state._current_state.psi_rad
    #         for other_npc_id, other_npc_polygon in self._map.other_vehicle_polygon.items():
    #             other_vehicles_state_dict[other_npc_id] = dict()
    #             other_npc_motion_state = self._map.other_vehicle_motion_state[other_npc_id]
    #             other_vehicles_state_dict[other_npc_id]['loc'] = [other_npc_motion_state.x, other_npc_motion_state.y]
    #             other_vehicles_state_dict[other_npc_id]['speed'] = math.sqrt(other_npc_motion_state.vx ** 2 + other_npc_motion_state.vy ** 2)
    #             other_vehicles_state_dict[other_npc_id]['heading'] = other_npc_motion_state.psi_rad
    #             other_vehicles_state_dict[other_npc_id]['polygon'] = other_npc_polygon
