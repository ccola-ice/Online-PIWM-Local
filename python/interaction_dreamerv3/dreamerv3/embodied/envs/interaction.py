import sys
import pathlib
import gym
import numpy as np
import embodied

# TODO: client_interface should in gym folder?
from interaction_dreamerv3.client_interface import ClientInterface

# isim env
class Interaction(embodied.Env):
  
  def __init__(self, task, args):
    self._task = task
    self._args = args
    self._env = ClientInterface(self._args)

    self._done = True
    print('Set I-SIM env successfully!')

  @property
  def obs_space(self):
    # for prediction decode target, needs to separate different vehicles and predict each of them
    if self._task == 'prediction':
      # ego observation space
      obs_space = {
          'ego': embodied.Space(np.float64, (19, 5)),
          'ego_prediction': embodied.Space(np.float64, (self._args['predict_horizen'], 2)),
          # 'ego_map': embodied.Space(np.float64, shape),
      }
      # plus npc observation space
      for i in range(self._args['npc_num']):
        obs_space[f'npc_{i+1}'] = embodied.Space(np.float64, (19, 5))
        obs_space[f'npc_{i+1}_prediction'] = embodied.Space(np.float64, (self._args['predict_horizen'], 2))
        # obs_space[f'npc_{i+1}_map'] = embodied.Space(np.float64, shape)
      obs_space.update({
                        'id_npc': embodied.Space(np.int32, (self._args['npc_num'])),
                        'mask_npc': embodied.Space(np.int32, (self._args['npc_num'])),
                        'should_init_npc': embodied.Space(np.int32, (self._args['npc_num'])),
                        })
      # plus other(npc with no interaction / is too away from ego) observation space
      for i in range(self._args['other_num']):
        obs_space[f'other_{i+1}'] = embodied.Space(np.float64, (19, 5))
      obs_space.update({
                        'mask_other': embodied.Space(np.int32, (self._args['other_num'])),
                        'should_init_other': embodied.Space(np.int32, (self._args['other_num'])),
                        })
      
    # for PIM with only branch structure network, we need to recon every vehicle's state
    elif self._task == 'branch':
      obs_space = {'ego': embodied.Space(np.float64, (19, 5))}
      # plus npc observation space
      for i in range(self._args['npc_num']):
        obs_space[f'npc_{i+1}'] = embodied.Space(np.float64, (19, 5))
      obs_space.update({
                        'id_npc': embodied.Space(np.int32, (self._args['npc_num'])),
                        'mask_npc': embodied.Space(np.int32, (self._args['npc_num'])),
                        'should_init_npc': embodied.Space(np.int32, (self._args['npc_num'])),
                        })
      # plus other(npc with no interaction / is too away from ego) observation space
      for i in range(self._args['other_num']):
        obs_space[f'other_{i+1}'] = embodied.Space(np.float64, (19, 5))
      obs_space.update({
                        'mask_other': embodied.Space(np.int32, (self._args['other_num'])),
                        'should_init_other': embodied.Space(np.int32, (self._args['other_num'])),
                        })

    # for reconstraction decode target, we treat all vehicles as one whole state
    elif self._task == 'recon':
      # different state sizes
      ego_state_size = (19, 5)
      npc_state_size = (19, 5)
      other_state_size = (19, 5)
      # full state size
      full_state_size = np.prod(ego_state_size) + np.prod(npc_state_size) * self._args['npc_num'] + np.prod(other_state_size) * self._args['other_num'] 
      # ego observation space
      obs_space = {
          'state': embodied.Space(np.float64, (int(full_state_size))),
      }

    # plus episode observation space
    obs_space.update({
                      'reward': embodied.Space(np.float32),
                      'is_first': embodied.Space(bool),
                      'is_last': embodied.Space(bool),
                      'is_terminal': embodied.Space(bool),
    })
    # plus other useful statistics observation space
    obs_space.update({
                      'sta_speed': embodied.Space(np.float32),
                      'sta_collision': embodied.Space(np.int32),
                      'sta_success': embodied.Space(np.int32),
                      'sta_complet': embodied.Space(np.float32),
                      'sta_gt_distance': embodied.Space(np.float32),
    })
    return obs_space

  @property
  def act_space(self):
    return {'action': embodied.Space(np.int32, (), 0, self._env.action_space.n),
            }

  @property
  def state_frame(self):
    return self._args['state_frame']

  def step(self, action):
    # reset the environment
    if self._done:
      self._done = False
      state_dict = self._env.reset()
      # TODO: we only consider one ego vehicle for now
      for ego_id in state_dict.keys():
        state = state_dict[ego_id]

      # prediction decode target, separate vehicle state branch
      if self._task == 'prediction':
        # ego obs
        obs = {
          'ego': state['ego'],
          # 'ego_map': state['ego_map'],
          }
        # npc obs
        for i in range(self._args['npc_num']):
          obs.update({f'npc_{i+1}': state[f'npc_{i+1}']})
          # obs.update({f'npc_{i+1}_map': state[f'npc_{i+1}']})
        obs.update({
                    'id_npc': state['id_npc'],
                    'mask_npc': state['mask_npc'],
                    'should_init_npc': state['should_init_npc'],
                    })
        # other obs
        for i in range(self._args['other_num']):
          obs.update({f'other_{i+1}': state[f'other_{i+1}']})
        obs.update({
                    'mask_other': state['mask_other'],
                    'should_init_other': state['should_init_other'],
                    })
        
      # branch decode target, separate vehicle state branch
      elif self._task == 'branch':
        # ego obs
        obs = {'ego': state['ego']}
        # npc obs
        for i in range(self._args['npc_num']):
          obs.update({f'npc_{i+1}': state[f'npc_{i+1}']})
        obs.update({
                    'id_npc': state['id_npc'],
                    'mask_npc': state['mask_npc'],
                    'should_init_npc': state['should_init_npc'],
                    })
        for i in range(self._args['other_num']):
          obs.update({f'other_{i+1}': state[f'other_{i+1}']})
        obs.update({
                    'mask_other': state['mask_other'],
                    'should_init_other': state['should_init_other'],
                    })

      # recon decode target, concat all vehicles states as one(by their order, from cloest to farest, zero padding)
      elif self._task == 'recon':
        order = state['index_npc'] + state['index_other']
        # for zero-padding npc and other
        for i in range(self._args['npc_num']):
          if f'npc_{i+1}' not in order:
            order.append(f'npc_{i+1}')
        for i in range(self._args['other_num']):
          if f'other_{i+1}' not in order:
            order.append(f'other_{i+1}')
        # concat ego, npcs and others features, npcs and others features are ordered by distance
        value = state['ego'].reshape((-1))
        for key_order in order:
          value = np.concatenate([value, state[key_order].reshape(-1)], axis = 0)
        obs = {'state': value}

      # episode obs
      obs.update({
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
      })
      # statistics obs
      obs.update({
        'sta_speed': 0.0,
        'sta_collision': 0,
        'sta_success': 0,
        'sta_complet': 0,
        'sta_gt_distance': 0,
      })
      return obs
    
    # step the environment
    # TODO: consider multiple ego vehicles
    action_dict = {self._env.ego_id_list[0]: [action['action']]}
    prediction = action['prediction'] if 'prediction' in action.keys() else None
    state_dict, reward_dict, done_dict, aux_info_dict = self._env.step(action_dict, prediction=prediction)
    # TODO: we only consider one ego vehicle for now
    for ego_id in state_dict.keys():
      state = state_dict[ego_id]
      reward = reward_dict[ego_id]
      self._done = done_dict[ego_id]
      aux_info = aux_info_dict[ego_id]

    # prediction decode target, separate vehicle state branch
    if self._task == 'prediction':
      # ego obs
      obs = {'ego': state['ego'],
            # 'ego_map': state['ego_map'],
            }
      # npc obs
      for i in range(self._args['npc_num']):
        obs.update({f'npc_{i+1}': state[f'npc_{i+1}']})
        # obs.update({f'npc_{i+1}_map': state[f'npc_{i+1}']})
      obs.update({
                  'id_npc': state['id_npc'],
                  'mask_npc': state['mask_npc'],
                  'should_init_npc': state['should_init_npc'],
                  })
      # other obs
      for i in range(self._args['other_num']):
        obs.update({f'other_{i+1}': state[f'other_{i+1}']})
      obs.update({
                  'mask_other': state['mask_other'],
                  'should_init_other': state['should_init_other'],
                  })
      
    # branch decode target, separate vehicle state branch
    elif self._task == 'branch':
      # ego obs
      obs = {'ego': state['ego']}
      # npc obs
      for i in range(self._args['npc_num']):
        obs.update({f'npc_{i+1}': state[f'npc_{i+1}']})
      obs.update({
                  'id_npc': state['id_npc'],
                  'mask_npc': state['mask_npc'],
                  'should_init_npc': state['should_init_npc'],
                  })
      for i in range(self._args['other_num']):
        obs.update({f'other_{i+1}': state[f'other_{i+1}']})
      obs.update({
                  'mask_other': state['mask_other'],
                  'should_init_other': state['should_init_other'],
                  })
      
    # recon decode target, concat all vehicles states as one(by their order, from cloest to farest, zero padding)
    elif self._task == 'recon':
      order = state['index_npc'] + state['index_other']
      # for zero-padding npc and other
      for i in range(self._args['npc_num']):
        if f'npc_{i+1}' not in order:
          order.append(f'npc_{i+1}')
      for i in range(self._args['other_num']):
        if f'other_{i+1}' not in order:
          order.append(f'other_{i+1}')
      # concat ego and npcs features, npc feature is ordered by distance
      value = state['ego'].reshape((-1))
      for key_order in order:
        value = np.concatenate([value, state[key_order].reshape(-1)], axis=0)
      obs = {'state': value}
    
    # episode obs
    obs.update({
        'reward': reward,
        'is_first': False,
        'is_last': self._done,
        'is_terminal': self._done,
    })
    # statistics obs
    obs.update({
      'sta_speed': aux_info['speed'],
      'sta_collision': aux_info['result'] == 'collision',
      'sta_success': aux_info['result'] == 'success',
      'sta_complet': aux_info['completion_rate'],
      'sta_gt_distance': aux_info['distance_to_gt'],
    })
    
    return obs
