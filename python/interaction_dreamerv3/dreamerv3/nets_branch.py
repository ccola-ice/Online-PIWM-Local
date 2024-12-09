import re

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

f32 = jnp.float32
tfd = tfp.distributions
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

from . import jaxutils
from . import ninjax as nj
cast = jaxutils.cast_to_compute

''' Only contains some modules that are used in branch abalation only, and other modules for abalation study are in nets_PIM.py '''

class OneBranchRSSM(nj.Module):
  def __init__(
      self, shapes, deter=1024, stoch=32, classes=32, unroll=False, initial='learned',
      unimix=0.01, action_clip=1.0, **kw):
    
    # multi heads of rssm(ego, npc_1...npc_n, other_1....other_n)
    # excluded = ('reward', 'is_first', 'is_last', 'is_terminal')
    # self._keys = [k for k in shapes.keys() if 'prediction' not in k and k not in excluded]
    rssm_included = [k for k in shapes.keys() if k.startswith(('ego', 'npc', 'other')) and 'prediction' not in k]
    self._keys = [k for k in shapes.keys() if k in rssm_included]
    print('RSSM keys:', self._keys)

    # rssm network hyparameters
    self._deter = deter # 'h' in the paper        #隐藏状态hidden state
    self._stoch = stoch # 'z' in the paper        #潜状态latent state
    self._classes = classes # classes of 'z'      #潜状态z_t的类别数
    self._unroll = unroll # TODO: usually False?  #是否展开
    self._initial = initial # 'learned' by config default
    self._unimix = unimix # true out = 99% net out + 1% random
    self._action_clip = action_clip #动作剪枝
    self._kw = kw #关键字参数
  
  def initial(self, batch_size): # bs=batch_size
    state_dict = {}       #状态的字典
    for key in self._keys:
      # initialize latent state(z) based on different branches if use learned weights to do it
      state = self._initial_single_state(key, batch_size)
      state_dict.update({key: state})
    return state_dict
  
  def observe(self, embed, action, is_first, should_init_npc, should_init_other, state_dict=None):
    # ego state and npc state should be divided, also, npc state should be divided into each npc
    if state_dict is None: # size is batch_size (16 by default)
      state_dict = self.initial(batch_size=action.shape[0])

    # swap: change x shape from (16, 64, xx) to (64, 16, xx) (or from (64, 16, xx) to (16, 64, xx))
    # TODO: notice for obs_step, the start is state_dict, not tuple(state_dict, state_dict), figure out why it works like this
    step = lambda prev, inputs: self.obs_step(prev[0], *inputs)
    inputs = self._swap(action), self._swap(embed), self._swap(is_first), self._swap(should_init_npc), self._swap(should_init_other)
    start = state_dict, state_dict
    post_dict, prior_dict = jaxutils.scan(step, inputs, start, self._unroll)

    # swap it back to the original shape (64, 16, xx)
    post_dict = {k: self._swap(v) for k, v in post_dict.items()}
    prior_dict = {k: self._swap(v) for k, v in prior_dict.items()}
    return post_dict, prior_dict

  # TODO: imagine function in rssm class is only used for wm.report?
  def imagine(self, action, state_dict=None):
    state_dict = self.initial(action.shape[0]) if state_dict is None else state_dict
    assert isinstance(state_dict, dict), state_dict
    # swap: change x shape from (16, 64, xx) to (64, 16, xx), or from (64, 16, xx) to (16, 64, xx)
    action = self._swap(action)
    prior_dict = jaxutils.scan(self.img_step, action, state_dict, self._unroll)
    # swap it back to the original shape (64, 16, xx)
    prior_dict = {k: self._swap(v) for k, v in prior_dict.items()}
    return prior_dict

  def obs_step(self, prev_state, prev_action, embed_dict, is_first, should_init_npc, should_init_other):
    # change data type
    is_first = cast(is_first)
    prev_action = cast(prev_action)
    # action clip
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))
    
    # intialize latent state for first frame
    # if is first frame, than state and action is set to 0, else keep it as it is
    prev_state, prev_action = jax.tree_util.tree_map(
        lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
    # for state in first frame is then initialized by the rssm
    prev_state = jax.tree_util.tree_map(
        lambda x, y: x + self._mask(y, is_first), prev_state, self.initial(len(is_first)))
    
    # TODO: initial muilple times when is_first is True, waste of calcualtion
    # initialize latent state for zero-padded(masked) npcs/others or new npcs/others
    should_init_npc, should_init_other = cast(should_init_npc), cast(should_init_other)
    for key in prev_state.keys():
      if key.startswith('npc_'):
        npc_index = int(key.split('_')[-1])
        should_init =  should_init_npc[:, npc_index - 1]
        # for npcs who are new to the observation, set its state to 0 first
        zero_paded_state = jax.tree_util.tree_map(lambda x: self._mask(x, 1.0 - should_init), prev_state[key])
        # and then initialize it using rssm initial function
        prev_state[key] = jax.tree_util.tree_map(lambda x, y: x + self._mask(y, should_init), zero_paded_state, self._initial_single_state(key, len(should_init)))
      elif key.startswith('other_'):
        other_index = int(key.split('_')[-1])
        should_init =  should_init_other[:, other_index - 1]
        zero_paded_state = jax.tree_util.tree_map(lambda x: self._mask(x, 1.0 - should_init), prev_state[key])
        prev_state[key] = jax.tree_util.tree_map(lambda x, y: x + self._mask(y, should_init), zero_paded_state, self._initial_single_state(key, len(should_init)))
    
    # get prior ^z(t) and h(t)
    prior_dict = self.img_step(prev_state, prev_action)
    post_dict = {}
    # different branches
    for key in prior_dict.keys():
      prior = prior_dict[key]
      if key.startswith('ego'):
        # x = h(t) + o(t)
        x = jnp.concatenate([prior['deter'], embed_dict[key]], -1)
        x = self.get('ego_obs_out', Linear, **self._kw)(x)
        # get post z(t)
        stats = self._stats('ego_obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())
        # for post and prior dict, h(deter) is the same
        post = {'stoch': stoch, 'deter': prior['deter'], **stats}
      elif key.startswith('npc_') or key.startswith('other_'):
        x = jnp.concatenate([prior['deter'], embed_dict[key]], -1)
        x = self.get('vehicle_obs_out', Linear, **self._kw)(x)
        stats = self._stats('vehicle_obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())
        post = {'stoch': stoch, 'deter': prior['deter'], **stats}
      post_dict.update({key: post})
    return cast(post_dict), cast(prior_dict)
  
  def img_step(self, prev_state_dict, prev_action):
    # change data type
    prev_action = cast(prev_action)
    # action clip
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))

    # different branch for ego, npc_1...npc_n, other_1...other_n state
    prior_dict = {}
    for key in prev_state_dict.keys():
      if key.startswith(('ego', 'npc_', 'other_')):
        prev_stoch = prev_state_dict[key]['stoch'] # z(t-1)
        deter = prev_state_dict[key]['deter'] # h(t-1)
        # reshape discret stoch(z), make it flatten, e.g. batch x 32 x 32 -> batch x 1024
        if self._classes:
          shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
          prev_stoch = prev_stoch.reshape(shape)
        # for 2D actions.
        if len(prev_action.shape) > len(prev_stoch.shape):  
          shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
          prev_action = prev_action.reshape(shape)

        # different branches for ego, npcs and others
        if key.startswith('ego'):
          # give (z,a)(t-1) and h(t-1) to GRU, get ^z(t) and h(t) and logits(t)
          x = jnp.concatenate([prev_stoch, prev_action], -1)
          x = self.get('ego_img_in', Linear, **self._kw)(x)
          x, deter = self._gru('ego_gru', x, deter)
          x = self.get('ego_img_out', Linear, **self._kw)(x)
          # get logits, which is a mix of uniform and network output
          stats = self._stats('ego_img_stats', x)
          # sample ^z(t) using logit
          dist = self.get_dist(stats)
          stoch = dist.sample(seed=nj.rng())
          # prior
          prior = {'stoch': stoch, 'deter': deter, **stats}
        elif key.startswith('npc_') or key.startswith('other_'):
          x = jnp.concatenate([prev_stoch, prev_action], -1)
          x = self.get('vehicle_img_in', Linear, **self._kw)(x)
          x, deter = self._gru('vehicle_gru', x, deter)
          x = self.get('vehicle_img_out', Linear, **self._kw)(x)
          stats = self._stats('vehicle_img_stats', x)
          dist = self.get_dist(stats)
          stoch = dist.sample(seed=nj.rng())
          prior = {'stoch': stoch, 'deter': deter, **stats}
        # update prior dict
        prior_dict.update({key: prior})
    return cast(prior_dict)

  # gets distribution from network outputs
  def get_dist(self, state, argmax=False):
    if self._classes:
      logit = state['logit'].astype(f32)
      return tfd.Independent(jaxutils.OneHotDist(logit), 1)
    else:
      mean = state['mean'].astype(f32)
      std = state['std'].astype(f32)
      return tfp.MultivariateNormalDiag(mean, std)
  
  # # gets initial distribution from network outputs
  def _get_stoch(self, key, deter, weight='shared'):
    if weight == 'shared':
      x = self.get('init_stoch_layer', Linear, **self._kw)(deter)
      stats = self._stats('init_stoch_stats', x)
      dist = self.get_dist(stats)
    elif weight == 'branch':
      if key.startswith('ego'):
        x = self.get('ego_img_out', Linear, **self._kw)(deter)
        stats = self._stats('ego_img_stats', x)
        dist = self.get_dist(stats)
      elif key.startswith('npc_') or key.startswith('other_'):
        x = self.get('vehicle_img_out', Linear, **self._kw)(deter)
        stats = self._stats('vehicle_img_stats', x)
        dist = self.get_dist(stats)
    return cast(dist.mode())
  
  def _initial_single_state(self, key, batch_size):
    # discrete or continuous latent space
    if self._classes:
      state = dict(
          deter=jnp.zeros([batch_size, self._deter], f32),
          logit=jnp.zeros([batch_size, self._stoch, self._classes], f32),
          stoch=jnp.zeros([batch_size, self._stoch, self._classes], f32))
    else:
      state = dict(
          deter=jnp.zeros([batch_size, self._deter], f32),
          mean=jnp.zeros([batch_size, self._stoch], f32),
          std=jnp.ones([batch_size, self._stoch], f32),
          stoch=jnp.zeros([batch_size, self._stoch], f32))
    # weights initialization
    # NOTE: we found out that initialize latent states with 'zeros' or 'shared' weights 
    # maybe silightly accelerate & stable early learning (for the policy) and enhance its robustness
    # but after all these 3 initial methods can achieve similar performances in the end
    if self._initial == 'zeros': # remain zeros
      state = cast(state)
    elif self._initial == 'learned': # use learned network to initialize
      deter = self.get('initial', jnp.zeros, state['deter'][0].shape, f32)
      state['deter'] = jnp.repeat(jnp.tanh(deter)[None], batch_size, 0)
      state['stoch'] = self._get_stoch(key, cast(state['deter']), weight='branch')
      state = cast(state)
    else:
      raise NotImplementedError(self._initial)
    return state
  
  # gru cell
  def _gru(self, name, x, deter): # inputs is MLP(z+a), state is h
    kw = {**self._kw, 'act': 'none', 'units': 3 * self._deter}
    x = jnp.concatenate([deter, x], -1)
    # GRU contains 3 parts -- r, z, h(t-1)'(=cond now)
    x = self.get(name, Linear, **kw)(x)
    # GRU reset progress
    reset, cand, update = jnp.split(x, 3, -1)
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    # GRU update progress
    deter = update * cand + (1 - update) * deter
    return deter, deter

  def _stats(self, name, x):
    if self._classes:
      x = self.get(name, Linear, self._stoch * self._classes)(x)
      logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
      if self._unimix: # mix of 1% uniform + 99% network output
        probs = jax.nn.softmax(logit, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        logit = jnp.log(probs)
      stats = {'logit': logit}
      return stats
    else:
      x = self.get(name, Linear, 2 * self._stoch)(x)
      mean, std = jnp.split(x, 2, -1)
      std = 2 * jax.nn.sigmoid(std / 2) + 0.1
      return {'mean': mean, 'std': std}

  def _mask(self, value, mask):
    return jnp.einsum('b...,b->b...', value, mask.astype(value.dtype))

  def _swap(self, input):
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    if isinstance(input, dict):
      return {k: swap(v) for k, v in input.items()}
    else:
      return swap(input)
    
  def dyn_loss(self, post_dict, prior_dict, mask_npc=1, mask_other=1, impl='kl', free=1.0):
    # get loss dims
    loss_dims = post_dict['ego']['deter'].shape[:2]
    loss_dict = {}
    # sum up losses for npcs or others branch
    npc_dyn_kl_loss = jnp.zeros(loss_dims)
    other_dyn_kl_loss = jnp.zeros(loss_dims)
    for key in post_dict.keys():
      # how to calculate dyn loss, impl is 'kl' by default
      if impl == 'kl':
        loss = self.get_dist(sg(post_dict[key])).kl_divergence(self.get_dist(prior_dict[key]))
      elif impl == 'logprob':
        loss = -self.get_dist(prior_dict[key]).log_prob(sg(post_dict[key]['stoch']))
      else:
        raise NotImplementedError(impl)
      # kl free
      if free:
        loss = jnp.maximum(loss, free)
      # npc sum up
      if key.startswith('npc_'):
        npc_index = int(key.split('_')[-1])
        mask = mask_npc[:, :, npc_index - 1]
        npc_dyn_kl_loss += loss * mask
        loss_dict.update({'npc_dyn_kl': npc_dyn_kl_loss})
      # other sum up
      elif key.startswith('other_'):
        other_index = int(key.split('_')[-1])
        mask = mask_other[:, :, other_index - 1]
        other_dyn_kl_loss += loss * mask
        loss_dict.update({'other_dyn_kl': other_dyn_kl_loss})
      # ego
      else:
        loss_dict.update({key + '_dyn_kl': loss})
    # print(loss_dict.keys())
    return loss_dict

  def rep_loss(self, post_dict, prior_dict, mask_npc=1, mask_other=1, impl='kl', free=1.0):
    # get loss dims
    loss_dims = post_dict['ego']['deter'].shape[:2]
    loss_dict = {}
    # sum up losses for npcs or others branch
    npc_rep_kl_loss = jnp.zeros(loss_dims)
    other_rep_kl_loss = jnp.zeros(loss_dims)
    for key in post_dict.keys():
      # how to calculate rep loss, impl is 'kl' by default
      if impl == 'kl':
        loss = self.get_dist(post_dict[key]).kl_divergence(self.get_dist(sg(prior_dict[key])))
      elif impl == 'uniform':
        uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior_dict[key])
        loss = self.get_dist(post_dict[key]).kl_divergence(self.get_dist(uniform))
      elif impl == 'entropy':
        loss = -self.get_dist(post_dict[key]).entropy()
      elif impl == 'none':
        loss = jnp.zeros(post_dict[key]['deter'].shape[:-1])
      else:
        raise NotImplementedError(impl)
      # kl free
      if free:
        loss = jnp.maximum(loss, free)
      # npc sum up
      if key.startswith('npc_'):
        npc_index = int(key.split('_')[-1])
        mask = mask_npc[:, :, npc_index - 1]
        npc_rep_kl_loss += loss * mask
        loss_dict.update({'npc_rep_kl': npc_rep_kl_loss})
      # other sum up
      elif key.startswith('other_'):
        other_index = int(key.split('_')[-1])
        mask = mask_other[:, :, other_index - 1]
        other_rep_kl_loss += loss * mask
        loss_dict.update({'other_rep_kl': other_rep_kl_loss})
      # ego
      else:
        loss_dict.update({key + '_rep_kl': loss})
    # print(loss_dict.keys())
    return loss_dict
  

class TwoBranchRSSM(nj.Module):

  def __init__(
      self, shapes, deter=1024, stoch=32, classes=32, unroll=False, initial='learned',
      unimix=0.01, action_clip=1.0, **kw):
    
    # multi heads of rssm(ego, npc_1...npc_n, other_1....other_n)
    # excluded = ('reward', 'is_first', 'is_last', 'is_terminal')
    # self._keys = [k for k in shapes.keys() if 'prediction' not in k and k not in excluded]
    rssm_included = [k for k in shapes.keys() if k.startswith(('ego', 'npc', 'other')) and 'prediction' not in k]
    self._keys = [k for k in shapes.keys() if k in rssm_included]
    print('RSSM keys:', self._keys)

    # rssm network hyparameters
    self._deter = deter # 'h' in the paper
    self._stoch = stoch # 'z' in the paper
    self._classes = classes # classes of 'z'
    self._unroll = unroll # TODO: usually False?
    self._initial = initial # 'learned' by config default
    self._unimix = unimix # true out = 99% net out + 1% random
    self._action_clip = action_clip
    self._kw = kw
  
  def initial(self, batch_size): # bs=batch_size
    state_dict = {}
    for key in self._keys:
      # initialize latent state(z) based on different branches if use learned weights to do it
      state = self._initial_single_state(key, batch_size)
      state_dict.update({key: state})
    return state_dict
  
  def observe(self, embed, action, is_first, should_init_npc, should_init_other, state_dict=None):
    # ego state and npc state should be divided, also, npc state should be divided into each npc
    if state_dict is None: # size is batch_size (16 by default)
      state_dict = self.initial(batch_size=action.shape[0])

    # swap: change x shape from (16, 64, xx) to (64, 16, xx) (or from (64, 16, xx) to (16, 64, xx))
    # TODO: notice for obs_step, the start is state_dict, not tuple(state_dict, state_dict), figure out why it works like this
    step = lambda prev, inputs: self.obs_step(prev[0], *inputs)
    inputs = self._swap(action), self._swap(embed), self._swap(is_first), self._swap(should_init_npc), self._swap(should_init_other)
    start = state_dict, state_dict
    post_dict, prior_dict = jaxutils.scan(step, inputs, start, self._unroll)

    # swap it back to the original shape (64, 16, xx)
    post_dict = {k: self._swap(v) for k, v in post_dict.items()}
    prior_dict = {k: self._swap(v) for k, v in prior_dict.items()}
    return post_dict, prior_dict

  # TODO: imagine function in rssm class is only used for wm.report?
  def imagine(self, action, state_dict=None):
    state_dict = self.initial(action.shape[0]) if state_dict is None else state_dict
    assert isinstance(state_dict, dict), state_dict
    # swap: change x shape from (16, 64, xx) to (64, 16, xx), or from (64, 16, xx) to (16, 64, xx)
    action = self._swap(action)
    prior_dict = jaxutils.scan(self.img_step, action, state_dict, self._unroll)
    # swap it back to the original shape (64, 16, xx)
    prior_dict = {k: self._swap(v) for k, v in prior_dict.items()}
    return prior_dict

  def obs_step(self, prev_state, prev_action, embed_dict, is_first, should_init_npc, should_init_other):
    # change data type
    is_first = cast(is_first)
    prev_action = cast(prev_action)
    # action clip
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))
    
    # intialize latent state for first frame
    # if is first frame, than state and action is set to 0, else keep it as it is
    prev_state, prev_action = jax.tree_util.tree_map(
        lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
    # for state in first frame is then initialized by the rssm
    prev_state = jax.tree_util.tree_map(
        lambda x, y: x + self._mask(y, is_first), prev_state, self.initial(len(is_first)))
    
    # TODO: initial muilple times when is_first is True, waste of calcualtion
    # initialize latent state for zero-padded(masked) npcs/others or new npcs/others
    should_init_npc, should_init_other = cast(should_init_npc), cast(should_init_other)
    for key in prev_state.keys():
      if key.startswith('npc_'):
        npc_index = int(key.split('_')[-1])
        should_init =  should_init_npc[:, npc_index - 1]
        # for npcs who are new to the observation, set its state to 0 first
        zero_paded_state = jax.tree_util.tree_map(lambda x: self._mask(x, 1.0 - should_init), prev_state[key])
        # and then initialize it using rssm initial function
        prev_state[key] = jax.tree_util.tree_map(lambda x, y: x + self._mask(y, should_init), zero_paded_state, self._initial_single_state(key, len(should_init)))
      elif key.startswith('other_'):
        other_index = int(key.split('_')[-1])
        should_init =  should_init_other[:, other_index - 1]
        zero_paded_state = jax.tree_util.tree_map(lambda x: self._mask(x, 1.0 - should_init), prev_state[key])
        prev_state[key] = jax.tree_util.tree_map(lambda x, y: x + self._mask(y, should_init), zero_paded_state, self._initial_single_state(key, len(should_init)))
    
    # get prior ^z(t) and h(t)
    prior_dict = self.img_step(prev_state, prev_action)
    post_dict = {}
    # different branches
    for key in prior_dict.keys():
      prior = prior_dict[key]
      if key.startswith('ego'):
        # x = h(t) + o(t)
        x = jnp.concatenate([prior['deter'], embed_dict[key]], -1)
        x = self.get('ego_obs_out', Linear, **self._kw)(x)
        # get post z(t)
        stats = self._stats('ego_obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())
        # for post and prior dict, h(deter) is the same
        post = {'stoch': stoch, 'deter': prior['deter'], **stats}
      elif key.startswith('npc_'):
        x = jnp.concatenate([prior['deter'], embed_dict[key]], -1)
        x = self.get('npc_obs_out', Linear, **self._kw)(x)
        stats = self._stats('npc_obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())
        post = {'stoch': stoch, 'deter': prior['deter'], **stats}
      elif key.startswith('other_'):
        x = jnp.concatenate([prior['deter'], embed_dict[key]], -1)
        x = self.get('other_obs_out', Linear, **self._kw)(x)
        stats = self._stats('other_obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())
        post = {'stoch': stoch, 'deter': prior['deter'], **stats}
      post_dict.update({key: post})
    return cast(post_dict), cast(prior_dict)
  
  def img_step(self, prev_state_dict, prev_action):
    # change data type
    prev_action = cast(prev_action)
    # action clip
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))

    # different branch for ego, npc_1...npc_n, other_1...other_n state
    prior_dict = {}
    for key in prev_state_dict.keys():
      if key.startswith(('ego', 'npc_', 'other_')):
        prev_stoch = prev_state_dict[key]['stoch'] # z(t-1)
        deter = prev_state_dict[key]['deter'] # h(t-1)
        # reshape discret stoch(z), make it flatten, e.g. batch x 32 x 32 -> batch x 1024
        if self._classes:
          shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
          prev_stoch = prev_stoch.reshape(shape)
        # for 2D actions.
        if len(prev_action.shape) > len(prev_stoch.shape):  
          shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
          prev_action = prev_action.reshape(shape)

        # different branches for ego, npcs and others
        if key.startswith('ego'):
          # give (z,a)(t-1) and h(t-1) to GRU, get ^z(t) and h(t) and logits(t)
          x = jnp.concatenate([prev_stoch, prev_action], -1)
          x = self.get('ego_img_in', Linear, **self._kw)(x)
          x, deter = self._gru('ego_gru', x, deter)
          x = self.get('ego_img_out', Linear, **self._kw)(x)
          # get logits, which is a mix of uniform and network output
          stats = self._stats('ego_img_stats', x)
          # sample ^z(t) using logit
          dist = self.get_dist(stats)
          stoch = dist.sample(seed=nj.rng())
          # prior
          prior = {'stoch': stoch, 'deter': deter, **stats}
        elif key.startswith('npc_'):
          x = jnp.concatenate([prev_stoch, prev_action], -1)
          x = self.get('npc_img_in', Linear, **self._kw)(x)
          x, deter = self._gru('npc_gru', x, deter)
          x = self.get('npc_img_out', Linear, **self._kw)(x)
          stats = self._stats('npc_img_stats', x)
          dist = self.get_dist(stats)
          stoch = dist.sample(seed=nj.rng())
          prior = {'stoch': stoch, 'deter': deter, **stats}
        elif key.startswith('other_'):
          x = jnp.concatenate([prev_stoch, prev_action], -1)
          x = self.get('other_img_in', Linear, **self._kw)(x)
          x, deter = self._gru('other_gru', x, deter)
          x = self.get('other_img_out', Linear, **self._kw)(x)
          stats = self._stats('other_img_stats', x)
          dist = self.get_dist(stats)
          stoch = dist.sample(seed=nj.rng())
          prior = {'stoch': stoch, 'deter': deter, **stats}
        # update prior dict
        prior_dict.update({key: prior})
    return cast(prior_dict)

  # gets distribution from network outputs
  def get_dist(self, state, argmax=False):
    if self._classes:
      logit = state['logit'].astype(f32)
      return tfd.Independent(jaxutils.OneHotDist(logit), 1)
    else:
      mean = state['mean'].astype(f32)
      std = state['std'].astype(f32)
      return tfp.MultivariateNormalDiag(mean, std)
  
  # # gets initial distribution from network outputs
  def _get_stoch(self, key, deter, weight='shared'):
    if weight == 'shared':
      x = self.get('init_stoch_layer', Linear, **self._kw)(deter)
      stats = self._stats('init_stoch_stats', x)
      dist = self.get_dist(stats)
    elif weight == 'branch':
      if key.startswith('ego'):
        x = self.get('ego_img_out', Linear, **self._kw)(deter)
        stats = self._stats('ego_img_stats', x)
        dist = self.get_dist(stats)
      elif key.startswith('npc_'):
        x = self.get('npc_img_out', Linear, **self._kw)(deter)
        stats = self._stats('npc_img_stats', x)
        dist = self.get_dist(stats)
      elif key.startswith('other_'):
        x = self.get('other_img_out', Linear, **self._kw)(deter)
        stats = self._stats('other_img_stats', x)
        dist = self.get_dist(stats)
    return cast(dist.mode())
  
  def _initial_single_state(self, key, batch_size):
    # discrete or continuous latent space
    if self._classes:
      state = dict(
          deter=jnp.zeros([batch_size, self._deter], f32),
          logit=jnp.zeros([batch_size, self._stoch, self._classes], f32),
          stoch=jnp.zeros([batch_size, self._stoch, self._classes], f32))
    else:
      state = dict(
          deter=jnp.zeros([batch_size, self._deter], f32),
          mean=jnp.zeros([batch_size, self._stoch], f32),
          std=jnp.ones([batch_size, self._stoch], f32),
          stoch=jnp.zeros([batch_size, self._stoch], f32))
    # weights initialization
    # NOTE: we found out that initialize latent states with 'zeros' or 'shared' weights 
    # maybe silightly accelerate & stable early learning (for the policy) and enhance its robustness
    # but after all these 3 initial methods can achieve similar performances in the end
    if self._initial == 'zeros': # remain zeros
      state = cast(state)
    elif self._initial == 'learned': # use learned network to initialize
      deter = self.get('initial', jnp.zeros, state['deter'][0].shape, f32)
      state['deter'] = jnp.repeat(jnp.tanh(deter)[None], batch_size, 0)
      state['stoch'] = self._get_stoch(key, cast(state['deter']), weight='branch')
      state = cast(state)
    else:
      raise NotImplementedError(self._initial)
    return state
  
  # gru cell
  def _gru(self, name, x, deter): # inputs is MLP(z+a), state is h
    kw = {**self._kw, 'act': 'none', 'units': 3 * self._deter}
    x = jnp.concatenate([deter, x], -1)
    # GRU contains 3 parts -- r, z, h(t-1)'(=cond now)
    x = self.get(name, Linear, **kw)(x)
    # GRU reset progress
    reset, cand, update = jnp.split(x, 3, -1)
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    # GRU update progress
    deter = update * cand + (1 - update) * deter
    return deter, deter

  def _stats(self, name, x):
    if self._classes:
      x = self.get(name, Linear, self._stoch * self._classes)(x)
      logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
      if self._unimix: # mix of 1% uniform + 99% network output
        probs = jax.nn.softmax(logit, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        logit = jnp.log(probs)
      stats = {'logit': logit}
      return stats
    else:
      x = self.get(name, Linear, 2 * self._stoch)(x)
      mean, std = jnp.split(x, 2, -1)
      std = 2 * jax.nn.sigmoid(std / 2) + 0.1
      return {'mean': mean, 'std': std}

  def _mask(self, value, mask):
    return jnp.einsum('b...,b->b...', value, mask.astype(value.dtype))

  def _swap(self, input):
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    if isinstance(input, dict):
      return {k: swap(v) for k, v in input.items()}
    else:
      return swap(input)
    
  def dyn_loss(self, post_dict, prior_dict, mask_npc=1, mask_other=1, impl='kl', free=1.0):
    # get loss dims
    loss_dims = post_dict['ego']['deter'].shape[:2]
    loss_dict = {}
    # sum up losses for npcs or others branch
    npc_dyn_kl_loss = jnp.zeros(loss_dims)
    other_dyn_kl_loss = jnp.zeros(loss_dims)
    for key in post_dict.keys():
      # how to calculate dyn loss, impl is 'kl' by default
      if impl == 'kl':
        loss = self.get_dist(sg(post_dict[key])).kl_divergence(self.get_dist(prior_dict[key]))
      elif impl == 'logprob':
        loss = -self.get_dist(prior_dict[key]).log_prob(sg(post_dict[key]['stoch']))
      else:
        raise NotImplementedError(impl)
      # kl free
      if free:
        loss = jnp.maximum(loss, free)
      # npc sum up
      if key.startswith('npc_'):
        npc_index = int(key.split('_')[-1])
        mask = mask_npc[:, :, npc_index - 1]
        npc_dyn_kl_loss += loss * mask
        loss_dict.update({'npc_dyn_kl': npc_dyn_kl_loss})
      # other sum up
      elif key.startswith('other_'):
        other_index = int(key.split('_')[-1])
        mask = mask_other[:, :, other_index - 1]
        other_dyn_kl_loss += loss * mask
        loss_dict.update({'other_dyn_kl': other_dyn_kl_loss})
      # ego
      else:
        loss_dict.update({key + '_dyn_kl': loss})
    # print(loss_dict.keys())
    return loss_dict

  def rep_loss(self, post_dict, prior_dict, mask_npc=1, mask_other=1, impl='kl', free=1.0):
    # get loss dims
    loss_dims = post_dict['ego']['deter'].shape[:2]
    loss_dict = {}
    # sum up losses for npcs or others branch
    npc_rep_kl_loss = jnp.zeros(loss_dims)
    other_rep_kl_loss = jnp.zeros(loss_dims)
    for key in post_dict.keys():
      # how to calculate rep loss, impl is 'kl' by default
      if impl == 'kl':
        loss = self.get_dist(post_dict[key]).kl_divergence(self.get_dist(sg(prior_dict[key])))
      elif impl == 'uniform':
        uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior_dict[key])
        loss = self.get_dist(post_dict[key]).kl_divergence(self.get_dist(uniform))
      elif impl == 'entropy':
        loss = -self.get_dist(post_dict[key]).entropy()
      elif impl == 'none':
        loss = jnp.zeros(post_dict[key]['deter'].shape[:-1])
      else:
        raise NotImplementedError(impl)
      # kl free
      if free:
        loss = jnp.maximum(loss, free)
      # npc sum up
      if key.startswith('npc_'):
        npc_index = int(key.split('_')[-1])
        mask = mask_npc[:, :, npc_index - 1]
        npc_rep_kl_loss += loss * mask
        loss_dict.update({'npc_rep_kl': npc_rep_kl_loss})
      # other sum up
      elif key.startswith('other_'):
        other_index = int(key.split('_')[-1])
        mask = mask_other[:, :, other_index - 1]
        other_rep_kl_loss += loss * mask
        loss_dict.update({'other_rep_kl': other_rep_kl_loss})
      # ego
      else:
        loss_dict.update({key + '_rep_kl': loss})
    # print(loss_dict.keys())
    return loss_dict
  

class OneBranchEncoder(nj.Module):

  # 根据输入数据的形状和类型，初始化编码器模块CNN或MLP
  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', mlp_layers=4,
      mlp_units=512, cnn='resize', cnn_depth=48,
      cnn_blocks=2, resize='stride',
      symlog_inputs=False, minres=4, **kw):

    excluded = ('is_first', 'is_last')
    shapes = {k: v for k, v in shapes.items() if (k not in excluded and not k.startswith('log_') and not k.endswith('_prediction'))}
    self.cnn_shapes = {k: v for k, v in shapes.items() if (
        len(v) == 3 and re.match(cnn_keys, k))}
    self.mlp_shapes = {k: v for k, v in shapes.items() if (
        len(v) in (1, 2) and re.match(mlp_keys, k))}
    self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
    print('Encoder CNN shapes:', self.cnn_shapes)
    print('Encoder MLP shapes:', self.mlp_shapes)

    # cnn layers
    cnn_kw = {**kw, 'minres': minres, 'name': 'cnn'}
    if self.cnn_shapes: # 如果有CNN输入
      if cnn == 'resnet':
        self._cnn = ImageEncoderResnet(cnn_depth, cnn_blocks, resize, **cnn_kw)
      else:
        raise NotImplementedError(cnn)
    
    # mlp layers, 2 shared layers for trajectory and 2 layers for each branch
    if self.mlp_shapes: # 如果有MLP输入
      # vehicle info
      enc_mlp_layer = int(mlp_layers / 2)

      # encode trajectory using the same mlp
      # 初始化共享轨迹编码器，提取轨迹特征，对应论文中的shared trajectory encoder
      self._traj_mlp = MLP(None, enc_mlp_layer, mlp_units, dist='none', **kw, symlog_inputs=symlog_inputs, name='traj_mlp')
      
      # for ego, npc and other features, using different mlp
      # 
      self._ego_mlp = MLP(None, enc_mlp_layer, mlp_units, dist='none', **kw, symlog_inputs=False, name='ego_mlp')
      
      #
      self._vehicle_mlp = MLP(None, enc_mlp_layer, mlp_units, dist='none', **kw, symlog_inputs=False, name='vehicle_mlp')
      # TODO: map info
      # self._map_mlp = MLP(None, enc_mlp_layer, mlp_units, dist='none', **kw, symlog_inputs=symlog_inputs, name='map_mlp')

  # 前向传播
  def __call__(self, data):
    
    # 1.数据重塑
    # to get batch dims, and reshape the data 
    some_key, some_shape = list(self.shapes.items())[0]
    batch_dims = data[some_key].shape[:-len(some_shape)]
    data = {
        k: v.reshape((-1,) + v.shape[len(batch_dims):])
        for k, v in data.items()}

    outputs_dict = {}

    # 2.处理CNN输入(图像)
    # for image inputs
    if self.cnn_shapes:
      inputs = jnp.concatenate([data[k] for k in self.cnn_shapes], -1) # 将CNN数据在最后一个维度拼接，合并成一个输入
      output = self._cnn(inputs) # 处理后的数据输入到CNN  (self._cnn = ImageEncoderResnet(cnn_depth, cnn_blocks, resize, **cnn_kw))
      output = output.reshape((output.shape[0], -1)) # 将CNN输出展开成二维向量
      outputs_dict.update({'cnn': output}) # 更新输出字典

    # 3.处理MLP输入(向量/地图)
    # for vector inputs (vehicle and map)
    if self.mlp_shapes:
      for key in self.mlp_shapes:
        ob = jaxutils.cast_to_compute(data[key].astype(f32))
        if key.startswith(('ego', 'npc_', 'other_')):
          traj_features = self._traj_mlp(ob)
          traj_features = traj_features.reshape((traj_features.shape[0], -1))
          if key.startswith('ego'):
            features = self._ego_mlp(traj_features)
          elif key.startswith('npc_') or key.startswith('other_'):
            features = self._vehicle_mlp(traj_features)
        else:
          # features = self._map_mlp(ob)
          pass
        outputs_dict.update({key: features.reshape(batch_dims + features.shape[1:])})
    
    return outputs_dict
  

def multi_head_attention(q, k, v, mask, drop_out=0.1):
  d_k = q.shape[-1]
  att_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(d_k)
  att_logits = jnp.where(mask, att_logits, -1e9)
  attention = jax.nn.softmax(att_logits, axis=-1)
  # TODO: add drop_out layers
  if drop_out:
    # attention = dropout(attention)
    pass
  out = jnp.matmul(attention, v)

  return out, attention

class BranchEgoAttention(nj.Module):
  # it considers all vehicles around ego (npc + other) compared with EgoAttention class
  def __init__(
    self, shape, layers, heads, units_per_head, inputs=['tensor'], dims=None, 
    symlog_inputs=False, **kw):

    # data shape type transition
    assert shape is None or isinstance(shape, (int, tuple, dict)), shape
    if isinstance(shape, int):
      shape = (shape,)
    # network hyparameters
    self._shape = shape
    self._layers = layers
    self._heads = heads
    self._units_per_head = units_per_head
    self._inputs = Input(inputs, dims=dims)
    self._symlog_inputs = symlog_inputs
    # key words for dense layers and output(distribution) layers
    distkeys = (
        'dist', 'outscale', 'minstd', 'maxstd', 'outnorm', 'unimix', 'bins')
    self._dense = {k: v for k, v in kw.items() if k not in distkeys}
    self._dist  = {k: v for k, v in kw.items() if k in distkeys}

  def __call__(self, inputs_dict):
    # preprocess inputs
    feature_dict = {key: self._inputs(value) for key, value in inputs_dict.items() if isinstance(value, dict)}
    if self._symlog_inputs:
      feature_dict = {key: jaxutils.symlog(value) for key, value in feature_dict.items()}
    # use attention mechanism to fuse ego and npc features together
    npc_mask = inputs_dict['mask_npc']
    npc_num = npc_mask.shape[-1]
    other_mask = inputs_dict['mask_other']
    other_num = other_mask.shape[-1]
    # concat ego and npc features together in entity dimension
    feature_dict = {key: jnp.expand_dims(value, axis=-2) for key, value in feature_dict.items()}
    ego_features = feature_dict['ego']
    ego_features = jaxutils.cast_to_compute(ego_features)
    vehicle_features = jnp.concatenate([feature_dict['ego']] + [feature_dict[f'npc_{i+1}'] for i in range(npc_num)] + [feature_dict[f'other_{i+1}'] for i in range(other_num)], axis=-2)
    # vehicle_features = jnp.concatenate([feature_dict['ego']] + [feature_dict[f'npc_{i+1}'] for i in range(npc_num)], axis=-2)
    vehicle_features = jaxutils.cast_to_compute(vehicle_features)
    # attention
    # Dimensions: Batch*Length, entity, head, feature_per_head
    q_ego = self.get('query', Linear, units=self._heads*self._units_per_head, **self._dense)(ego_features).reshape([-1, 1, self._heads, self._units_per_head])
    k_all = self.get('key', Linear, units=self._heads*self._units_per_head, **self._dense)(vehicle_features).reshape([-1, 1 + npc_num + other_num, self._heads, self._units_per_head])
    v_all = self.get('value', Linear, units=self._heads*self._units_per_head, **self._dense)(vehicle_features).reshape([-1, 1 + npc_num + other_num, self._heads, self._units_per_head])
    # k_all = self.get('key', Linear, units=self._heads*self._units_per_head, **self._dense)(vehicle_features).reshape([-1, 1 + npc_num, self._heads, self._units_per_head])
    # v_all = self.get('value', Linear, units=self._heads*self._units_per_head, **self._dense)(vehicle_features).reshape([-1, 1 + npc_num, self._heads, self._units_per_head])
    # Dimensions: Batch*Length, head, entity, feature_per_head
    q_ego = q_ego.transpose(0,2,1,3)
    k_all = k_all.transpose(0,2,1,3)
    v_all = v_all.transpose(0,2,1,3)
    print('q :', q_ego)
    print('k :', k_all)
    print('v :', v_all)
    print('q shape:', q_ego.shape)
    print('k shape:', k_all.shape)
    print('v shape:', v_all.shape)
    # mask Dimensions: Batch*Length, head, 1, entity
    ego_mask = jnp.ones(list(q_ego.shape[:1]) + [1,1]) # Batch*Length, 1, 1
    npc_mask = npc_mask.reshape(-1, 1, npc_num)
    other_mask = other_mask.reshape(-1, 1, other_num)
    mask = jnp.concatenate([ego_mask, npc_mask, other_mask], axis=-1).reshape([-1, 1, 1, npc_num + other_num + 1])
    # mask = jnp.concatenate([ego_mask, npc_mask], axis=-1).reshape([-1, 1, 1, npc_num + 1])
    mask = jnp.repeat(mask, self._heads, axis=1)
    # print('mask shape', mask.shape)
    # TODO: the attention of ego and npc can be get through 'different attention layers', 
    # since they do different tasks in latter parts(npc for future prediction and ego for actor/critic/reward/count)
    # yet we use different mlp head to get a different attention result for ego and npc for now
    ego_attention_out, ego_attention_mat = multi_head_attention(q_ego, k_all, v_all, mask, drop_out=False)
    # Dimensions(back to): Batch*Length, entity, head, feature_per_head
    ego_attention_out = ego_attention_out.transpose(0,2,1,3)
    ego_attention_mat = ego_attention_mat.transpose(0,2,1,3)

    # attention matrix for ego
    mat = ego_attention_mat[..., 0, :, :]
    # attention output for ego
    x = ego_attention_out[..., 0, :, :]
    x = x.reshape([x.shape[0], -1])
    x = self.get('ego_out_mlp', Linear, units=self._heads*self._units_per_head, **self._dense)(x)
    out = x.reshape(list(vehicle_features.shape[:-2]) + [-1]) # Batch, Length, out_feature

    return out, mat
  

class OneBranchDecoder(nj.Module):

  def __init__(
      self, shapes, inputs=['tensor'], cnn_keys=r'.*', mlp_keys=r'.*',
      mlp_layers=4, mlp_units=512, cnn='resize', cnn_depth=48, cnn_blocks=2,
      image_dist='mse', vector_dist='mse', resize='stride', bins=255,
      outscale=1.0, minres=4, cnn_sigmoid=False, **kw):
    
    # pick decode targets and their shapes
    excluded = ('is_first', 'is_last', 'is_terminal', 'reward')
    shapes = {k: v for k, v in shapes.items() if k not in excluded}
    self.cnn_shapes = {
        k: v for k, v in shapes.items()
        if re.match(cnn_keys, k) and len(v) == 3}
    
    self.mlp_shapes = {
        k: v for k, v in shapes.items()
        if re.match(mlp_keys, k) and len(v) in (1, 2) and not k.endswith('_prediction')}
    
    self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
    print('Decoder CNN shapes:', self.cnn_shapes)
    print('Decoder MLP shapes:', self.mlp_shapes)

    # inputs preprocess
    self._inputs = Input(inputs, dims='deter') # inputs: ['deter', 'stoch'], dims:'deter'
    # different kinds of decode networks
    # decode image
    cnn_kw = {**kw, 'minres': minres, 'sigmoid': cnn_sigmoid}
    if self.cnn_shapes:
      shapes = list(self.cnn_shapes.values())
      assert all(x[:-1] == shapes[0][:-1] for x in shapes)
      shape = shapes[0][:-1] + (sum(x[-1] for x in shapes),)
      if cnn == 'resnet':
        self._cnn = ImageDecoderResnet(
            shape, cnn_depth, cnn_blocks, resize, **cnn_kw, name='cnn')
      else:
        raise NotImplementedError(cnn)
    self._image_dist = image_dist

    # decode vector
    mlp_kw = {**kw, 'dist': vector_dist, 'outscale': outscale, 'bins': bins}
    if self.mlp_shapes:
      # vehicle history trajectory reconstruction
      self._ego_mlp = MLP(self.mlp_shapes['ego'], mlp_layers, mlp_units, **mlp_kw, name='ego_mlp')
      self._vehicle_mlp = MLP(self.mlp_shapes['npc_1'], mlp_layers, mlp_units, **mlp_kw, name='vehicle_mlp')

  def __call__(self, inputs_dict, drop_loss_indices=None):
    dists_dict = {}
    # decode image
    if self.cnn_shapes:
      featrue_dict = self._inputs(inputs_dict)
      feat = featrue_dict
      if drop_loss_indices is not None:
        feat = feat[:, drop_loss_indices]
      flat = feat.reshape([-1, feat.shape[-1]])
      output = self._cnn(flat)
      output = output.reshape(feat.shape[:-1] + output.shape[1:])
      split_indices = np.cumsum([v[-1] for v in self.cnn_shapes.values()][:-1])
      means = jnp.split(output, split_indices, -1)
      dists_dict.update({
          key: self._make_image_dist(key, mean)
          for (key, shape), mean in zip(self.cnn_shapes.items(), means)})
    # decode vector
    if self.mlp_shapes:
      for key in self.mlp_shapes:
        # consider key may contains 'prediction' according to the training target
        featrue = self._inputs(inputs_dict[key.replace('_prediction', '')])
        if key.startswith('ego'):
          dist = self._ego_mlp(featrue)
        elif key.startswith('npc_') or key.startswith('other_'):
          dist = self._vehicle_mlp(featrue)
        dists_dict.update({key: dist})
    return dists_dict

  def _make_image_dist(self, name, mean):
    mean = mean.astype(f32)
    if self._image_dist == 'normal':
      return tfd.Independent(tfd.Normal(mean, 1), 3)
    if self._image_dist == 'mse':
      return jaxutils.MSEDist(mean, 3, 'sum')
    raise NotImplementedError(self._image_dist)


class TwoBranchDecoder(nj.Module):

  def __init__(
      self, shapes, inputs=['tensor'], cnn_keys=r'.*', mlp_keys=r'.*',
      mlp_layers=4, mlp_units=512, cnn='resize', cnn_depth=48, cnn_blocks=2,
      image_dist='mse', vector_dist='mse', resize='stride', bins=255,
      outscale=1.0, minres=4, cnn_sigmoid=False, **kw):
    
    # pick decode targets and their shapes
    excluded = ('is_first', 'is_last', 'is_terminal', 'reward')
    shapes = {k: v for k, v in shapes.items() if k not in excluded}
    self.cnn_shapes = {
        k: v for k, v in shapes.items()
        if re.match(cnn_keys, k) and len(v) == 3}
    
    self.mlp_shapes = {
        k: v for k, v in shapes.items()
        if re.match(mlp_keys, k) and len(v) in (1, 2) and not k.endswith('_prediction')}
    
    self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
    print('Decoder CNN shapes:', self.cnn_shapes)
    print('Decoder MLP shapes:', self.mlp_shapes)

    # inputs preprocess
    self._inputs = Input(inputs, dims='deter') # inputs: ['deter', 'stoch'], dims:'deter'
    # different kinds of decode networks
    # decode image
    cnn_kw = {**kw, 'minres': minres, 'sigmoid': cnn_sigmoid}
    if self.cnn_shapes:
      shapes = list(self.cnn_shapes.values())
      assert all(x[:-1] == shapes[0][:-1] for x in shapes)
      shape = shapes[0][:-1] + (sum(x[-1] for x in shapes),)
      if cnn == 'resnet':
        self._cnn = ImageDecoderResnet(
            shape, cnn_depth, cnn_blocks, resize, **cnn_kw, name='cnn')
      else:
        raise NotImplementedError(cnn)
    self._image_dist = image_dist

    # decode vector
    mlp_kw = {**kw, 'dist': vector_dist, 'outscale': outscale, 'bins': bins}
    if self.mlp_shapes:
      # vehicle history trajectory reconstruction
      self._ego_mlp = MLP(self.mlp_shapes['ego'], mlp_layers, mlp_units, **mlp_kw, name='ego_mlp')
      self._npc_mlp = MLP(self.mlp_shapes['npc_1'], mlp_layers, mlp_units, **mlp_kw, name='npc_mlp')
      self._other_mlp = MLP(self.mlp_shapes['other_1'], mlp_layers, mlp_units, **mlp_kw, name='other_mlp')

  def __call__(self, inputs_dict, drop_loss_indices=None):
    dists_dict = {}
    # decode image
    if self.cnn_shapes:
      featrue_dict = self._inputs(inputs_dict)
      feat = featrue_dict
      if drop_loss_indices is not None:
        feat = feat[:, drop_loss_indices]
      flat = feat.reshape([-1, feat.shape[-1]])
      output = self._cnn(flat)
      output = output.reshape(feat.shape[:-1] + output.shape[1:])
      split_indices = np.cumsum([v[-1] for v in self.cnn_shapes.values()][:-1])
      means = jnp.split(output, split_indices, -1)
      dists_dict.update({
          key: self._make_image_dist(key, mean)
          for (key, shape), mean in zip(self.cnn_shapes.items(), means)})
    # decode vector
    if self.mlp_shapes:
      for key in self.mlp_shapes:
        # consider key may contains 'prediction' according to the training target
        featrue = self._inputs(inputs_dict[key.replace('_prediction', '')])
        if key.startswith('ego'):
          dist = self._ego_mlp(featrue)
        elif key.startswith('npc_'):
          dist = self._npc_mlp(featrue)
        elif key.startswith('other_'):
          dist = self._other_mlp(featrue)
        dists_dict.update({key: dist})
    return dists_dict

  def _make_image_dist(self, name, mean):
    mean = mean.astype(f32)
    if self._image_dist == 'normal':
      return tfd.Independent(tfd.Normal(mean, 1), 3)
    if self._image_dist == 'mse':
      return jaxutils.MSEDist(mean, 3, 'sum')
    raise NotImplementedError(self._image_dist)


class MLP(nj.Module):

  def __init__(
      self, shape, layers, units, inputs=['tensor'], dims=None,
      symlog_inputs=False, **kw):
    # **kw 其他关键字参数

    # data shape type transition
    assert shape is None or isinstance(shape, (int, tuple, dict)), shape
    if isinstance(shape, int):
      shape = (shape,)  # 将整数类型shape转换为元组

    # network hyparameters
    self._shape = shape   # 输出数据的形状，可以是整数、元组或字典
    self._layers = layers # MLP的层数
    self._units = units   # 每层的神经元数量
    self._inputs = Input(inputs, dims=dims) # 输入类型，默认为['tensor']
    self._symlog_inputs = symlog_inputs # 是否对输入进行对称对数变换 False

    # key words for dense layers and output(distribution) layers
    distkeys = (
        'dist', 'outscale', 'minstd', 'maxstd', 'outnorm', 'unimix', 'bins')
    # 全连接层参数：
    self._dense = {k: v for k, v in kw.items() if k not in distkeys}  #将distkeys之外的关键字参数保存到self._dense中,作为全连接层参数
    # 输出层参数：
    self._dist = {k: v for k, v in kw.items() if k in distkeys} #将distkeys之内的关键字参数保存到self._dist中,作为输出层参数

  # 多层感知机的前向传播
  def __call__(self, inputs):
    
    # 1.输入预处理
    # preprocess inputs    
    feat = self._inputs(inputs)
    if self._symlog_inputs:
      feat = jaxutils.symlog(feat)
    x = jaxutils.cast_to_compute(feat) # 将输入数据转换为浮点数

    # 2.多层感知机的前向传播,特征提取
    # make it flatten
    x = x.reshape([-1, x.shape[-1]]) # 展平输入的形状：x.shape[-1]表示保持最后一个维度的值不变，-1表示自动计算其他维度的数值。比如：原始输入是(32,10,256)，那么x = x.reshape([-1,256])的输出是(320,256)
    for i in range(self._layers): # 遍历MLP每一层
      x = self.get(f'h{i}', Linear, self._units, **self._dense)(x)
      # f'h{i}' 是该层的名称，‘h{0},h{1}’
      # Linear  是该层的类型，(线性层)
      # self._units是该层的输出维度(神经元数量)
      # **self._dense：解引用全连接层的参数，比如激活函数
      # self.get()函数首先判断名为'h{i}'的线性层是否已经存在，如果不存在则创建一个新的Linear层，参数为self._units和self._dense。
      # 其返回值为一个配置好的线性层对象，这个对象包含:权重矩阵、偏置、激活函数等参数。
      # 然后用这样一个线性层对象处理x的输入，进行前向计算得到输出x。
    x = x.reshape(feat.shape[:-1] + (x.shape[-1],)) # 将多层n线性变换的结果重塑为原始形状
    # feat.shape[:-1] 表示保持原始输入feat的所有维度除了最后一个维度，
    # x.shape[-1] 表示经过MLP处理后的特征维度(神经元数量)，x为输入经过所有线性层的输出，而输出的最后一个维度是self._units。 (线性层输出的最后一个维度是self._units，所以x.shape[-1] = self._units）
    # 最终这句话使得e多层线性变换后的形状没变，只有最后一个维度的值改变了。

    # 3.多层感知机的前向传播，输出层
    # different kinds of outputs according to self._shape's type
    if self._shape is None:
      return x # 直接返回特征
    elif isinstance(self._shape, tuple):
      return self._out('out', self._shape, x) # 根据self._dist参数生成单个概率分布输出 #调用_out()函数，‘out’赋给name，self._shape赋给shape，x赋给x。进一步调用self.get，原理同上
    elif isinstance(self._shape, dict):
      return {k: self._out(k, v, x) for k, v in self._shape.items()} # 生成多个命名的概率分布输出
    else:
      raise ValueError(self._shape)

  def _out(self, name, shape, x):
    return self.get(f'dist_{name}', Dist, shape, **self._dist)(x) # 将x转换为不同的概率分布


# 基于Resnet的图像编码器
class ImageEncoderResnet(nj.Module):

  def __init__(self, depth, blocks, resize, minres, **kw):
    self._depth = depth # 卷积层输出通道数
    self._blocks = blocks # 每阶段残差块数
    self._resize = resize # 下采样方式
    self._minres = minres # 最小分辨率
    self._kw = kw # 其他参数

  def __call__(self, x):
    stages = int(np.log2(x.shape[-2]) - np.log2(self._minres)) # 计算阶段数 = log2(输入分辨率) - log2(最小分辨率)
    depth = self._depth # 设置初始特征通道数
    x = jaxutils.cast_to_compute(x) - 0.5 # 预处理输入数据：归一化到[-0.5, 0.5]
    # print(x.shape)
    
    #遍历每个阶段
    for i in range(stages): 
      kw = {**self._kw, 'preact': False} # 设置卷积层参数，preact=False表示在卷积之后激活
      
      # 下采样层：将特征图尺寸缩小一半
      if self._resize == 'stride':
        x = self.get(f's{i}res', Conv2D, depth, 4, 2, **kw)(x) # 使用步幅为2、卷积核大小为 4 的卷积进行下采样。
      elif self._resize == 'stride3':
        s = 2 if i else 3
        k = 5 if i else 4
        x = self.get(f's{i}res', Conv2D, depth, k, s, **kw)(x) # 在第一个阶段使用步幅为 3、卷积核大小为 4，下一个阶段使用步幅为 2、卷积核大小为 5。
      elif self._resize == 'mean':
        N, H, W, D = x.shape
        x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x) # 先卷积，
        x = x.reshape((N, H // 2, W // 2, 4, D)).mean(-2)      # 然后通过重塑()和求平均()实现下采样。
      elif self._resize == 'max':
        x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x) # 先卷积，
        x = jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), 'same') # 然后使用最大池化进行下采样。
      else:
        raise NotImplementedError(self._resize)
      
      # 每个阶段中遍历所有残差块：进行特征提取
      for j in range(self._blocks): 
        skip = x # 保存残差连接的输入
        kw = {**self._kw, 'preact': True} # 设置残差块内卷积的参数，preact=True表示在卷积前激活
        x = self.get(f's{i}b{j}conv1', Conv2D, depth, 3, **kw)(x) # 残差块的3*3卷积层，‘=’左边的x相当于经过了第一个卷积层
        x = self.get(f's{i}b{j}conv2', Conv2D, depth, 3, **kw)(x) # 残差块的3*3卷积层，‘=’左边的x相当于经过了第二个卷积层
        x += skip # 添加残差连接，将输入与输出相加 将经过了两个卷积层的输出与最开始的输入相加
        # print(x.shape)

      # 每个阶段结束时将特征通道数翻倍  
      depth *= 2
    
    if self._blocks:
      x = get_act(self._kw['act'])(x) # 首先从关键字参数self.kw[]中获取激活函数名称，get_act()函数返回对应的激活函数，(ReLU、tanh...)。对最终的输出特征图x进行激活
    
    x = x.reshape((x.shape[0], -1)) # 将激活后的x特征图展平成二维向量。将特征图的第一维(batch_size)，与特征图的高、宽、深度展平成一维向量后的组合成一个二维向量。
                                    # 如：从(batch_size, height, width, channels)变为(batch_size, height * width * channels)
    # print(x.shape)
    return x #输出CNN编码器的特征向量
  

class ImageDecoderResnet(nj.Module):

  def __init__(self, shape, depth, blocks, resize, minres, sigmoid, **kw):
    self._shape = shape
    self._depth = depth
    self._blocks = blocks
    self._resize = resize
    self._minres = minres
    self._sigmoid = sigmoid
    self._kw = kw

  def __call__(self, x):
    stages = int(np.log2(self._shape[-2]) - np.log2(self._minres))
    depth = self._depth * 2 ** (stages - 1)
    x = jaxutils.cast_to_compute(x)
    x = self.get('in', Linear, (self._minres, self._minres, depth))(x)
    for i in range(stages):
      for j in range(self._blocks):
        skip = x
        kw = {**self._kw, 'preact': True}
        x = self.get(f's{i}b{j}conv1', Conv2D, depth, 3, **kw)(x)
        x = self.get(f's{i}b{j}conv2', Conv2D, depth, 3, **kw)(x)
        x += skip
        # print(x.shape)
      depth //= 2
      kw = {**self._kw, 'preact': False}
      if i == stages - 1:
        kw = {}
        depth = self._shape[-1]
      if self._resize == 'stride':
        x = self.get(f's{i}res', Conv2D, depth, 4, 2, transp=True, **kw)(x)
      elif self._resize == 'stride3':
        s = 3 if i == stages - 1 else 2
        k = 5 if i == stages - 1 else 4
        x = self.get(f's{i}res', Conv2D, depth, k, s, transp=True, **kw)(x)
      elif self._resize == 'resize':
        x = jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)
        x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
      else:
        raise NotImplementedError(self._resize)
    if max(x.shape[1:-1]) > max(self._shape[:-1]):
      padh = (x.shape[1] - self._shape[0]) / 2
      padw = (x.shape[2] - self._shape[1]) / 2
      x = x[:, int(np.ceil(padh)): -int(padh), :]
      x = x[:, :, int(np.ceil(padw)): -int(padw)]
    # print(x.shape)
    assert x.shape[-3:] == self._shape, (x.shape, self._shape)
    if self._sigmoid:
      x = jax.nn.sigmoid(x)
    else:
      x = x + 0.5
    return x


'''Basic network units'''

# 获取激活函数
def get_act(name): 
  if callable(name):
    return name
  elif name == 'none':
    return lambda x: x
  elif name == 'mish':
    return lambda x: x * jnp.tanh(jax.nn.softplus(x))
  elif hasattr(jax.nn, name):
    return getattr(jax.nn, name)
  else:
    raise NotImplementedError(name)


class Input:

  def __init__(self, keys=['tensor'], dims=None):
    assert isinstance(keys, (list, tuple)), keys
    self._keys = tuple(keys)
    self._dims = dims or self._keys[0] # if dims has content then is dims, else is keys[0](most likely 'deter')

  def __call__(self, inputs):
    # make inputs as a dict
    if not isinstance(inputs, dict):
      inputs = {'tensor': inputs}
    inputs = inputs.copy()

    # if inputs need softmax
    for key in self._keys:
      if key.startswith('softmax_'):
        inputs[key] = jax.nn.softmax(inputs[key[len('softmax_'):]])
    if not all(k in inputs for k in self._keys):
      needs = f'{{{", ".join(self._keys)}}}'
      found = f'{{{", ".join(inputs.keys())}}}'
      raise KeyError(f'Cannot find keys {needs} among inputs {found}.')
    
    # keep value in the same shape
    values = [inputs[k] for k in self._keys]
    dims = len(inputs[self._dims].shape)
    for i, value in enumerate(values):
      if len(value.shape) > dims:
        values[i] = value.reshape(
            value.shape[:dims - 1] + (np.prod(value.shape[dims - 1:]),))
        
    # recover data type since value data type may change in last step
    values = [x.astype(inputs[self._dims].dtype) for x in values]
    # concat input values (for example concat h and z as the input of decoder, reward and count heads)
    return jnp.concatenate(values, -1)


class Dist(nj.Module):

  def __init__(
      self, shape, dist='mse', outscale=0.1, outnorm=False, minstd=1.0,
      maxstd=1.0, unimix=0.0, bins=255):
    assert all(isinstance(dim, int) for dim in shape), shape
    self._shape = shape
    self._dist = dist
    self._minstd = minstd
    self._maxstd = maxstd
    self._unimix = unimix
    self._outscale = outscale
    self._outnorm = outnorm
    self._bins = bins

  def __call__(self, inputs):
    dist = self.inner(inputs)
    assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (
        dist.batch_shape, dist.event_shape, inputs.shape)
    return dist

  def inner(self, inputs):
    # print('dist')
    kw = {}
    kw['outscale'] = self._outscale
    kw['outnorm'] = self._outnorm
    shape = self._shape
    # discrete output value shape
    if self._dist.endswith('_disc'): # critic & reward
      shape = (*self._shape, self._bins)
    # mlp -> mean
    out = self.get('out', Linear, int(np.prod(shape)), **kw)(inputs)
    out = out.reshape(inputs.shape[:-1] + shape).astype(f32)
    # mlp -> std
    if self._dist in ('normal', 'trunc_normal'):
      std = self.get('std', Linear, int(np.prod(self._shape)), **kw)(inputs)
      std = std.reshape(inputs.shape[:-1] + self._shape).astype(f32)
    # return distribution
    if self._dist == 'symlog_mse': # decoder
      return jaxutils.SymlogDist(out, len(self._shape), 'mse', 'sum')
    if self._dist == 'symlog_disc': # critic & reward
      return jaxutils.DiscDist(
          out, len(self._shape), -20, 20, jaxutils.symlog, jaxutils.symexp)
    if self._dist == 'mse': 
      return jaxutils.MSEDist(out, len(self._shape), 'sum')
    if self._dist == 'normal':
      lo, hi = self._minstd, self._maxstd
      std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
      dist = tfd.Normal(jnp.tanh(out), std)
      dist = tfd.Independent(dist, len(self._shape))
      dist.minent = np.prod(self._shape) * tfd.Normal(0.0, lo).entropy()
      dist.maxent = np.prod(self._shape) * tfd.Normal(0.0, hi).entropy()
      return dist
    if self._dist == 'binary':
      dist = tfd.Bernoulli(out)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'onehot':
      if self._unimix:
        probs = jax.nn.softmax(out, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        out = jnp.log(probs)
      dist = jaxutils.OneHotDist(out)
      if len(self._shape) > 1:
        dist = tfd.Independent(dist, len(self._shape) - 1)
      dist.minent = 0.0
      dist.maxent = np.prod(self._shape[:-1]) * jnp.log(self._shape[-1])
      return dist
    raise NotImplementedError(self._dist)


class Conv2D(nj.Module):

  def __init__(
      self, depth, kernel, stride=1, transp=False, act='none', norm='none',
      pad='same', bias=True, preact=False, winit='uniform', fan='avg'):
    self._depth = depth
    self._kernel = kernel
    self._stride = stride
    self._transp = transp
    self._act = get_act(act)
    self._norm = Norm(norm, name='norm')
    self._pad = pad.upper()
    self._bias = bias and (preact or norm == 'none')
    self._preact = preact
    self._winit = winit
    self._fan = fan

  def __call__(self, hidden):
    if self._preact:
      hidden = self._norm(hidden)
      hidden = self._act(hidden)
      hidden = self._layer(hidden)
    else:
      hidden = self._layer(hidden)
      hidden = self._norm(hidden)
      hidden = self._act(hidden)
    return hidden

  def _layer(self, x):
    if self._transp:
      shape = (self._kernel, self._kernel, self._depth, x.shape[-1])
      kernel = self.get('kernel', Initializer(
          self._winit, fan=self._fan), shape)
      kernel = jaxutils.cast_to_compute(kernel)
      x = jax.lax.conv_transpose(
          x, kernel, (self._stride, self._stride), self._pad,
          dimension_numbers=('NHWC', 'HWOI', 'NHWC'))
    else:
      shape = (self._kernel, self._kernel, x.shape[-1], self._depth)
      kernel = self.get('kernel', Initializer(
          self._winit, fan=self._fan), shape)
      kernel = jaxutils.cast_to_compute(kernel)
      x = jax.lax.conv_general_dilated(
          x, kernel, (self._stride, self._stride), self._pad,
          dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    if self._bias:
      bias = self.get('bias', jnp.zeros, self._depth, np.float32)
      bias = jaxutils.cast_to_compute(bias)
      x += bias
    return x


class Linear(nj.Module):

  def __init__(
      self, units, act='none', norm='none', bias=True, outscale=1.0,
      outnorm=False, winit='uniform', fan='avg'):
    self._units = tuple(units) if hasattr(units, '__len__') else (units,)
    self._act = get_act(act)
    self._norm = norm
    self._bias = bias and norm == 'none'
    # print('bias', bias, norm, self._bias)
    self._outscale = outscale
    self._outnorm = outnorm
    self._winit = winit
    self._fan = fan

  def __call__(self, x):
    # mlp = k*x + b
    shape = (x.shape[-1], np.prod(self._units)) # if layer from 200 to 400, then needs shape=(200, 400)
    kernel = self.get('kernel', Initializer(
        self._winit, self._outscale, fan=self._fan), shape) # Initializer provides initial value of the kernels
    kernel = jaxutils.cast_to_compute(kernel)
    # k * x
    x = x @ kernel # 1x200 * 200x400 = 1x400
    # + b
    if self._bias:
      bias = self.get('bias', jnp.zeros, np.prod(self._units), np.float32) # bias is 0 at the beginning
      bias = jaxutils.cast_to_compute(bias)
      x += bias
    if len(self._units) > 1:
      x = x.reshape(x.shape[:-1] + self._units)
    # normlization (layer by default)
    x = self.get('norm', Norm, self._norm)(x)
    # activation
    x = self._act(x)
    return x


class Norm(nj.Module):

  def __init__(self, impl):
    self._impl = impl

  def __call__(self, x):
    dtype = x.dtype
    if self._impl == 'none':
      return x
    elif self._impl == 'layer':
      x = x.astype(f32)
      x = jax.nn.standardize(x, axis=-1, epsilon=1e-3)
      x *= self.get('scale', jnp.ones, x.shape[-1], f32)
      x += self.get('bias', jnp.zeros, x.shape[-1], f32)
      return x.astype(dtype)
    else:
      raise NotImplementedError(self._impl)


class Initializer:

  def __init__(self, dist='uniform', scale=1.0, fan='avg'):
    self.dist = dist
    self.scale = scale
    self.fan = fan

  def __call__(self, shape):
    if self.scale == 0.0:
      value = jnp.zeros(shape, f32)
    elif self.dist == 'uniform':
      fanin, fanout = self._fans(shape)
      denoms = {'avg': (fanin + fanout) / 2, 'in': fanin, 'out': fanout}
      scale = self.scale / denoms[self.fan]
      limit = np.sqrt(3 * scale)
      value = jax.random.uniform(
          nj.rng(), shape, f32, -limit, limit)
    # normal by default
    elif self.dist == 'normal':
      fanin, fanout = self._fans(shape) # 200 -> 400 mlp ==> fanin = 200, fanout = 400
      denoms = {'avg': np.mean((fanin, fanout)), 'in': fanin, 'out': fanout} # avg = 200 + 400 /2 = 300
      scale = self.scale / denoms[self.fan] # scale = 1 / 300
      std = np.sqrt(scale) / 0.87962566103423978
      value = std * jax.random.truncated_normal(
          nj.rng(), -2, 2, shape, f32)
    elif self.dist == 'ortho':
      nrows, ncols = shape[-1], np.prod(shape) // shape[-1]
      matshape = (nrows, ncols) if nrows > ncols else (ncols, nrows)
      mat = jax.random.normal(nj.rng(), matshape, f32)
      qmat, rmat = jnp.linalg.qr(mat)
      qmat *= jnp.sign(jnp.diag(rmat))
      qmat = qmat.T if nrows < ncols else qmat
      qmat = qmat.reshape(nrows, *shape[:-1])
      value = self.scale * jnp.moveaxis(qmat, 0, -1)
    else:
      raise NotImplementedError(self.dist)
    return value

  def _fans(self, shape):
    if len(shape) == 0:
      return 1, 1
    elif len(shape) == 1:
      return shape[0], shape[0]
    elif len(shape) == 2: # len(shape) usually is 2
      return shape
    else:
      space = int(np.prod(shape[:-2]))
      return shape[-2] * space, shape[-1] * space
