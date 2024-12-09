import embodied
import jax
import jax.numpy as jnp
import ruamel.yaml as yaml
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

import logging
logger = logging.getLogger()
class CheckTypesFilter(logging.Filter):
  def filter(self, record):
    return 'check_types' not in record.getMessage()
logger.addFilter(CheckTypesFilter())

from . import behaviors
from . import jaxagent
from . import jaxutils
from . import nets_PIWM, nets_original, nets_branch
from . import ninjax as nj


@jaxagent.Wrapper
class Agent(nj.Module):

  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, step, config):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step

    # create world model
    self.wm = WorldModel(obs_space, act_space, config, name='wm')

    # create actor and critic
    self.task_behavior = getattr(behaviors, config.task_behavior)(
        self.wm, self.act_space, self.config, name='task_behavior')
    # expl_behavior is 'None' by default
    if config.expl_behavior == 'None':
      self.expl_behavior = self.task_behavior
    else:
      self.expl_behavior = getattr(behaviors, config.expl_behavior)(
          self.wm, self.act_space, self.config, name='expl_behavior')

  def policy_initial(self, batch_size):
    return (
        self.wm.initial(batch_size),
        self.task_behavior.initial(batch_size),
        self.expl_behavior.initial(batch_size))

  def train_initial(self, batch_size):
    return self.wm.initial(batch_size)

  def policy(self, obs, state, mode='train'):
    self.config.jax.jit and print('Tracing policy function.')
    obs = self.preprocess(obs)
    (prev_latent, prev_action), task_state, expl_state = state

    # sample a action from the actor
    # NOTE: obs for ego and npc include id and traj state ---- currently dont know how to use id state
    embed_dict = self.wm.encoder(obs)
    # TODO: an elegant way to use different modules
    if self.config.task == 'interaction_prediction':
      post_dict, _ = self.wm.rssm.obs_step(
          prev_latent, prev_action, embed_dict, obs['is_first'], obs['should_init_npc'], obs['should_init_other'], obs['mask_npc'], obs['mask_other'])
    elif self.config.task == 'interaction_branch':
      post_dict, _ = self.wm.rssm.obs_step(
          prev_latent, prev_action, embed_dict, obs['is_first'], obs['should_init_npc'], obs['should_init_other'])
    elif self.config.task == 'interaction_recon':
      post_dict, _ = self.wm.rssm.obs_step(
          prev_latent, prev_action, embed_dict, obs['is_first'])
    
    # for policy in branch sturture, we use ego_attention
    # TODO: an elegant way to use different modules
    if self.config.task == 'interaction_prediction':
      feats_dict = {k: v for k,v in post_dict.items()}
      feats_dict.update({'mask_npc': obs['mask_npc']})
      ego_attention_out, ego_attention_mat = self.wm.ego_attention(feats_dict)
    elif self.config.task == 'interaction_branch':
      feats_dict = {k: v for k,v in post_dict.items()}
      feats_dict.update({'mask_npc': obs['mask_npc']})
      feats_dict.update({'mask_other': obs['mask_other']})
      ego_attention_out, ego_attention_mat = self.wm.ego_attention_branch(feats_dict)
    
    # TODO: an elegant way to use different modules
    if self.config.task == 'interaction_prediction':
      self.expl_behavior.policy(post_dict, expl_state, ego_attention_out)
      task_outs, task_state = self.task_behavior.policy(post_dict, task_state, ego_attention_out)
      expl_outs, expl_state = self.expl_behavior.policy(post_dict, expl_state, ego_attention_out)
    elif self.config.task == 'interaction_branch':
      self.expl_behavior.policy(post_dict, expl_state, ego_attention_out)
      task_outs, task_state = self.task_behavior.policy(post_dict, task_state, ego_attention_out)
      expl_outs, expl_state = self.expl_behavior.policy(post_dict, expl_state, ego_attention_out)
    elif self.config.task == 'interaction_recon':
      self.expl_behavior.policy(post_dict, expl_state)
      task_outs, task_state = self.task_behavior.policy(post_dict, task_state)
      expl_outs, expl_state = self.expl_behavior.policy(post_dict, expl_state)

    if mode == 'eval': # task behavior, no entropy
      outs = task_outs
      outs['action'] = outs['action'].sample(seed=nj.rng())
      outs['log_entropy'] = jnp.zeros(outs['action'].shape[:1])
    elif mode == 'explore': # explore behavior and random outcome
      outs = expl_outs
      outs['log_entropy'] = outs['action'].entropy()
      outs['action'] = outs['action'].sample(seed=nj.rng())
    elif mode == 'train': # task behavior and random outcome
      outs = task_outs
      outs['log_entropy'] = outs['action'].entropy()
      outs['action'] = outs['action'].sample(seed=nj.rng())
    state = ((post_dict, outs['action']), task_state, expl_state)
    return outs, state

  def predictor(self, obs, state):
    self.config.jax.jit and print('Tracing trajectory pridictor (decoder) function.')
    obs = self.preprocess(obs)

    (post_dict, _), _, _ = state
    feats_dict = {k: v for k,v in post_dict.items()}
    # feats_dict.update({
    #                    'mask_npc': obs['mask_npc'],
    #                    'mask_other': obs['mask_other']
    #                    })
    # predict_attention_out_dict, predict_attention_mat_dict = self.wm.predict_attention(feats_dict)
    # prediciton_symlog_dist_dict = self.wm.heads['decoder'](feats_dict, predict_attention_out_dict)
    prediciton_symlog_dist_dict = self.wm.heads['decoder'](feats_dict)
    prediciton = {key: value.mode() for key, value in prediciton_symlog_dist_dict.items()}
    return prediciton

  def train(self, data, state):
    self.config.jax.jit and print('Tracing train function.')
    metrics = {}
    # 1. preprocess data
    data = self.preprocess(data)

    # 2. train world models, and record losses and other values
    state, wm_outs, mets = self.wm.train(data, state)
    metrics.update(mets)
    # print('2 - wm train done')

    # 3. get posterior model state, as the start state 
    # of imagine trajectory (for actor critic training), along with the data
    # TODO: in context, for data we only consider is_terminal like in dmv2, is that important?
    # TODO: an elegant way to use different modules
    context = {**wm_outs['post']} # context = {**data, **wm_outs['post']}
    aux_key_list = ['is_terminal', 'mask_npc', 'mask_other'] if self.config.task in ['interaction_prediction', 'interaction_branch'] else ['is_terminal']
    for key in aux_key_list:
      context.update({key: data[key]})
    start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
    # print('3 - a-c initial data done')
    
    # 4. using world model, policy from Actor, and every data point(for example 16x64=1024 data points, 
    # as the beginning of imagination) to produce imagined experiences, and train Actor Critic
    _, mets = self.task_behavior.train(self.wm.imagine, start, context)
    metrics.update(mets)
    # print('4 - a-c train done')

    # 5. if exploration behavior is not the same as task behavior, train it as well
    if self.config.expl_behavior != 'None':
      _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
      metrics.update({'expl_' + key: value for key, value in mets.items()})

    # TODO: empty outs dict?
    outs = {}
    return outs, state, metrics

  def report(self, data):
    self.config.jax.jit and print('Tracing report function.')
    data = self.preprocess(data)
    report = {}
    report.update(self.wm.report(data))
    mets = self.task_behavior.report(data)
    report.update({f'task_{k}': v for k, v in mets.items()})
    if self.expl_behavior is not self.task_behavior:
      mets = self.expl_behavior.report(data)
      report.update({f'expl_{k}': v for k, v in mets.items()})
    return report

  def preprocess(self, obs):
    # print(obs['id_npc'])
    # uniform data type and scale image
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_') or key in ('key',):
        continue
      if len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = jaxutils.cast_to_compute(value) / 255.0
      else:
        value = value.astype(jnp.float32)
      obs[key] = value
    obs['cont'] = 1.0 - obs['is_terminal'].astype(jnp.float32)
    return obs


class WorldModel(nj.Module):

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.config = config
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
    # intialize modules of the world model
    # TODO: an elegant way to use different modules
    if self.config.task == 'interaction_prediction':
      # encoder for drive trajectories and map information
      self.encoder = nets_PIWM.PIWMEncoder(shapes, **config.encoder, name='enc')
      # attention modules which are used in different heads
      self.predict_attention = nets_PIWM.PredictAttention((), **config.attention, name='pred_attention')
      # recurrent state space model, modified for branch structure
      self.rssm = nets_PIWM.PIWMRSSM(shapes, self.predict_attention, **config.rssm, name='rssm')
      self.ego_attention = nets_PIWM.EgoAttention((), **config.attention, name='ego_attention')
      # different heads for predicting future trajectories, predicting reward or predicting episode end
      self.heads = {
          'decoder': nets_PIWM.PredictionDecoder(shapes, **config.decoder, name='dec'),
          'reward': nets_PIWM.PIWMMLP((), **config.reward_head, name='rew'),
          'cont': nets_PIWM.PIWMMLP((), **config.cont_head, name='cont'),
          }

    elif self.config.task == 'interaction_branch':
      self.heads = {}
      self.ego_attention_branch = nets_branch.BranchEgoAttention((), **config.attention, name='ego_attention_branch')
      # one branch for surrounding vehicles
      # self.encoder = nets_branch.OneBranchEncoder(shapes, **config.encoder, name='enc')
      # self.rssm = nets_branch.OneBranchRSSM(shapes, **config.rssm, name='rssm')
      # self.heads.update({'decoder': nets_branch.OneBranchDecoder(shapes, **config.decoder, name='dec')})
      # two branches for surrounding vehicles
      self.encoder = nets_PIWM.PIWMEncoder(shapes, **config.encoder, name='enc')
      self.rssm = nets_branch.TwoBranchRSSM(shapes, **config.rssm, name='rssm')
      self.heads.update({'decoder': nets_branch.TwoBranchDecoder(shapes, **config.decoder, name='dec')})
      # for only branch structure, we don't need prediction attention modules since all vehicles just recon their own history
      self.heads.update({
          'reward': nets_PIWM.PIWMMLP((), **config.reward_head, name='rew'),
          'cont': nets_PIWM.PIWMMLP((), **config.cont_head, name='cont'),
          })

    elif self.config.task == 'interaction_recon':
      self.encoder = nets_original.MultiEncoder(shapes, **config.encoder, name='enc')
      self.rssm = nets_original.RSSM(**config.rssm, name='rssm')
      self.heads = {
          'decoder': nets_original.MultiDecoder(shapes, **config.decoder, name='dec'),
          'reward': nets_original.MLP((), **config.reward_head, name='rew'),
          'cont': nets_original.MLP((), **config.cont_head, name='cont')
          }
    else:
      raise NotImplementedError(f'Unknown task {self.config.task}')

    self.opt = jaxutils.Optimizer(name='model_opt', **config.model_opt)
    # different loss scale, decoder targets' loss scales are modified to match the names
    scales = self.config.loss_scales.copy()
    image, vector = scales.pop('image'), scales.pop('vector')
    scales.update({k: image for k in self.heads['decoder'].cnn_shapes})
    # TODO: an elegant way to ...
    if self.config.task == 'interaction_prediction':
      # the prediction loss for different npcs is summed up to show
      for k in self.heads['decoder'].mlp_shapes:
        if k.startswith('npc_'):
          if 'npc_prediction' not in scales.keys():
            scales.update({'npc_prediction': vector})
        else:
          scales.update({k: vector})
    elif self.config.task == 'interaction_branch':
      # the recon loss for ego and different npcs and others are summed up to show
      for k in self.heads['decoder'].mlp_shapes:
        if k.startswith(('ego', 'npc_', 'other_')):
          if 'vehicle_recon' not in scales.keys():
            scales.update({'vehicle_recon': vector})
        else:
          scales.update({k: vector})
    elif self.config.task == 'interaction_recon':
      for k in self.heads['decoder'].mlp_shapes:
        scales.update({k: vector})

    self.scales = scales

  def initial(self, batch_size):
    prev_latent = self.rssm.initial(batch_size)
    prev_action = jnp.zeros((batch_size, *self.act_space.shape))
    return prev_latent, prev_action

  def train(self, data, state):
    # TODO: an elegant way to use different modules
    if self.config.task == 'interaction_prediction':
      # encoder, rssm, predict_attention for future prediction, ego_attention for decision and reward predict, decoder, reward, cont
      modules = [self.encoder, self.rssm, self.predict_attention, self.ego_attention, *self.heads.values()] 
    elif self.config.task == 'interaction_branch':
      modules = [self.encoder, self.rssm, self.ego_attention_branch, *self.heads.values()]
    elif self.config.task == 'interaction_recon':
      modules = [self.encoder, self.rssm, *self.heads.values()]
    # print('modules len:', len(modules))
    # print('loss:', self.loss)
    # print('data:', data.keys())
    # print('state:', state)
    mets, (state, outs, metrics) = self.opt(
        modules, self.loss, data, state, has_aux=True)
    metrics.update(mets)
    # print(metrics.keys())
    return state, outs, metrics

  def loss(self, data, state):
    # extract features from observation
    embed_dict = self.encoder(data)

    # get z and ^z
    prev_latent, prev_action = state
    prev_actions = jnp.concatenate([
        prev_action[:, None], data['action'][:, :-1]], 1)
    
    # TODO: an elegant way to use different modules
    if self.config.task == 'interaction_prediction':
      post_dict, prior_dict = self.rssm.observe(
          embed_dict, prev_actions, data['is_first'], data['should_init_npc'], data['should_init_other'], data['mask_npc'], data['mask_other'], prev_latent)
    elif self.config.task == 'interaction_branch':
      post_dict, prior_dict = self.rssm.observe(
          embed_dict, prev_actions, data['is_first'], data['should_init_npc'], data['should_init_other'], prev_latent)
    elif self.config.task == 'interaction_recon':
      post_dict, prior_dict = self.rssm.observe(
          embed_dict, prev_actions, data['is_first'], prev_latent)
    
    # print('post keys:', post_dict.keys())
    
    # TODO: feats include embed in original v3? dont know why, since head dont use embed as input, dont consider it:
    # for key, embed in embed_dict.items():
    #   feats_dict[key].update({'embed': embed})
    feats_dict = {k: v for k,v in post_dict.items()}
        
    # TODO: an elegant way to use different modules
    # get agent to agent attention
    if self.config.task == 'interaction_prediction':
      feats_dict.update({
                        'mask_npc': data['mask_npc'],
                        'mask_other': data['mask_other'],
                         })
      # predict_attention_out_dict, predict_attention_mat_dict = self.predict_attention(feats_dict)
      ego_attention_out, ego_attention_mat = self.ego_attention(feats_dict)
    elif self.config.task == 'interaction_branch':
      feats_dict.update({
                        'mask_npc': data['mask_npc'],
                        'mask_other': data['mask_other'],
                         })
      ego_attention_out, ego_attention_mat = self.ego_attention_branch(feats_dict)

    # feat is a concat of z and h (stoch and deter)
    # TODO: use feats_dict or post_dict? seems like post_dict is enough
    dists = {}
    for name, head in self.heads.items():
      # TODO: an elegant way to use different modules
      if self.config.task == 'interaction_prediction':
        if name in ['decoder']:
          out = head(feats_dict if name in self.config.grad_heads else sg(feats_dict))
        elif name in ['reward', 'cont']:
          out = head(feats_dict, ego_attention_out if name in self.config.grad_heads else sg(feats_dict, ego_attention_out))
      elif self.config.task == 'interaction_branch':
        if name in ['decoder']:
          out = head(feats_dict if name in self.config.grad_heads else sg(feats_dict))
        elif name in ['reward', 'cont']:
          out = head(feats_dict, ego_attention_out if name in self.config.grad_heads else sg(feats_dict, ego_attention_out))
      elif self.config.task == 'interaction_recon':
        out = head(feats_dict if name in self.config.grad_heads else sg(feats_dict))
      out = out if isinstance(out, dict) else {name: out}
      dists.update(out)

    # calculate loss, for losses related to npc, we sum them up
    losses = {}
    # TODO: an elegant way to use different modules
    if self.config.task == 'interaction_prediction':
      # dyn and rep loss
      losses['dyn'] = self.rssm.dyn_loss(post_dict, prior_dict, data['mask_npc'], data['mask_other'], **self.config.dyn_loss)
      losses['rep'] = self.rssm.rep_loss(post_dict, prior_dict, data['mask_npc'], data['mask_other'], **self.config.rep_loss)
      # sum up npc future prediction loss
      loss = jnp.zeros(embed_dict['ego'].shape[:2])
      for key, dist in dists.items():
        if key.startswith('npc_'):
          gt = data[key].astype(jnp.float32)
          # mask out zero-padded future trajectories
          valid = jnp.where(gt, 1, 0)
          # print('gt and its valid shape:', gt.shape, valid.shape)
          loss += -dist.log_prob(gt, valid)
      # TODO: below 2 line can ctrl+[? 
      assert loss.shape == embed_dict['ego'].shape[:2], (key, loss.shape) # embed_dict['ego'].shape[:2] is batch size (16, 64)
      losses.update({'npc_prediction': loss})
      # other losses (ego future prediction, reward, cont)
      for key, dist in dists.items():
        if not key.startswith('npc_'):
          gt = data[key].astype(jnp.float32)
          loss = -dist.log_prob(gt)
          assert loss.shape == embed_dict['ego'].shape[:2], (key, loss.shape) # embed_dict['ego'].shape[:2] is batch size (16, 64)
          losses[key] = loss
      # scaled loss - no kl
      scaled = {k: v * self.scales[k] for k, v in losses.items() if k not in ('dyn', 'rep')}
      # scaled loss - add kl
      for kl in ('dyn', 'rep'):
        scaled.update({k: v * self.scales[kl] for k,v in losses[kl].items()})
        if isinstance(losses[kl], dict): # unzip losses from losses[kl] dict
          losses.update({k: v for k,v in losses[kl].items()})
          losses.pop(kl)
      model_loss = sum(scaled.values())

    elif self.config.task == 'interaction_branch':
      # dyn and rep loss
      losses['dyn'] = self.rssm.dyn_loss(post_dict, prior_dict, data['mask_npc'], data['mask_other'], **self.config.dyn_loss)
      losses['rep'] = self.rssm.rep_loss(post_dict, prior_dict, data['mask_npc'], data['mask_other'], **self.config.rep_loss)
      # sum up vehicle(npc + other) branch recon loss
      loss = jnp.zeros(embed_dict['ego'].shape[:2])
      for key, dist in dists.items():
        if key.startswith(('ego', 'npc_', 'other_')):
          loss += -dist.log_prob(data[key].astype(jnp.float32))
      assert loss.shape == embed_dict['ego'].shape[:2], (key, loss.shape)
      losses.update({'vehicle_recon': loss})
      # other losses (reward, cont)
      for key, dist in dists.items():
        if (not key.startswith('ego')) and (not key.startswith('npc_')) and (not key.startswith('other_')):
          loss = -dist.log_prob(data[key].astype(jnp.float32))
          assert loss.shape == embed_dict['ego'].shape[:2], (key, loss.shape)
          losses[key] = loss
      # scaled loss - no kl
      scaled = {k: v * self.scales[k] for k, v in losses.items() if k not in ('dyn', 'rep')}
      # scaled loss - add kl
      for kl in ('dyn', 'rep'):
        scaled.update({k: v * self.scales[kl] for k,v in losses[kl].items()})
        if isinstance(losses[kl], dict): # unzip losses from losses[kl] dict
          losses.update({k: v for k,v in losses[kl].items()})
          losses.pop(kl)
      model_loss = sum(scaled.values())

    elif self.config.task == 'interaction_recon':
      losses['dyn'] = self.rssm.dyn_loss(post_dict, prior_dict, **self.config.dyn_loss)
      losses['rep'] = self.rssm.rep_loss(post_dict, prior_dict, **self.config.rep_loss)
      for key, dist in dists.items():
        loss = -dist.log_prob(data[key].astype(jnp.float32))
        assert loss.shape == embed_dict.shape[:2], (key, loss.shape)
        losses[key] = loss
      scaled = {k: v * self.scales[k] for k, v in losses.items()}
      model_loss = sum(scaled.values())

    # out is for the beginning of a-c training in imagination
    out = {'embed': embed_dict, 'post': post_dict, 'prior': prior_dict}
    out.update({f'{k}_loss': v for k, v in losses.items()})
    # record loss value in tensorboard
    metrics = self._metrics(data, dists, post_dict, prior_dict, losses, model_loss)
    
    # TODO: what is last state (dict) for?
    last_state_dict = {}
    # TODO: an elegant way...
    if self.config.task == 'interaction_prediction':
      for key, key_dict in post_dict.items():
        last_state_dict[key] = {k: v[:, -1] for k, v in key_dict.items()}
    elif self.config.task == 'interaction_branch':
      for key, key_dict in post_dict.items():
        last_state_dict[key] = {k: v[:, -1] for k, v in key_dict.items()}
    elif self.config.task == 'interaction_recon':
      last_state_dict = {k: v[:, -1] for k, v in post_dict.items()}
    last_action = data['action'][:, -1]
    state = last_state_dict, last_action

    # print('metrics key:', metrics.keys())
    return model_loss.mean(), (state, out, metrics)

  def imagine(self, policy, start, horizon):
    # start keys: 'action', 'cont', 'ego', 'is_first', 'is_last', 'is_terminal', 'npc_1', 'reward'...
    first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)
    # start state, which means ego and npc_1...npc_n, original the keys should be deter, logit, stoch 
    keys = list(self.rssm.initial(1).keys())
    start_dict = {k: v for k, v in start.items() if k in keys}
    
    # NOTE: the mask of npc and other is unchanged during imagination (i.e the number of npc and other is fixed)
    # TODO: an elegant way to use different modules
    if self.config.task == 'interaction_prediction':
      # update mask for attention
      feats_dict = {k: v for k,v in start_dict.items()}
      feats_dict.update({
                        'mask_npc': start['mask_npc'],
                        'mask_other': start['mask_other'],
                        })
      # for actor, the gradient is not backpropagated to the world model, but ego attention is updated
      ego_attention_out, ego_attention_mat = self.ego_attention(sg(feats_dict))
      start_dict['attention'] = ego_attention_out
      # get start action from policy
      start_dict['action'] = policy(feats_dict, ego_attention_out)

    elif self.config.task == 'interaction_branch':
      # update mask for attention
      feats_dict = {k: v for k,v in start_dict.items()}
      feats_dict.update({
                        'mask_npc': start['mask_npc'],
                        'mask_other': start['mask_other'],
                        })
      ego_attention_out, ego_attention_mat = self.ego_attention_branch(sg(feats_dict))
      start_dict['attention'] = ego_attention_out
      start_dict['action'] = policy(feats_dict, ego_attention_out)

    elif self.config.task == 'interaction_recon':
      feats_dict = {k: v for k,v in start_dict.items()}
      start_dict['action'] = policy(feats_dict)
    # print(mask_npc.shape, start_dict['action'].shape)
    # print(start_dict.keys(), start_dict['ego']['deter'].shape, start_dict['ego']['stoch'].shape, start_dict['mask_npc'].shape, start_dict['action'].shape)

    # generate imagined trajetories using world model to train the actor and critc
    def step(prev, _):
      prev = prev.copy() # prev is prev prior_dict(or start_dict in the first step) plus action
      # TODO: an elegant way to use different modules
      if self.config.task == 'interaction_prediction':
        prior_dict, _ = self.rssm.img_step(prev, prev.pop('action'), start['mask_npc'], start['mask_other'])
        feats_dict = {k: v for k,v in prior_dict.items()}
        # NOTE: the mask of npc and other is unchanged during imagination
        feats_dict.update({
                          'mask_npc': start['mask_npc'],
                          'mask_other': start['mask_other'],
                          })
        # for actor, the gradient is not backpropagated to the world model
        ego_attention_out, ego_attention_mat = self.ego_attention(sg(feats_dict))
        # get action for these imagined states
        action = policy(feats_dict, ego_attention_out)
        out = {**prior_dict, 'action': action, 'attention': ego_attention_out}

      elif self.config.task == 'interaction_branch':
        prior_dict = self.rssm.img_step(prev, prev.pop('action'))
        feats_dict = {k: v for k,v in prior_dict.items()}
        # NOTE: the mask of npc and other is unchanged during imagination
        feats_dict.update({
                          'mask_npc': start['mask_npc'],
                          'mask_other': start['mask_other'],
                          })
        # for actor, the gradient is not backpropagated to the world model
        ego_attention_out, ego_attention_mat = self.ego_attention_branch(sg(feats_dict))
        # get action for these imagined states
        action = policy(feats_dict, ego_attention_out)
        out = {**prior_dict, 'action': action, 'attention': ego_attention_out}
      
      elif self.config.task == 'interaction_recon':
        prior_dict = self.rssm.img_step(prev, prev.pop('action'))
        feats_dict = {k: v for k,v in prior_dict.items()}
        # get action for these imagined states
        action = policy(feats_dict)
        out = {**prior_dict, 'action': action}
      return out
    
    traj = jaxutils.scan(
        step, jnp.arange(horizon), start_dict, self.config.imag_unroll) # imag_unroll is False by default

    # concat first ground truth state (from data) and 15 (imagine horizen) imagined state
    traj_concat = {}
    # TODO: an elegant way...
    if self.config.task == 'interaction_prediction':
      # for branch struchture
      for k, v in traj.items():
        traj_concat.update({k: {veh: jnp.concatenate([start_dict[k][veh][None], v[veh]], 0) for veh in v.keys()}} if isinstance(v, dict) else {k: jnp.concatenate([start_dict[k][None], v], 0)})
    elif self.config.task == 'interaction_branch':
      for k, v in traj.items():
        traj_concat.update({k: {veh: jnp.concatenate([start_dict[k][veh][None], v[veh]], 0) for veh in v.keys()}} if isinstance(v, dict) else {k: jnp.concatenate([start_dict[k][None], v], 0)})
    elif self.config.task == 'interaction_recon':
      traj_concat = {k: jnp.concatenate([start_dict[k][None], v], 0) for k, v in traj.items()}

    # pridict ends or not for each imagined state and calculate discount weight for these states
    # TODO: an elegant way to use different modules
    if self.config.task == 'interaction_prediction':
      cont = self.heads['cont'](traj_concat, traj_concat['attention']).mode()
    elif self.config.task == 'interaction_branch':
      cont = self.heads['cont'](traj_concat, traj_concat['attention']).mode()
    elif self.config.task == 'interaction_recon':
      cont = self.heads['cont'](traj_concat).mode()
    traj_concat['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)
    discount = 1 - 1 / self.config.horizon
    traj_concat['weight'] = jnp.cumprod(discount * traj_concat['cont'], 0) / discount
    # print('traj_concat keys', traj_concat.keys())
    return traj_concat

  def report(self, data):
    state = self.initial(len(data['is_first']))
    report = {}
    report.update(self.loss(data, state)[-1][-1])
    context, _ = self.rssm.observe(
        self.encoder(data)[:6, :5], data['action'][:6, :5],
        data['is_first'][:6, :5])
    start = {k: v[:, -1] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    openl = self.heads['decoder'](
        self.rssm.imagine(data['action'][:6, 5:], start))
    for key in self.heads['decoder'].cnn_shapes.keys():
      truth = data[key][:6].astype(jnp.float32)
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      error = (model - truth + 1) / 2
      video = jnp.concatenate([truth, model, error], 2)
      report[f'openl_{key}'] = jaxutils.video_grid(video)
    return report

  def _metrics(self, data, dists, post_dict, prior_dict, losses, model_loss):
    metrics = {}
    # TODO: dont consider entropy since data(for npc) changes across episodes (or consider its mean?, its shape is (batch x horizon))
    # entropy = lambda state: self.rssm.get_dist(state).entropy()
    # for key in prior_dict.keys():
    #   prior, post = prior_dict[key], post_dict[key]
    #   metrics.update(jaxutils.tensorstats(entropy(prior), key + '_prior_ent'))
    #   metrics.update(jaxutils.tensorstats(entropy(post), key + '_post_ent'))

    # mean and std for different modules' original losses
    metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    # total model loss, after scale
    metrics['model_loss_mean'] = model_loss.mean()
    metrics['model_loss_std'] = model_loss.std()
    metrics['reward_max_data'] = jnp.abs(data['reward']).max()
    metrics['reward_max_pred'] = jnp.abs(dists['reward'].mean()).max()

    if 'reward' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
      metrics.update({f'reward_{k}': v for k, v in stats.items()})
    if 'cont' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
      metrics.update({f'cont_{k}': v for k, v in stats.items()})
    return metrics


class ImagActorCritic(nj.Module):

  def __init__(self, critics, scales, act_space, config):
    # define critic(=vfunction)
    critics = {k: v for k, v in critics.items() if scales[k]}
    for key, scale in scales.items():
      assert not scale or key in critics, key
    self.critics = {k: v for k, v in critics.items() if scales[k]}
    self.scales = scales
    self.act_space = act_space
    self.config = config
    disc = act_space.discrete
    self.grad = config.actor_grad_disc if disc else config.actor_grad_cont

    # TODO: an elegant way to use different modules
    if self.config.task == 'interaction_prediction':
      self.actor = nets_PIWM.PIWMMLP(
          name='actor', dims='deter', shape=act_space.shape, **config.actor,
          dist=config.actor_dist_disc if disc else config.actor_dist_cont)
    elif self.config.task == 'interaction_branch':
      self.actor = nets_PIWM.PIWMMLP(
          name='actor', dims='deter', shape=act_space.shape, **config.actor,
          dist=config.actor_dist_disc if disc else config.actor_dist_cont)
    elif self.config.task == 'interaction_recon':
      self.actor = nets_original.MLP(
          name='actor', dims='deter', shape=act_space.shape, **config.actor,
          dist=config.actor_dist_disc if disc else config.actor_dist_cont)
    
    # TODO: what is retnorms?
    self.retnorms = {
        k: jaxutils.Moments(**config.retnorm, name=f'retnorm_{k}')
        for k in critics}
    self.opt = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)

  def initial(self, batch_size):
    return {}

  def policy(self, state, carry, attention=None):
    # TODO: an elegant way to use different modules
    # TODO: is this for eval only?
    if self.config.task == 'interaction_prediction':
      action = self.actor(state, attention)
    elif self.config.task == 'interaction_branch':
      action = self.actor(state, attention)
    elif self.config.task == 'interaction_recon':
      action = self.actor(state)
    # TODO: carry is unchanged?
    return {'action': action}, carry

  def train(self, imagine, start, context):
    def loss(start):
      # TODO: an elegant way to...
      if self.config.task == 'interaction_prediction':
        # NOTE: grad pass through ego_attention module in actor and critic
        # ego attention itself as a part of a-c is trained during imagination, 
        # yet the gradients is not back propagated to the other parts world model
        policy = lambda s, att: self.actor(sg(s), att).sample(seed=nj.rng())
      elif self.config.task == 'interaction_branch':
        policy = lambda s, att: self.actor(sg(s), att).sample(seed=nj.rng())
      elif self.config.task == 'interaction_recon':
        policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
      # generate a trajectory in imagination
      traj = imagine(policy, start, self.config.imag_horizon)
      # calculate losses for actor and critic
      loss, metrics = self.loss(traj)
      return loss, (traj, metrics)
    mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
    metrics.update(mets)
    # update/train
    for key, critic in self.critics.items():
      mets = critic.train(traj, self.actor)
      metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    # print('critic train done')
    return traj, metrics

  def loss(self, traj):
    metrics = {}
    advs = []
    total = sum(self.scales[k] for k in self.critics)
    for key, critic in self.critics.items():
      rew, ret, base = critic.score(traj, self.actor)
      offset, invscale = self.retnorms[key](ret)
      normed_ret = (ret - offset) / invscale
      normed_base = (base - offset) / invscale
      advs.append((normed_ret - normed_base) * self.scales[key] / total)
      metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
      metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
      metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
      metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
    adv = jnp.stack(advs).sum(0)
    # TODO: an elegant way to use different modules
    if self.config.task == 'interaction_prediction':
      # NOTE: grad pass through ego_attention module in actor and critic and not back propagated to the other parts of world model
      policy = self.actor(sg(traj), traj['attention'])
    elif self.config.task == 'interaction_branch':
      policy = self.actor(sg(traj), traj['attention'])
    elif self.config.task == 'interaction_recon':
      policy = self.actor(sg(traj))
    # calculate loss
    logpi = policy.log_prob(sg(traj['action']))[:-1]
    loss = {'backprop': -adv, 'reinforce': -logpi * sg(adv)}[self.grad]
    ent = policy.entropy()[:-1]
    loss -= self.config.actent * ent
    loss *= sg(traj['weight'])[:-1]
    loss *= self.config.loss_scales.actor
    metrics.update(self._metrics(traj, policy, logpi, ent, adv))
    return loss.mean(), metrics

  def _metrics(self, traj, policy, logpi, ent, adv):
    metrics = {}
    ent = policy.entropy()[:-1]
    rand = (ent - policy.minent) / (policy.maxent - policy.minent)
    rand = rand.mean(range(2, len(rand.shape)))
    act = traj['action']
    act = jnp.argmax(act, -1) if self.act_space.discrete else act
    metrics.update(jaxutils.tensorstats(act, 'action'))
    metrics.update(jaxutils.tensorstats(rand, 'policy_randomness'))
    metrics.update(jaxutils.tensorstats(ent, 'policy_entropy'))
    metrics.update(jaxutils.tensorstats(logpi, 'policy_logprob'))
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])
    return metrics


class VFunction(nj.Module):

  def __init__(self, rewfn, config):
    self.rewfn = rewfn
    self.config = config
    # v net and its slow update target net
    # TODO: an elegant way to use different modules
    if self.config.task == 'interaction_prediction':
      self.net = nets_PIWM.PIWMMLP((), name='net', dims='deter', **self.config.critic)
      self.slow = nets_PIWM.PIWMMLP((), name='slow', dims='deter', **self.config.critic)
    elif self.config.task == 'interaction_branch':
      self.net = nets_PIWM.PIWMMLP((), name='net', dims='deter', **self.config.critic)
      self.slow = nets_PIWM.PIWMMLP((), name='slow', dims='deter', **self.config.critic)
    elif self.config.task == 'interaction_recon':
      self.net = nets_original.MLP((), name='net', dims='deter', **self.config.critic)
      self.slow = nets_original.MLP((), name='slow', dims='deter', **self.config.critic)

    # target update
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update)
    # optimizer
    self.opt = jaxutils.Optimizer(name='critic_opt', **self.config.critic_opt)

  def train(self, traj, actor):
    target = sg(self.score(traj)[1])
    mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
    metrics.update(mets)
    self.updater()
    return metrics

  def loss(self, traj, target):
    metrics = {}
    # exacpt the last one from original traj
    traj_slide = {}
    for k, v in traj.items():
      traj_slide.update({k: {k2: v2[:-1] for k2, v2 in v.items()}} if isinstance(v, dict) else {k: v[:-1]})
    # get value predictions
    # TODO: an elegant way to use different modules
    if self.config.task == 'interaction_prediction':
      v_dist = self.net(traj_slide, traj_slide['attention'])
      slow_dist = self.slow(traj_slide, traj_slide['attention'])
    elif self.config.task == 'interaction_branch':
      v_dist = self.net(traj_slide, traj_slide['attention'])
      slow_dist = self.slow(traj_slide, traj_slide['attention'])
    elif self.config.task == 'interaction_recon':
      v_dist = self.net(traj_slide)
      slow_dist = self.slow(traj_slide)
      
    # calculate loss
    loss = -v_dist.log_prob(sg(target))
    if self.config.critic_slowreg == 'logprob': # 'logprob' by default
      reg = -v_dist.log_prob(sg(slow_dist.mean()))
    elif self.config.critic_slowreg == 'xent':
      reg = -jnp.einsum(
          '...i,...i->...',
          sg(slow_dist.probs),
          jnp.log(v_dist.probs))
    else:
      raise NotImplementedError(self.config.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    loss = (loss * sg(traj_slide['weight'])).mean()
    loss *= self.config.loss_scales.critic
    metrics = jaxutils.tensorstats(v_dist.mean())
    return loss, metrics

  def score(self, traj, actor=None):
    # using reward head from world model to calculate reward for imagined trajectory
    # TODO: an elegant way to use different modules
    if self.config.task == 'interaction_prediction':
      rew = self.rewfn(traj, traj['attention'])
      value = self.net(traj, traj['attention']).mean()
    elif self.config.task == 'interaction_branch':
      rew = self.rewfn(traj, traj['attention'])
      value = self.net(traj, traj['attention']).mean()
    elif self.config.task == 'interaction_recon':
      rew = self.rewfn(traj)
      value = self.net(traj).mean()
    assert len(rew) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
    # calculate discounted returns
    discount = 1 - 1 / self.config.horizon
    disc = traj['cont'][1:] * discount
    vals = [value[-1]]
    interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
    for t in reversed(range(len(disc))):
      vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    ret = jnp.stack(list(reversed(vals))[:-1])
    return rew, ret, value[:-1]