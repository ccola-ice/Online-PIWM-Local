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

''' Modified networks for Predictive Indivitial World Model (PIWM) '''

class PIWMRSSM(nj.Module):

  def __init__(
      self, shapes, pred_attention, deter=1024, stoch=32, classes=32, unroll=False, initial='learned',
      unimix=0.01, action_clip=1.0, **kw):
    
    # multi heads of rssm(ego, npc_1...npc_n, other_1....other_n)
    rssm_included = [k for k in shapes.keys() if k.startswith(('ego', 'npc', 'other')) and 'prediction' not in k]
    self._keys = [k for k in shapes.keys() if k in rssm_included]
    print('RSSM keys:', self._keys) # TODO:RSSM keys ?

    # prediction attention network for merging vehicles' feature
    self._pred_attention = pred_attention # self-attention module, consider vehicles' interactive features when preducing z
    self._attention = self._pred_attention.out_dim # attention  dims

    # rssm network hyparameters
    self._deter = deter # 'h' in the paper
    self._stoch = stoch # 'z' in the paper
    self._classes = classes # classes of 'z'
    self._unroll = unroll # TODO: usually False?
    self._initial = initial # 'learned' by config default, means zero-initial h and use learned nets produce z 
    self._unimix = unimix # true out = (1-unimix) * net out + unimix * random
    self._action_clip = action_clip
    self._kw = kw
  
  def initial(self, batch_size): # bs=batch_size 16 by default
    state_dict = {}
    for key in self._keys:
      # initialize latent state(z) based on different branches if use learned weights to do it
      state = self._initial_single_state(key, batch_size)
      state_dict.update({key: state})
    return state_dict
  
  def _get_post(self, key, prior, pred_attention, embed):
    # 获取对应的前缀(ego/npc/other)
    prefix = 'ego' if key.startswith('ego') else 'npc' if key.startswith('npc_') else 'other'
    # 特征拼接
    x = jnp.concatenate([prior['deter'], pred_attention, embed], -1)
    # 获取观测输出
    x = self.get(f'{prefix}_obs_out', Linear, **self._kw)(x)
    # 获取后验分布
    stats = self._stats(f'{prefix}_obs_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    return {'stoch': stoch, 'deter': prior['deter'], **stats}

  def observe(self, embed, action, is_first, should_init_npc, should_init_other, mask_npc, mask_other, state_dict=None):
    # embed: x_t in paper, action: a_t in paper
    # ego state and npc state should be divided, also, npc state should be divided into each npc
    if state_dict is None: 
      state_dict = self.initial(batch_size=action.shape[0]) # size is batch_size (16 by default)

    # swap: change x shape from (16, 64, xx) to (64, 16, xx) (or from (64, 16, xx) to (16, 64, xx))
    # TODO: notice for obs_step, the start is state_dict, not tuple(state_dict, state_dict), figure out why it works like this
    step = lambda prev, inputs: self.obs_step(prev[0], *inputs)
    # 对所有的输入交换前两个维度，使其变为(64, 16, xx)
    inputs = self._swap(action), self._swap(embed), self._swap(is_first), self._swap(should_init_npc), self._swap(should_init_other), self._swap(mask_npc), self._swap(mask_other)
    start = state_dict, state_dict

    # 从 start 开始，按顺序将 inputs 的每个时间步输入到 step 函数，就是obs_step中。
    # post_dict 和 prior_dict 收集所有时间步的前验和后验状态。
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
    prior_dict, pred_attention_dict = jaxutils.scan(self.img_step, action, state_dict, self._unroll)
    # swap it back to the original shape (64, 16, xx)
    prior_dict = {k: self._swap(v) for k, v in prior_dict.items()}
    return prior_dict

  # 进行一个观测步训练
  # 输入：前一个状态h(t-1)，前一个动作a(t-1)，嵌入字典e(t)，是否为第一帧，是否初始化npc，是否初始化other，npc掩码，other掩码
  # 输出：z_t z_t_head
  def obs_step(self, prev_state, prev_action, embed_dict, is_first, should_init_npc, should_init_other, mask_npc, mask_other):
       
    # change data type
    is_first = cast(is_first)
    prev_action = cast(prev_action)
    
    # action clip 动作剪枝
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(prev_action)))
    # self._action_clip / jnp.maximum(self._action_clip, jnp.abs(prev_action)) : 缩放因子
    # sg:停止梯度传播，确保剪枝操作不影响反向传播。
  
    # intialize latent state for first frame
    prev_state, prev_action = jax.tree_util.tree_map(
        lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
    # if is first frame, than state and action is set to 0, else keep it as it is
    # 使用 jax.tree_util.tree_map 对 prev_state 和 prev_action 进行掩码处理：
    # 如果是第一帧，则使用 prev_state = self._mask(prev_state, 1.0 - 1.0); prev_action = self._mask(prev_action, 1.0 - 1.0) 对状态和动作进行掩码处理。
    # 如果不是第一帧，则为 prev_state = self._mask(prev_state, 1.0 - 0.0); prev_action = self._mask(prev_action, 1.0 - 0.0)。相当于维持不变。

    prev_state = jax.tree_util.tree_map(
        lambda x, y: x + self._mask(y, is_first), prev_state, self.initial(len(is_first)))
    # for state in first frame is then initialized by the rssm
    # 同理，如果是第一帧，则为prev_state = prev_state + self._mask(self.initial(len(is_first)),1.0)
    # 如果不是第一帧，   则为prev_state = prev_state + self._mask(self.initial(len(is_first)),0.0)

    # initialize latent state for new or zero-padded(masked) npcs/others
    # TODO: initial muilple times when is_first is True, waste of calculation
    should_init_npc, should_init_other = cast(should_init_npc), cast(should_init_other) #should_init_npc:(16,5) should_init_other:(16,5)
    for key in prev_state.keys():
      if key.startswith('npc_'):
        npc_index = int(key.split('_')[-1]) # npc_index:1
        should_init = should_init_npc[:, npc_index - 1] # should_init:(float16[16])
        # for npcs who are new to the observation, set its state to 0 first
        zero_paded_state = jax.tree_util.tree_map(lambda x: self._mask(x, 1.0 - should_init), prev_state[key]) # zero_paded_state = self._mask(prev_state['npc_1'], 1.0 - should_init)
        # and then initialize it using rssm initial function
        prev_state[key] = jax.tree_util.tree_map(lambda x, y: x + self._mask(y, should_init), zero_paded_state, self._initial_single_state(key, len(should_init))) # prev_state['npc_1'] = zero_paded_state + self._mask(self._initial_single_state('npc_1', 16), should_init)
      elif key.startswith('other_'):
        other_index = int(key.split('_')[-1])
        should_init = should_init_other[:, other_index - 1]
        zero_paded_state = jax.tree_util.tree_map(lambda x: self._mask(x, 1.0 - should_init), prev_state[key])
        prev_state[key] = jax.tree_util.tree_map(lambda x, y: x + self._mask(y, should_init), zero_paded_state, self._initial_single_state(key, len(should_init)))
    
    # 计算当前的先验状态并预测注意力值
    # get prior ^z(t)
    prior_dict, pred_attention_dict = self.img_step(prev_state, prev_action, mask_npc, mask_other)
    
    post_dict = {}# 
    # different branches
    # TODO: merge codes below
    for key in prior_dict.keys():
      if key.startswith(('ego', 'npc_', 'other_')):
        prior = prior_dict[key]
        post = self._get_post(
            key,
            prior,
            pred_attention_dict[key],
            embed_dict[key]
        )
        post_dict.update({key: post})

    return cast(post_dict), cast(prior_dict) 
  
  def img_step(self, prev_state_dict, prev_action, mask_npc, mask_other):
    # 输入
    # change data type
    prev_action = cast(prev_action)
    # action clip
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))

    def process_prior(self, key, feats_dict, pred_attention_dict, deter_dict):
      # 获取前缀(ego/npc/other)
      prefix = key.split('_')[0] if '_' in key else key
      
      # 拼接特征和注意力
      x = jnp.concatenate([feats_dict[key], pred_attention_dict[key]], -1)
      # 使用对应前缀的网络组件
      x = self.get(f'{prefix}_img_out', Linear, **self._kw)(x)
      stats = self._stats(f'{prefix}_img_stats', x)
      
      # 采样和构建prior
      dist = self.get_dist(stats)
      stoch = dist.sample(seed=nj.rng())
      
      return {
          'stoch': stoch,
          'deter': deter_dict[key]['deter'],
          **stats
      }

    # different branch for ego, npc_1...npc_n, other_1...other_n state
    deter_dict = {}
    for key in prev_state_dict.keys():
      if key.startswith(('ego', 'npc_', 'other_')):
        prev_stoch = prev_state_dict[key]['stoch']
        deter = prev_state_dict[key]['deter']
        
        # reshape处理
        if self._classes:
          shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
          prev_stoch = prev_stoch.reshape(shape)
        if len(prev_action.shape) > len(prev_stoch.shape):
          shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
          prev_action = prev_action.reshape(shape)
        
        # GRU处理
        prefix = key.split('_')[0] if '_' in key else key
        x = jnp.concatenate([prev_stoch, prev_action], -1)
        x = self.get(f'{prefix}_img_in', Linear, **self._kw)(x)
        x, deter = self._gru(f'{prefix}_gru', x, deter)
        deter_dict[key] = {'out': x, 'deter': deter}

    # 计算attention
    feats_dict = {k: v['out'] for k,v in deter_dict.items()}
    feats_dict.update({
        'mask_npc': mask_npc,
        'mask_other': mask_other
    })
    pred_attention_dict, pred_attention_dict_post, _ = self._pred_attention(feats_dict)

    # 计算prior
    prior_dict = {}
    for key in feats_dict.keys():
        if key.startswith(('ego', 'npc_', 'other_')):
            prior = process_prior(self, key, feats_dict, pred_attention_dict, deter_dict)
            prior_dict[key] = prior

    return cast(prior_dict), cast(pred_attention_dict_post)

  # gets distribution from network outputs
  def get_dist(self, state, argmax=False):
    if self._classes: # 32
      logit = state['logit'].astype(f32) # Traced<ShapedArray(float32[16,16,16])>with<DynamicJaxprTrace(level=1/0)>
      return tfd.Independent(jaxutils.OneHotDist(logit), 1) # 根据logit创建一个OneHot分布,然后再包装为独立分布，独立分布的维度为1
    else:
      mean = state['mean'].astype(f32)
      std = state['std'].astype(f32)
      return tfp.MultivariateNormalDiag(mean, std)
  
  # # gets initial distribution from network outputs
  def _get_stoch(self, key, deter, attention=None, weight='shared'):
    if weight == 'shared':
      x = jnp.concatenate([deter, attention], -1) if attention is not None else deter
      x = self.get('init_stoch_layer', Linear, **self._kw)(x)
      stats = self._stats('init_stoch_stats', x)
      dist = self.get_dist(stats)
    elif weight == 'branch':
      if key.startswith('ego'):
        x = jnp.concatenate([deter, attention], -1) if attention is not None else deter
        x = self.get('ego_img_out', Linear, **self._kw)(x)
        stats = self._stats('ego_img_stats', x)
        dist = self.get_dist(stats)
      elif key.startswith('npc_'):
        x = jnp.concatenate([deter, attention], -1) if attention is not None else deter
        x = self.get('npc_img_out', Linear, **self._kw)(x)
        stats = self._stats('npc_img_stats', x)
        dist = self.get_dist(stats)
      elif key.startswith('other_'):
        x = jnp.concatenate([deter, attention], -1) if attention is not None else deter
        x = self.get('other_img_out', Linear, **self._kw)(x)
        stats = self._stats('other_img_stats', x)
        dist = self.get_dist(stats)
    return cast(dist.mode())
  
  def _initial_single_state(self, key, batch_size):
    # discrete or continuous latent space
    if self._classes: # 32
      state = dict(
          deter=jnp.zeros([batch_size, self._deter], f32), # 16 * 200
          logit=jnp.zeros([batch_size, self._stoch, self._classes], f32), # 16 * 32 * 32
          stoch=jnp.zeros([batch_size, self._stoch, self._classes], f32)) # 16 * 32 * 32
    else:
      state = dict(
          deter=jnp.zeros([batch_size, self._deter], f32),
          mean=jnp.zeros([batch_size, self._stoch], f32),
          std=jnp.ones([batch_size, self._stoch], f32),
          stoch=jnp.zeros([batch_size, self._stoch], f32))
    # weights initialization
    # NOTE: we found that initialize latent states with 'zeros' or 'shared'(trick?)
    # weights may helps stablize early training, but after all these 3 methods preform almost equaly at last
    if self._initial == 'zeros': # remain zeros
      state = cast(state)
    elif self._initial == 'learned': # use learned network to initialize
      deter = self.get('initial', jnp.zeros, state['deter'][0].shape, f32)
      state['deter'] = jnp.repeat(jnp.tanh(deter)[None], batch_size, 0)
      attention = self.get('initial_attention', jnp.zeros, (self._attention,), f32) # 
      attention = jnp.repeat(jnp.tanh(attention)[None], batch_size, 0) # 
      state['stoch'] = self._get_stoch(key, cast(state['deter']), cast(attention), weight='branch')
      state = cast(state)
    else:
      raise NotImplementedError(self._initial)
    return state
  
  # gru cell
  # input:  x: z(t-1), deter: h(t-1), 
  # output: h(t)
  def _gru(self, name, x, deter):     # inputs is MLP(z+a), state is h
    kw = {**self._kw, 'act': 'none', 'units': 3 * self._deter}
    # kw = {kw, 激活函数none, 神经网络单元数：3*1024 = 3072}
    x = jnp.concatenate([deter, x], -1) # concat MLP(z+a) and h
    # GRU contains 3 parts -- r, z, h(t-1)'(=cond now)
    
    x = self.get(name, Linear, **kw)(x)

    # GRU reset progress 
    # reset,cand,update相当于互相等价的三个元素，初始值相同
    reset, cand, update = jnp.split(x, 3, -1)  # reset:(batch_size, feature_dim) cand:(batch_size, feature_dim) update:(batch_size, feature_dim) 
    reset = jax.nn.sigmoid(reset) # 获得充值门的输出
    cand = jnp.tanh(reset * cand) # 获得候选隐状态值，充值门和h(t-1)相乘，再经过tanh激活函数
    update = jax.nn.sigmoid(update - 1) # 获得更新门的输出，更新门的输出减1后经过sigmoid激活函数

    # GRU update progress
    deter = update * cand + (1 - update) * deter # 获得ht输出
    return deter, deter

  # 输入: name,x。name是层的名称，x是输入的张量
  # 输出：对数变换后的logit概率分布
  def _stats(self, name, x):
    if self._classes: 
      # self._stoch * self._classes = 16 * 16 = 256.
      # x输入：(16,200)
      x = self.get(name, Linear, self._stoch * self._classes)(x)
      # x输出：(16,256)
      logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes)) # 把x重塑成(16,16,16),输出给shape
      if self._unimix: # mix of 1% uniform + 99% network output
        # 在logit的最后一个维度即self._classes维度上进行softmax归一化计算
        # probs输出对应的类别的概率
        probs = jax.nn.softmax(logit, -1) # 对logits最后一个维度进行softmax变换
        uniform = jnp.ones_like(probs) / probs.shape[-1] # 形状与probs相同的全1张量除以16 # uniform是用一个与probs形状相同的全1张量除以probs最后一个维度的大小，换句话说uniform的每个元素值都是1/16 
        probs = (1 - self._unimix) * probs + self._unimix * uniform # 用 (1-unimix) * net out + unimix * random 更新prob输出
        # 对数变换
        logit = jnp.log(probs)
      stats = {'logit': logit}
      return stats
    else:
      x = self.get(name, Linear, 2 * self._stoch)(x)
      mean, std = jnp.split(x, 2, -1)
      std = 2 * jax.nn.sigmoid(std / 2) + 0.1
      return {'mean': mean, 'std': std}

  # 通过 jnp.einsum 按批次维度对 value 和 mask 进行对应元素相乘，实现了高效的掩码操作。
  # 逐批次样本的掩码, 对于每个批次 b，value[b]将乘以mask[b]。
  # value，需要进行掩码操作的数据，形状(batch_size, ...)
  # mask，每个批次样本的掩码值，   形状(batch_size,)的一维张量
  def _mask(self, value, mask):
    return jnp.einsum('b...,b->b...', value, mask.astype(value.dtype))

  # 交换输入张量的前两个维度。如果输入是一个字典，则对字典中的每个值执行同样的操作。将批次维度和时间维度互换
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


class PIWMEncoder(nj.Module):
  
  # 根据输入数据的形状和类型，初始化编码器模块CNN或MLP
  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', mlp_layers=4,
      mlp_units=512, cnn='resize', cnn_depth=48,
      cnn_blocks=2, resize='stride',
      symlog_inputs=False, minres=4, **kw):
    
    excluded = ('is_first', 'is_last')
    
    # 初始形状过滤 
    shapes = {k: v for k, v in shapes.items() if (k not in excluded and not k.startswith('log_') and not k.endswith('_prediction'))}
    # 过滤条件：不在 excluded 列表中、不以 'log_' 开头、不以 '_prediction' 结尾

    # CNN形状提取，键值的长度为3，图像(height,width,depth)，且键名匹配CNN形状(正则表达式cnn_keys)
    self.cnn_shapes = {k: v for k, v in shapes.items() if (
        len(v) == 3 and re.match(cnn_keys, k))} # 注意，v的值是shapes.items()中值的元组，其含义是形状，那么v的长度为3，表示这个形状有三项，一般表示是图像数据
    
    # MLP形状提取，键值的长度为1或2，(向量或序列数据)，且键名匹配MLP形状(正则表达式mlp_keys)
    self.mlp_shapes = {k: v for k, v in shapes.items() if (
        len(v) in (1, 2) and re.match(mlp_keys, k))}
    
    # 合并二者形状
    self.shapes = {**self.cnn_shapes, **self.mlp_shapes} 
    print('Encoder CNN shapes:', self.cnn_shapes)
    print('Encoder MLP shapes:', self.mlp_shapes)

    # cnn layers, cnn shape = {} 
    # kw = {'act': 'silu', 'norm': 'layer', 'winit': 'normal', 'fan': 'avg'}
    cnn_kw = {**kw, 'minres': minres, 'name': 'cnn'} # cnn_kw = {'act': 'silu', 'norm': 'layer', 'winit': 'normal', 'fan': 'avg', 'minres': 4, 'name': 'cnn'}
    if self.cnn_shapes: # 如果有CNN输入
      if cnn == 'resnet':
        self._cnn = ImageEncoderResnet(cnn_depth, cnn_blocks, resize, **cnn_kw) # Resnet处理图像输入
      else:
        raise NotImplementedError(cnn)

    # mlp layers, 2 shared layers for trajectory and 2 layers for each branch
    if self.mlp_shapes: # 如果有MLP输入
      # vehicle info
      enc_mlp_layer = int(mlp_layers / 2) # 2

      # encode trajectory using the same mlp
      # 初始化共享轨迹编码器，提取轨迹特征，对应论文中的shared trajectory encoder，用于编码轨迹信息，由两层MLP组成
      # shape = None, layers = 2, units = 512(输出层特征维度), dist = 'none'(直接输出特征信息，不采取概率分布), symlog_inputs = False(不进行对数变换), name = 'traj_mlp'
      self._traj_mlp = MLP(None, enc_mlp_layer, mlp_units, dist='none', **kw, symlog_inputs=symlog_inputs, name='traj_mlp')
      # self._traj_mlp = MLP():实例化MLP类，同时初始化

      # for ego, npc and other features, using different mlp
      # 分别定义ego、npc和other的MLP编码器，对应论文中的ego、vpi和vdi，每个编码器由两层MLP组成
      self._ego_mlp = MLP(None, enc_mlp_layer, mlp_units, dist='none', **kw, symlog_inputs=False, name='ego_mlp')
      self._npc_mlp = MLP(None, enc_mlp_layer, mlp_units, dist='none', **kw, symlog_inputs=False, name='npc_mlp')
      self._other_mlp = MLP(None, enc_mlp_layer, mlp_units, dist='none', **kw, symlog_inputs=False, name='other_mlp')
      
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
      inputs = jnp.concatenate([data[k] for k in self.cnn_shapes], -1)# 将CNN数据在最后一个维度拼接，合并成一个输入
      output = self._cnn(inputs) # 处理后的数据输入到CNN  (self._cnn = ImageEncoderResnet(cnn_depth, cnn_blocks, resize, **cnn_kw))
      output = output.reshape((output.shape[0], -1)) # 将CNN输出展开成二维向量
      outputs_dict.update({'cnn': output}) # 更新输出字典

    # 3.处理(向量/地图)输入
    # for vector inputs (vehicle and map)
    if self.mlp_shapes:
      for key in self.mlp_shapes: # 遍历向量(mlp_shapes)输入
        ob = jaxutils.cast_to_compute(data[key].astype(f32)) # 对要处理的输入数据类型转换
        if key.startswith(('ego', 'npc_', 'other_')): # 筛选ego、npc和other的MLP编码器
          traj_features = self._traj_mlp(ob) # 获取共享轨迹编码器的输出特征
          traj_features = traj_features.reshape((traj_features.shape[0], -1)) #重塑形状 output shape: (batch_size, 512)
          if key.startswith('ego'):
            features = self._ego_mlp(traj_features)   # 在共享轨迹编码器之后，再分别输入到对应的MLP中，et_ego
          elif key.startswith('npc_'):
            features = self._npc_mlp(traj_features)   # 在共享轨迹编码器之后，再分别输入到对应的MLP中，et_vdi
          elif key.startswith('other_'):
            features = self._other_mlp(traj_features) # 在共享轨迹编码器之后，再分别输入到对应的MLP中，et_vpi
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

# 多头自注意力机制的实现，对应论文PIWM中的Self-Attention
class PredictAttention(nj.Module):
  # self-attention, produce ego and npc's attention value for future prediciton task
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
    self._units_per_head = units_per_head   # 
    self._inputs = Input(inputs, dims=dims)
    self._symlog_inputs = symlog_inputs
    # key words for dense layers and output(distribution) layers
    distkeys = (
        'dist', 'outscale', 'minstd', 'maxstd', 'outnorm', 'unimix', 'bins')
    self._dense = {k: v for k, v in kw.items() if k not in distkeys}
    self._dist = {k: v for k, v in kw.items() if k in distkeys}
    self.out_dim = self._heads * self._units_per_head # 输出维度：注意力头数量 × 每个头的特征维度
  
  # 处理ego与npc和other的特征
  def __call__(self, inputs_dict):
    # 1.输入预处理
    # preprocess inputs
    # feature_dict = {key: self._inputs(value) for key, value in inputs_dict.items() if isinstance(value, dict)}
    feature_dict = {key: value for key, value in inputs_dict.items() if key.startswith(('ego', 'npc_', 'other_'))} # 筛选保留inputs_dict中的以ego、npc_和other_开头的键值对
    if self._symlog_inputs: # 是否对数变换
      feature_dict = {key: jaxutils.symlog(value) for key, value in feature_dict.items()}
    
    # 2.特征拼接
    # use attention mechanism to fuse ego and npc features together
    npc_mask = inputs_dict['mask_npc']
    npc_num = npc_mask.shape[-1]
    other_mask = inputs_dict['mask_other']
    other_num = other_mask.shape[-1]
    # concat ego and npc features together in entity dimension
    # 将feature_dict中的所有value在倒数第2个i位置新插入一个新维度，比如，value原来的维度是2*3(2,3)的数组，新的就变成 2*1*3(2,1,3)
    feature_dict = {key: jnp.expand_dims(value, axis=-2) for key, value in feature_dict.items()} # add entity dimension

    # 拼接ego与npc和other车辆的特征
    # 特征拼接，ego与npc和other车辆的特征
    # vehicle_features_q = jnp.concatenate([feature_dict['ego']] + [feature_dict[f'npc_{i+1}'] for i in range(npc_num)], axis=-2)
    vehicle_features_q = jnp.concatenate([feature_dict['ego']] + [feature_dict[f'npc_{i+1}'] for i in range(npc_num)] + [feature_dict[f'other_{i+1}'] for i in range(other_num)], axis=-2)
    vehicle_features_kv = jnp.concatenate([feature_dict['ego']] + [feature_dict[f'npc_{i+1}'] for i in range(npc_num)] + [feature_dict[f'other_{i+1}'] for i in range(other_num)], axis=-2)
    # vehicle_features_q的shape: (16,11,200)
    # vehicle_features_kv的shape: (16,11,200)  
    
    # 3.特征变换
    q_x = jaxutils.cast_to_compute(vehicle_features_q)    # 数据类型转换 
    kv_x = jaxutils.cast_to_compute(vehicle_features_kv)  # 数据类型转换
    # attention
    # Dimensions: Batch*Length, entity, head, feature_per_head
    # q = self.get('query', Linear, units=self._heads*self._units_per_head, **self._dense)(q_x).reshape([-1, 1 + npc_num, self._heads, self._units_per_head])
    q = self.get('query', Linear, units=self._heads*self._units_per_head, **self._dense)(q_x).reshape([-1, 1 + npc_num + other_num, self._heads, self._units_per_head])
    # 获取名为'query'的Linear层，输入是q_x，输出维度为heads * units_per_head = 400(即units数)，然后将输出reshape成[-1, 11, 2, 200]的形状，赋值给q [-1,11,2,200]表示的就是[batch_size,实体数量,注意力头的数量、每个头的特征维度]
    k = self.get('key', Linear, units=self._heads*self._units_per_head, **self._dense)(kv_x).reshape([-1, 1 + npc_num + other_num, self._heads, self._units_per_head])
    # 获取名为'key'的Linear层，输入是kv_x，输出维度为heads * units_per_head = 400，然后将输出reshape成[-1, 11, 2, 200]的形状，赋值给k
    v = self.get('value', Linear, units=self._heads*self._units_per_head, **self._dense)(kv_x).reshape([-1, 1 + npc_num + other_num, self._heads, self._units_per_head])
    # 获取名为'value'的Linear层，输入是kv_x，输出维度为heads * units_per_head = 400，然后将输出reshape成[-1, 11, 2, 200]的形状，赋值给v
    # q,k,v的shape均为(16,11,2,200)

    # 4.维度变换
    # Dimensions: Batch*Length, head, entity, feature_per_head
    q = q.transpose(0,2,1,3) # 交换第2和第3个维度，即将实体数量和注意力头的数量交换，变成[batch_size,2,11,200]
    k = k.transpose(0,2,1,3) # 同上
    v = v.transpose(0,2,1,3) # 同上
    # print('q shape:', q.shape)
    # print('k shape:', k.shape)
    # print('v shape:', v.shape)
    # q,k,v的shape变为(16,2,11,200)

    # 5.掩码处理
    # mask Dimensions: Batch*Length, head, 1, entity
    ego_mask = jnp.ones(list(q.shape[:1]) + [1,1])    # 创建一个全为1的张量，形状为[batch_size,1,1]，作为ego的掩码。q.shape是一个包含q张量各个维度大小的元组(16,2,11,200) q.shape[:-1]表示从元组中获取第一个元素，结果是一个包含第一个元素的元组(16,) list(q.shape[:1])：[16]
    npc_mask = npc_mask.reshape(-1, 1, npc_num)       # 将npc_mask的形状reshape成[-1,1,npc_num]，-1表示自动计算该维度的大小，1表示在该维度上插入一个新维度，假设 npc_mask 的形状为 (32, 10)，则新的npc_mask的维度是(32, 1, npc_num)
    other_mask = other_mask.reshape(-1, 1, other_num) # 同上
    mask = jnp.concatenate([ego_mask, npc_mask, other_mask], axis=-1).reshape([-1, 1, 1, 1 + npc_num + other_num])
    # 将 ego_mask、npc_mask 和 other_mask 在最后一个维度（axis=-1）上进行拼接。
    # 然后重塑形状为[-1,1,1,11]
    # 注意reshape(-1，1，1)和reshape([-1,1,1])的区别
    mask = jnp.repeat(mask, self._heads, axis=1)
    
    # 多头注意力值计算
    # TODO: the attention of ego and npc can be get through 'different attention layers', 
    # since they do different tasks in latter parts(npc for future prediction only and ego for actor/critic/reward/count)
    # we use different mlp head to get a different attention result for ego and npc for now
    self_attention_out, self_attention_mat = multi_head_attention(q, k, v, mask, drop_out=False)
    # Dimensions(back to): Batch*Length, entity, head, feature_per_head
    self_attention_out = self_attention_out.transpose(0,2,1,3)
    self_attention_mat = self_attention_mat.transpose(0,2,1,3)
    
    # 输出处理
    self_attention_out_dict = {}
    self_attention_mat_dict = {}
    # for i in range(npc_num + 1): # ego and npc
    for i in range(npc_num + other_num + 1): # ego and npc
      if i == 0:
        # TODO: ego maybe not useful
        # attention matrix for ego
        self_attention_mat_dict['ego'] = self_attention_mat[..., 0, :, :]
        # attention output for ego
        x = self_attention_out[..., 0, :, :]
        x = x.reshape([x.shape[0], -1])
        x = self.get('ego_out_mlp', Linear, units=self._heads*self._units_per_head, **self._dense)(x)
        self_attention_out_dict['ego'] = x.reshape(list(vehicle_features_q.shape[:-2]) + [-1])
      elif i < npc_num + 1:
        # attention matrix for npc
        self_attention_mat_dict[f'npc_{i}'] = self_attention_mat[..., i, :, :]
        # attention output for npc
        x = self_attention_out[..., i, :, :]
        x = x.reshape([x.shape[0], -1])
        x = self.get('npc_out_mlp', Linear, units=self._heads*self._units_per_head, **self._dense)(x)
        self_attention_out_dict[f'npc_{i}'] = x.reshape(list(vehicle_features_q.shape[:-2]) + [-1])
      else:
        # attention matrix for other
        self_attention_mat_dict[f'other_{i-npc_num}'] = self_attention_mat[..., i, :, :]
        # attention output for other
        x = self_attention_out[..., i, :, :]
        x = x.reshape([x.shape[0], -1])
        x = self.get('other_out_mlp', Linear, units=self._heads*self._units_per_head, **self._dense)(x)
        self_attention_out_dict[f'other_{i-npc_num}'] = x.reshape(list(vehicle_features_q.shape[:-2]) + [-1])
    return self_attention_out_dict, self_attention_out_dict, self_attention_mat_dict
  

class EgoAttention(nj.Module):
  # cross-attention or ego-attention, produce attention value for ego's task of actor/critic/predicting reward/predicting count
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
    self._dist = {k: v for k, v in kw.items() if k in distkeys}

  def __call__(self, inputs_dict):
    # preprocess inputs
    feature_dict = {key: self._inputs(value) for key, value in inputs_dict.items() if isinstance(value, dict) and key != 'attention_npc'}
    # npc(and ego) feature contains its individual feature and its self-attention (prediction attention) to surrounding vehicles, as interactive features
    # for key, value in feature_dict.items():
    #   if key.startswith('npc_') or key.startswith('ego'):
    #     # print(key, value.shape, inputs_dict['attention_npc'][key].shape)
    #     feature_dict[key] = jnp.concatenate([value, inputs_dict['attention_npc'][key]], axis=-1)
        # print(feature_dict[key].shape)
    if self._symlog_inputs:
      feature_dict = {key: jaxutils.symlog(value) for key, value in feature_dict.items()}
    # use attention mechanism to fuse ego and npc features together
    npc_mask = inputs_dict['mask_npc']
    npc_num = npc_mask.shape[-1]
    # concat ego and npc features together in entity dimension
    feature_dict = {key: jnp.expand_dims(value, axis=-2) for key, value in feature_dict.items()}
    ego_features = feature_dict['ego']
    ego_features = jaxutils.cast_to_compute(ego_features)
    vehicle_features = jnp.concatenate([feature_dict['ego']] + [feature_dict[f'npc_{i+1}'] for i in range(npc_num)], axis=-2)
    vehicle_features = jaxutils.cast_to_compute(vehicle_features)
    # attention
    # Dimensions: Batch*Length, entity, head, feature_per_head
    q_ego = self.get('query', Linear, units=self._heads*self._units_per_head, **self._dense)(ego_features).reshape([-1, 1, self._heads, self._units_per_head])
    k_all = self.get('key', Linear, units=self._heads*self._units_per_head, **self._dense)(vehicle_features).reshape([-1, 1 + npc_num, self._heads, self._units_per_head])
    v_all = self.get('value', Linear, units=self._heads*self._units_per_head, **self._dense)(vehicle_features).reshape([-1, 1 + npc_num, self._heads, self._units_per_head])
    # Dimensions: Batch*Length, head, entity, feature_per_head
    q_ego = q_ego.transpose(0,2,1,3)
    k_all = k_all.transpose(0,2,1,3)
    v_all = v_all.transpose(0,2,1,3)
    # print('q shape:', q.shape)
    # print('k shape:', k.shape)
    # print('v shape:', v.shape)
    # mask Dimensions: Batch*Length, head, 1, entity
    ego_mask = jnp.ones(list(q_ego.shape[:1]) + [1,1]) # Batch*Length, 1, 1
    npc_mask = npc_mask.reshape(-1, 1, npc_num)
    mask = jnp.concatenate([ego_mask, npc_mask], axis=-1).reshape([-1, 1, 1, npc_num + 1])
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


class PredictionDecoder(nj.Module):

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
        if re.match(mlp_keys, k) and len(v) in (1, 2) and k.endswith('_prediction')}
    
    self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
    print('Decoder CNN shapes:', self.cnn_shapes)
    print('Decoder MLP shapes:', self.mlp_shapes)

    # inputs preprocess
    self._inputs = Input(inputs, dims='deter') # inputs: ['deter', 'stoch'], dims: 'deter'

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
      # vehicle future trajectory prediction
      self._ego_mlp = MLP(self.mlp_shapes['ego_prediction'], mlp_layers, mlp_units, **mlp_kw, name='ego_mlp')
      self._npc_mlp = MLP(self.mlp_shapes['npc_1_prediction'], mlp_layers, mlp_units, **mlp_kw, name='npc_mlp')

  def __call__(self, featrue_dict, drop_loss_indices=None):
    dists_dict = {}
    # decode image
    if self.cnn_shapes:
      feature = self._inputs(featrue_dict)
      if drop_loss_indices is not None:
        feature = feature[:, drop_loss_indices]
      flat = feature.reshape([-1, feature.shape[-1]])
      output = self._cnn(flat)
      output = output.reshape(feature.shape[:-1] + output.shape[1:])
      split_indices = np.cumsum([v[-1] for v in self.cnn_shapes.values()][:-1])
      means = jnp.split(output, split_indices, -1)
      dists_dict.update({
          key: self._make_image_dist(key, mean)
          for (key, shape), mean in zip(self.cnn_shapes.items(), means)})
      
    # decode vector(future trajectory prediction)
    if self.mlp_shapes:
      for key in self.mlp_shapes:
        # input feature is vehicle feature concats its attention
        x = self._inputs(featrue_dict[key.replace('_prediction', '')]) # consider key may contains 'prediction' according to the training target
        if key.startswith('ego'):
          dist = self._ego_mlp(x)
        elif key.startswith('npc_'):
          dist = self._npc_mlp(x)
        dists_dict.update({key: dist})
    return dists_dict

  def _make_image_dist(self, name, mean):
    mean = mean.astype(f32)
    if self._image_dist == 'normal':
      return tfd.Independent(tfd.Normal(mean, 1), 3)
    if self._image_dist == 'mse':
      return jaxutils.MSEDist(mean, 3, 'sum')
    raise NotImplementedError(self._image_dist)
    

class PIWMMLP(nj.Module):
  
  # 初始化MLP参数
  # for modules like actor, critic, reward prediction, countinue prediction, we use ego's feature and ego-attention as input
  def __init__(
      self, shape, layers, units, inputs=['tensor'], dims=None, 
      symlog_inputs=False, **kw):
    # **kw 其他关键字参数

    # data shape type transition
    assert shape is None or isinstance(shape, (int, tuple, dict)), shape
    if isinstance(shape, int):
      shape = (shape,) # 将整数类型shape转换为元组

    # network hyparameters
    self._shape = shape   # 输出数据的形状，可以是整数、元组或字典
    self._layers = layers # MLP的层数
    self._units = units   # 每层的神经元数量
    self._inputs = Input(inputs, dims=dims) # 输入类型，默认为['tensor']
    self._symlog_inputs = symlog_inputs     # 是否对输入进行对称对数变换 False
    # key words for dense layers and output(distribution) layers
    distkeys = (
        'dist', 'outscale', 'minstd', 'maxstd', 'outnorm', 'unimix', 'bins')
    # 全连接层参数：
    self._dense = {k: v for k, v in kw.items() if k not in distkeys}
    # 输出层参数：
    self._dist = {k: v for k, v in kw.items() if k in distkeys}

  # 多层感知机的前向传播
  def __call__(self, feature_dict, attention, concat_feat=False): # feature_dict: {'ego': ego_features, 'npc_1': npc_1_features, ...}, attention_dict: {'ego': ego_attention_to_other, 'npc_1': npc_1_attention_to_other, ...}
    # 1.数据预处理
    # preprocess feature inputs, only consider vehicle features
    feature_dict = {key: self._inputs(value) for key, value in feature_dict.items() if key.startswith(('ego', 'npc_'))}
    # 筛选feature_dict中以ego和npc_开头的键值对，将其值转换为张量，返回处理后的新字典。
    print(feature_dict)
    if self._symlog_inputs: # 判断是否进行对数变换
      feature_dict = {key: jaxutils.symlog(value) for key, value in feature_dict.items() if key.startswith(('ego', 'npc_'))}

    # 2.特征连接
    # TODO: concat npc features or ego attention should be a hyparameter, yet concat should in distance order
    if concat_feat: # concat ego and npc features together  # 将ego和npc特征连接在一起
      feat = jnp.concatenate(list(feature_dict.values()), axis=-1)
    else: # use ego feature and its attention               # 不将ego和npc特征连接在一起
      if isinstance(attention, dict):
        attention = attention['ego'] # 提取attention列表中'ego'的注意力值(并不是一个值)
      feat = jnp.concatenate([feature_dict['ego'], attention], axis=-1) # 拼接ego特征和ego的注意力值，沿最后一个维度
    x = jaxutils.cast_to_compute(feat) # 对feat特征进行类型转换

    # 3.多层感知机的前向传播,特征提取，全连接层
    # after concat, pass the features through mlp layer
    x = x.reshape([-1, x.shape[-1]])# 展平输入的形状：x.shape[-1]表示保持最后一个维度的值不变，-1表示自动计算其他维度的数值。比如：原始输入是(32,10,256)，那么x = x.reshape([-1,256])的输出是(320,256)
    for i in range(self._layers):# 遍历MLP每一层
      x = self.get(f'h{i}', Linear, self._units, **self._dense)(x)
      # f'h{i}' 是该层的名称，‘h{0},h{1}’
      # Linear  是该层的类型，(线性层)
      # self._units是该层的输出维度(神经元数量)
      # **self._dense：解引用全连接层的参数，比如激活函数
      # self.get()函数首先判断名为'h{i}'的线性层是否已经存在，如果不存在则创建一个新的Linear层，参数为self._units和self._dense。
      # 其返回值为一个配置好的线性层对象，这个对象包含:权重矩阵、偏置、激活函数等参数。
      # 然后用这样一个线性层对象处理x的输入，进行前向计算得到输出x。
    x = x.reshape(feat.shape[:-1] + (x.shape[-1],))
    # feat.shape[:-1] 表示保持原始输入feat的所有维度除了最后一个维度，
    # x.shape[-1] 表示经过MLP处理后的特征维度(神经元数量)，x为输入经过所有线性层的输出，而输出的最后一个维度是self._units。 (线性层输出的最后一个维度是self._units，所以x.shape[-1] = self._units）
    # 最终这句话使得e多层线性变换后的形状没变，只有最后一个维度的值改变了。

    # 4.多层感知机的前向传播，输出层
    # output mlp layer, different kinds of outputs
    if self._shape is None:
      return x # 直接返回特征
    elif isinstance(self._shape, tuple):
      return self._out('out', self._shape, x) # 根据self._dist参数生成单个概率分布输出 #调用_out()函数，‘out’赋给name，self._shape赋给shape，x赋给x。进一步调用self.get，原理同上
    elif isinstance(self._shape, dict):
      return {k: self._out(k, v, x) for k, v in self._shape.items()} # 生成多个命名的概率分布输出
    else:
      raise ValueError(self._shape)

  def _out(self, name, shape, x):
    return self.get(f'dist_{name}', Dist, shape, **self._dist)(x)  # 将x转换为不同的概率分布


class MLP(nj.Module):
  # basic mlp layers used in encoder and decoder
  def __init__(
      self, shape, layers, units, inputs=['tensor'], dims=None,
      symlog_inputs=False, **kw):
    # data shape type transition
    assert shape is None or isinstance(shape, (int, tuple, dict)), shape # 断言语句，判断shape是否为None或者int或tuple或dict之一，不是就报错
    if isinstance(shape, int):
      shape = (shape,)
    # network hyparameters
    self._shape = shape
    self._layers = layers
    self._units = units
    self._inputs = Input(inputs, dims=dims)
    self._symlog_inputs = symlog_inputs
    # key words for dense layers and output(distribution) layers
    distkeys = (
        'dist', 'outscale', 'minstd', 'maxstd', 'outnorm', 'unimix', 'bins')
    self._dense = {k: v for k, v in kw.items() if k not in distkeys}
    self._dist = {k: v for k, v in kw.items() if k in distkeys}

  def __call__(self, inputs):
    # preprocess inputs
    feat = self._inputs(inputs)
    if self._symlog_inputs:
      feat = jaxutils.symlog(feat)
    x = jaxutils.cast_to_compute(feat)

    # make it flatten
    x = x.reshape([-1, x.shape[-1]])
    for i in range(self._layers):
      x = self.get(f'h{i}', Linear, self._units, **self._dense)(x)
    x = x.reshape(feat.shape[:-1] + (x.shape[-1],))

    # different kinds of outputs according to self._shape's type
    if self._shape is None:
      return x
    elif isinstance(self._shape, tuple):
      return self._out('out', self._shape, x)
    elif isinstance(self._shape, dict):
      return {k: self._out(k, v, x) for k, v in self._shape.items()}
    else:
      raise ValueError(self._shape)

  def _out(self, name, shape, x):
    return self.get(f'dist_{name}', Dist, shape, **self._dist)(x)


class ImageEncoderResnet(nj.Module):

  def __init__(self, depth, blocks, resize, minres, **kw):
    self._depth = depth
    self._blocks = blocks
    self._resize = resize
    self._minres = minres
    self._kw = kw

  def __call__(self, x):
    stages = int(np.log2(x.shape[-2]) - np.log2(self._minres))
    depth = self._depth
    x = jaxutils.cast_to_compute(x) - 0.5
    # print(x.shape)
    for i in range(stages):
      kw = {**self._kw, 'preact': False}
      if self._resize == 'stride':
        x = self.get(f's{i}res', Conv2D, depth, 4, 2, **kw)(x)
      elif self._resize == 'stride3':
        s = 2 if i else 3
        k = 5 if i else 4
        x = self.get(f's{i}res', Conv2D, depth, k, s, **kw)(x)
      elif self._resize == 'mean':
        N, H, W, D = x.shape
        x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
        x = x.reshape((N, H // 2, W // 2, 4, D)).mean(-2)
      elif self._resize == 'max':
        x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
        x = jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), 'same')
      else:
        raise NotImplementedError(self._resize)
      for j in range(self._blocks):
        skip = x
        kw = {**self._kw, 'preact': True}
        x = self.get(f's{i}b{j}conv1', Conv2D, depth, 3, **kw)(x)
        x = self.get(f's{i}b{j}conv2', Conv2D, depth, 3, **kw)(x)
        x += skip
        # print(x.shape)
      depth *= 2
    if self._blocks:
      x = get_act(self._kw['act'])(x)
    x = x.reshape((x.shape[0], -1))
    # print(x.shape)
    return x


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
    # discrete output value shape for critic & reward head
    if self._dist.endswith('_disc'):
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