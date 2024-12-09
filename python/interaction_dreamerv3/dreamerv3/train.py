import importlib
import pathlib
import sys
import warnings
from functools import partial as bind
import yaml
import os



warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import embodied
from embodied import wrappers



def main(argv=None):
  from . import agent as agt

  with open('configs.yaml', 'r') as file:
    config = yaml.safe_load(file)
  print("config1:",config) 

  # 从configs.yaml文件中获取npc_num和other_num的值
  npc_num = config['interaction_prediction']['env']['interaction']['npc_num']
  other_num = config['interaction_prediction']['env']['interaction']['other_num']

  parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
  config = embodied.Config(agt.Agent.configs['defaults'])
  for name in parsed.configs:
    config = config.update(agt.Agent.configs[name])
  config = embodied.Flags(config).parse(other)
  # args contains items in 'run', logdir, and batch size info
  args = embodied.Config(
      **config.run, logdir=config.logdir, 
      batch_steps=config.batch_size * config.batch_length)
  
  print("config2:",config) 
  print("npc_num",npc_num)
  print("other_num",other_num)

  # 生成encoder的mlp_keys列表
  encoder_mlp_keys = ['ego'] + \
      [f'npc_{i}' for i in range(1, npc_num + 1)] + \
      [f'other_{i}' for i in range(1, other_num + 1)]
  # 生成decoder的mlp_keys列表
  decoder_mlp_keys = ['ego_prediction'] + \
      [f'npc_{i}_prediction' for i in range(1, npc_num + 1)]
  # 更新到config中
  config = config.update({
    'encoder': {'mlp_keys': encoder_mlp_keys},
    'decoder': {'mlp_keys': decoder_mlp_keys},
  })

  print("config3:",config) 
 
  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  config.save(logdir / 'config.yaml')
  step = embodied.Counter()
  logger = make_logger(parsed, logdir, step, config)

  cleanup = []
  try:
    if args.script == 'train' and config.task.startswith('interaction'):
      # change replay from v3 version (save as chunks) to v2 version (save as episodes, which can be used to make prediction data)
      if args.replay_style == 'chunk':
        replay = make_replay(config, logdir / 'replay') # 初始化replay_buffer
      elif args.replay_style == 'ep':
        replay = make_replay_ep(config, logdir / 'replay') # ep方式初始化replay_buffer

      env = make_envs(config) ## 返回的env为：BatchEnv(len=1, obs_space={'ego': Space(dtype=float64, shape=(19, 5), low=-inf, high=inf), 'ego_prediction': Space(dtype=float64, shape=(20, 2), low=-inf, high=inf), 'npc_1': Space(dtype=float64, shape=(19, 5), low=-inf, high=inf), 'npc_1_prediction': Space(dtype=float64, shape=(20, 2), low=-inf, high=inf), 'npc_2': Space(dtype=float64, shape=(19, 5), low=-inf, high=inf), 'npc_2_prediction': Space(dtype=float64, shape=(20, 2), low=-inf, high=inf), 'npc_3': Space(dtype=float64, shape=(19, 5), low=-inf, high=inf), 'npc_3_prediction': Space(dtype=float64, shape=(20, 2), low=-inf, high=inf), 'npc_4': Space(dtype=float64, shape=(19, 5), low=-inf, high=inf), 'npc_4_prediction': Space(dtype=float64, shape=(20, 2), low=-inf, high=inf), 'npc_5': Space(dtype=float64, shape=(19, 5), low=-inf, high=inf), 'npc_5_prediction': Space(dtype=float64, shape=(20, 2), low=-inf, high=inf), 'id_npc': Space(dtype=int32, shape=(5,), low=-2147483648, high=2147483647), 'mask_npc': Space(dtype=int32, shape=(5,), low=-2147483648, high=2147483647), 'should_init_npc': Space(dtype=int32, shape=(5,), low=-2147483648, high=2147483647), 'other_1': Space(dtype=float64, shape=(19, 5), low=-inf, high=inf), 'other_2': Space(dtype=float64, shape=(19, 5), low=-inf, high=inf), 'other_3': Space(dtype=float64, shape=(19, 5), low=-inf, high=inf), 'other_4': Space(dtype=float64, shape=(19, 5), low=-inf, high=inf), 'other_5': Space(dtype=float64, shape=(19, 5), low=-inf, high=inf), 'mask_other': Space(dtype=int32, shape=(5,), low=-2147483648, high=2147483647), 'should_init_other': Space(dtype=int32, shape=(5,), low=-2147483648, high=2147483647), 'reward': Space(dtype=float32, shape=(), low=-inf, high=inf), 'is_first': Space(dtype=bool, shape=(), low=False, high=True), 'is_last': Space(dtype=bool, shape=(), low=False, high=True), 'is_terminal': Space(dtype=bool, shape=(), low=False, high=True), 'sta_speed': Space(dtype=float32, shape=(1,), low=-inf, high=inf), 'sta_collision': Space(dtype=int32, shape=(), low=-2147483648, high=2147483647), 'sta_success': Space(dtype=int32, shape=(), low=-2147483648, high=2147483647), 'sta_complet': Space(dtype=float32, shape=(1,), low=-inf, high=inf), 'sta_gt_distance': Space(dtype=float32, shape=(1,), low=-inf, high=inf)}, act_space={'action': Space(dtype=float32, shape=(4,), low=0, high=1)})
      cleanup.append(env) ## 将env加入cleanup列表
      agent = agt.Agent(env.obs_space, env.act_space, step, config) # 设置agent
      embodied.run.train(agent, env, replay, logger, args) # 开始训练

    elif args.script == 'eval_only' and config.task.startswith('interaction'):
      # env is set to eval mode, every vehilce has the same oppotunity to be selected
      config = config.update({'env':{'interaction':{'eval': True}}})
      
      env = make_envs(config)  # mode='eval'
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.eval_only(agent, env, logger, args, config.task)

    elif args.script == 'train':
      replay = make_replay(config, logdir / 'replay')
      env = make_envs(config)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train(agent, env, replay, logger, args)

    elif args.script == 'train_save':
      replay = make_replay(config, logdir / 'replay')
      env = make_envs(config)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_save(agent, env, replay, logger, args)

    elif args.script == 'train_eval':
      replay = make_replay(config, logdir / 'replay')
      eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
      env = make_envs(config)
      eval_env = make_envs(config)  # mode='eval'
      cleanup += [env, eval_env]
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_eval(
          agent, env, eval_env, replay, eval_replay, logger, args)

    elif args.script == 'train_holdout':
      replay = make_replay(config, logdir / 'replay')
      if config.eval_dir:
        assert not config.train.eval_fill
        eval_replay = make_replay(config, config.eval_dir, is_eval=True)
      else:
        assert 0 < args.eval_fill <= config.replay_size // 10, args.eval_fill
        eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
      env = make_envs(config)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_holdout(
          agent, env, replay, eval_replay, logger, args)

    elif args.script == 'eval_only':
      env = make_envs(config)  # mode='eval'
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.eval_only(agent, env, logger, args)

    elif args.script == 'parallel':
      assert config.run.actor_batch <= config.envs.amount, (
          config.run.actor_batch, config.envs.amount)
      step = embodied.Counter()
      env = make_env(config)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      env.close()
      replay = make_replay(config, logdir / 'replay', rate_limit=True)
      embodied.run.parallel(
          agent, replay, logger, bind(make_env, config),
          num_envs=config.envs.amount, args=args)

    else:
      raise NotImplementedError(args.script)
  finally:
    for obj in cleanup:
      obj.close()


def make_logger(parsed, logdir, step, config):
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  # print('Create logger...')
  log_list = [
      embodied.logger.TerminalOutput(config.filter),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ]
  if config.task.startswith('interaction'):
    log_list += [
      embodied.logger.JSONLOutput(logdir, 'avg_speed.jsonl', 'episode/avg_speed'),
      embodied.logger.JSONLOutput(logdir, 'completion.jsonl', 'episode/completion'),
      embodied.logger.JSONLOutput(logdir, 'collision_ticks.jsonl', 'episode/collision_ticks'),
    ]
  logger = embodied.Logger(step, log_list, multiplier)
  return logger


def make_replay(
    config, directory=None, is_eval=False, rate_limit=False, **kwargs):
  assert config.replay == 'uniform' or not rate_limit
  # set replay size
  length = config.batch_length # 64 by default
  size = config.replay_size // 10 if is_eval else config.replay_size # 1e6/2e6 or 1e5/2e5 by default
  if config.replay == 'uniform' or is_eval: # 'uniform' replay by default
    kw = {'online': config.replay_online} # replay_online = False by default
    if rate_limit and config.run.train_ratio > 0: # rate_limit is False by default
      kw['samples_per_insert'] = config.run.train_ratio / config.batch_length # 32 / 64 by default
      kw['tolerance'] = 10 * config.batch_size # 10 * 16
      kw['min_size'] = config.batch_size # 16 by default
    replay = embodied.replay.Uniform(length, size, directory, **kw)
  elif config.replay == 'reverb':
    replay = embodied.replay.Reverb(length, size, directory)
  elif config.replay == 'chunks':
    replay = embodied.replay.NaiveChunks(length, size, directory)
  else:
    raise NotImplementedError(config.replay)
  return replay


def make_replay_ep(
    config, directory=None, is_eval=False, rate_limit=False):
  
  assert config.replay == 'uniform' or not rate_limit #初始条件下config.replay为uniform，rate_limit为False
  # set replay size
  length = config.batch_length # 50 by default
  size = config.replay_size // 10 if is_eval else config.replay_size # 默认 size = config.replay_size = 2000000 # 1e6/2e6 or 1e5/2e5 by default
  
  npc_num = config.env.interaction.npc_num # 5 by default 5个vdi
  other_num = config.env.interaction.other_num # 5 by default 5个vpi
  predict_horizen = config.env.interaction.predict_horizen # 20 means 2s in 10hz for prediction task

  replay = embodied.replay.ReplayEp(directory, capacity=size, batch_size=config.batch_size, batch_length=length, npc_num=npc_num, other_num=other_num, predict_horizen=predict_horizen,
                                    ongoing=False, minlen=config.batch_length, maxlen=config.batch_length, prioritize_ends=True)

  return replay


def make_envs(config, **overrides):
  print("Overrides:", overrides)
  suite, task = config.task.split('_', 1) # suite = interaction, task = prediction
  ctors = [] # 存储构造函数的列表
  for index in range(config.envs.amount): # config.envs.amount = 1 by default
    ctor = lambda: make_env(config, **overrides) # 等价于直接执行make_env(config, **overrides)，返回一个env赋值给ctor
    if config.envs.parallel != 'none': # parallel = none by default
      ctor = bind(embodied.Parallel, ctor, config.envs.parallel) # 用embodied.Parallel包装ctor
    if config.envs.restart: # config.envs.restart = True by default
      ctor = bind(wrappers.RestartOnException, ctor) # 用wrappers.RestartOnException包装ctor 如果 ctor 在执行过程中抛出异常，wrappers.RestartOnException 会捕获该异常并重新调用 ctor
    ctors.append(ctor) # 将包装好的ctor函数加入ctors列表的末尾
  envs = [ctor() for ctor in ctors] # 用ctors列表中的每个ctor函数创建一个env，遍历ctors中的每个ctor函数，将每个ctor函数的结果/返回值存在envs列表中
  return embodied.BatchEnv(suite, envs, parallel=(config.envs.parallel != 'none')) # 返回一个BatchEnv对象，该对象包含了envs列表中的所有env


def make_env(config, **overrides):
  # You can add custom environments by creating and returning the environment
  # instance here. Environments with different interfaces can be converted
  # using `embodied.envs.from_gym.FromGym` and `embodied.envs.from_dm.FromDM`.
  suite, task = config.task.split('_', 1) # suite = interaction, task = prediction
  ctor = {
      'dummy': 'embodied.envs.dummy:Dummy',
      'gym': 'embodied.envs.from_gym:FromGym',
      'dm': 'embodied.envs.from_dmenv:FromDM',
      'crafter': 'embodied.envs.crafter:Crafter',
      'dmc': 'embodied.envs.dmc:DMC',
      'atari': 'embodied.envs.atari:Atari',
      'dmlab': 'embodied.envs.dmlab:DMLab',
      'minecraft': 'embodied.envs.minecraft:Minecraft',
      'loconav': 'embodied.envs.loconav:LocoNav',
      'pinpad': 'embodied.envs.pinpad:PinPad',
      'interaction': 'embodied.envs.interaction:Interaction',
  }[suite] # 根据 suite 变量的值获取对应的构造函数路径 ctor = embodied.envs.interaction:Interaction 
           # 即ctor最终为embodied.envs.interaction模块中的Interaction类
  if isinstance(ctor, str): # 如果ctor是一个字符串
    module, cls = ctor.split(':') # module = embodied.envs.interaction, cls = Interaction 将ctor字符串按照':'分割，分别赋值给module和cls 
    module = importlib.import_module(module) #导入embodied.envs.interaction模块，赋值给module，即module = embodied.envs.interaction
    ctor = getattr(module, cls) # 获取module模块中的cls类，赋值给ctor，即ctor = embodied.envs.interaction.Interaction类 
  kwargs = config.env.get(suite, {}) # 从 config.env 字典中获取与 suite 相关的配置参数，并将其赋值给 kwargs 变量。如果 suite 不在 config.env 中，则返回一个空字典。 kwargs = config.env['interaction'] = {}
  # print(kwargs)
  kwargs.update(overrides) # 将overrides字典中的键值对更新到kwargs字典中
  env = ctor(task, kwargs) if suite == 'interaction' else ctor(task, **kwargs) # env = embodied.envs.interaction.Interaction(task, kwargs)   # task = prediction, kwargs = overrides = config = {config.yaml}
  return wrap_env(env, config) 

#
def wrap_env(env, config):
  args = config.wrapper # wrapper: {checks: false, discretize: 0, length: 0, reset: true} by default
  for name, space in env.act_space.items(): # env.act_space = {'acion': Space(dtype=int32, shape=(), low=0, high=4)} by default
  #遍历env.act_space字典的所有项，name为键，space为值
    if name == 'reset':
      continue
    elif space.discrete: # space.discrete = True by default
      env = wrappers.OneHotAction(env, name) # env就为传入的env参数，name = 'action'，即env = wrappers.OneHotAction(env, 'action')
    elif args.discretize: # args.discretize = 0 by default
      env = wrappers.DiscretizeAction(env, name, args.discretize)
    else:
      env = wrappers.NormalizeAction(env, name)
  env = wrappers.ExpandScalars(env) # 
  if args.length: # args.length = 0 by default
    env = wrappers.TimeLimit(env, args.length, args.reset)
  if args.checks: # args.checks = False by default
    env = wrappers.CheckSpaces(env)
  for name, space in env.act_space.items(): # env.act_space = {'acion': Space(dtype=int32, shape=(), low=0, high=4)} by default
    if not space.discrete: # space.discrete = True by default
      env = wrappers.ClipAction(env, name)
  return env


if __name__ == '__main__':
  main()
