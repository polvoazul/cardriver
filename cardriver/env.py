import math
import random
from gym import spaces, Env
import numpy as np
import pandas as pd

MAX_SPEED = 10.
MAX_ACCEL = 5.
MAX_BRAKE = MAX_ACCEL
INIT_DISTANCE = 100.

from ray.rllib.env.env_context import EnvContext

class SimpleRoad(Env):
    '''
    This env has observation_space: tuple(floats: [Distance, speed, time], light_color)
    '''
    metadata = {'render_modes': ['ascii'], "render_fps": 5}

    def __init__(self, config: EnvContext = {}):
        # distance, speed and time
        self.observation_space = spaces.Tuple((
            spaces.Box(
              low=np.array([0, 0, 0], dtype=np.float32),
              high=np.array([INIT_DISTANCE, MAX_SPEED, 55], dtype=np.float32),
              dtype=np.float32,
            ),
            spaces.Discrete(3) # light states Yellow Red Green
        ))
        self.action_space = spaces.Box(
            low=np.array([-MAX_BRAKE], dtype=np.float32),
            high=np.array([MAX_ACCEL], dtype=np.float32),
            dtype=np.float32
        )
        self.light = SimpleLight()
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.render_mode = config.get('render_mode')

        self.reset()

    def step(self, action):
        '''Accel for 1s'''
        accel = action[0]
        if accel + self.speed > MAX_SPEED: accel = MAX_SPEED - self.speed # accel cant go over max speed
        if accel + self.speed < 0: accel = 0 - self.speed # accel cant go less than 0
        self.last_accel = accel
        self.distance -= self.speed + (accel / 2)
        self.speed += accel
        self.time += 1
        reward = ( 
            -abs(accel) +  # cost of gas
            -1 # cost of time
           # self.speed + (accel / 2) # distance covered --- d = a*t^2/2 + s*t = a/2 + s
        )
        if self.time > 50: # Give up
          reward = -1000 
          done = True
        elif self.distance <= 0:
          # Suppose we have a target that is D away from the light.
          # Suppose we can freely accel after the light. So our reward should be based on the time
          # to get there.
          # from kinematic equation: d = a*t^2/2 + s*t
          # -> solving for t: t = (-s+sqrt(s*s+2ad))/a
          # our reward is penalized by time.
          reward -= (-self.speed+math.sqrt((self.speed*self.speed)+2*MAX_ACCEL*INIT_DISTANCE))/MAX_ACCEL
          # simple reward: reward += self.speed - self.time
          if self.light.light_at_time(self.time) == 1: # Red
            reward -= 1000 # Running through the Red light is expensive 
          done = True
          self.distance = 0 # Truncate to min
        else:
          done = False
        info = self._get_info()
        return self._get_obs(), reward, done, info

    def reset(self, seed=0):
        super().reset(seed=seed)
        self.time = 0
        self.speed = self.np_random.uniform(0, MAX_SPEED)
        self.distance = INIT_DISTANCE
        self.last_accel = 0
        self.light.reset()
        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs()

    def _get_obs(self):
        return (np.array([self.distance, self.speed, self.time], np.float32),
                self.light.light_at_time(self.time))

    def _get_info(self):
        return {}

    def render(self, mode='ascii'):
      if mode != 'ascii': return
      ROAD = int(INIT_DISTANCE) * '_'
      L = {0:'Y', 1:'R', 2:'G'}[self.light.light_at_time(self.time)]
      position = int(INIT_DISTANCE - self.distance)
      if self.distance == 0:
        print(ROAD + '_' + L + 'C')
      else:
        print(ROAD[:position] + 'C' + ROAD[position:] + L + '_')
      print(f'Speed: {self.speed:.1f} / Accel: {self.last_accel:.1f} / Time: {self.time}s')
      return True
      return np.array([
          [[10,20,30], [10,20,30], [10,20,30]],
          [[10,200,30], [10,200,30], [10,200,30]],
          [[10,20,30], [10,20,30], [10,20,30]],
      ], dtype='uint8')

    def close(self):
        if self.window is not None:
            pass #pygame.display.quit()
            #pygame.quit()

class SimpleLight:
  Yellow = 3
  Red = 10
  Green = 10
  Total = Yellow + Red + Green

  def __init__(self):
    self.reset()

  def reset(self):
    self.init_time = random.randint(0, self.Total - 1)

  def light_at_time(self, t):
    t = (self.init_time + t) % SimpleLight.Total
    if t < SimpleLight.Yellow: return 0
    if t < SimpleLight.Yellow + SimpleLight.Red: return 1
    else: return 2

def _test():
    test_env = SimpleRoad({'render_mode': 'ascii'})
    test_env.render()
    try:
        import ray.rllib.utils
        ray.rllib.utils.check_env(test_env)
    except ImportError: pass
    test_env.reset()
    for i in range(100):
      obs, reward, done, info = test_env.step([1.0])
      if done: print('ok', i, reward); break
    else:
      assert False

_test()
