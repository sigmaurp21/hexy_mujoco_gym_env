import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env


DEFAULT_CAMERA_CONFIG = {
    'distance': 1.5,
}


class HexyEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml_file='Hexy_ver_2.3/hexy-v2.3.xml'):
        utils.EzPickle.__init__(**locals())
        self._obs_buffer1 = np.zeros(18)
        self._obs_buffer2 = np.zeros(18)
        self._obs_buffer3 = np.zeros(18)
        self._act_buffer1 = np.zeros(18)
        self._act_buffer2 = np.zeros(18)
        self._act_buffer3 = np.zeros(18)
        # self._tor_buffer = np.array([0.3, 0.3, 0.3])
        mujoco_env.MujocoEnv.__init__(self, xml_file, 25)
        self.init_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, -0.7, 0.7,
                                   0.0, -0.7, 0.7,
                                   0.0, -0.7, 0.7,
                                   0.0, -0.7, 0.7,
                                   0.0, -0.7, 0.7,
                                   0.0, -0.7, 0.7])
        self.init_qvel = np.zeros(24)
        obslow = np.full((108,), -1.5, dtype=np.float64)
        obshigh = np.full((108,), 1.5, dtype=np.float64)
        self.observation_space = spaces.Box(low=obslow, high=obshigh, dtype=np.float64)
        actlow = np.full((18,), -1.5, dtype=np.float64)
        acthigh = np.full((18,), 1.5, dtype=np.float64)
        self.action_space = spaces.Box(low=actlow, high=acthigh, dtype=np.float64)

    @property
    def is_healthy(self):
        # if hexy was tilted or changed position too much, reset environments
        for i in range(len(self.sim.data.contact)):
            if not self.sim.data.contact[i].geom2 in [0, 12, 22, 32, 42, 52, 62]:
                print(self.sim.data.contact[i].geom2, 'collision occurred!')
                return False
        is_healthy = (np.abs(self.state_vector()[0:2]) < 0.1).all()\
        #    and self.state_vector()[2] > -0.04
        # np.abs(self.state_vector()[1]) < 0.1\
        # and (np.abs(self.state_vector()[3:6]) < 0.5).all()\
        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy
        return done

    def step(self, action):
        yaw_init = self.state_vector()[5]
        self.do_simulation(action, self.frame_skip)

        # update action and observation history
        self._act_buffer3 = self._act_buffer2
        self._act_buffer2 = self._act_buffer1
        self._act_buffer1 = action[:]
        self._obs_buffer3 = self._obs_buffer2
        self._obs_buffer2 = self._obs_buffer1
        self._obs_buffer1 = self.state_vector()[6:24]

        # calculate rewards and costs
        yaw_del = self.state_vector()[5] - yaw_init
        xy_err = np.sum(np.square(self.state_vector()[0:2]))
        # self._tor_buffer[:-1] = self._tor_buffer[1:]
        # self._tor_buffer[-1] = np.mean(np.square(self.sim.data.actuator_force[:]))
        torque_rms = np.sqrt(np.mean(np.square(self.sim.data.actuator_force[:])))
        reward = (yaw_del+0.02) / (torque_rms+1) / (xy_err+0.01)

        done = self.done
        observation = self._get_obs()
        info = {
            'x_delta': yaw_del,
            'torque_rms': torque_rms,
            'y_error': xy_err,
            'total': reward
        }

        return observation, reward, done, info

    def _get_obs(self):
        # take account of history
        return np.concatenate([self._obs_buffer1, self._obs_buffer2, self._obs_buffer3
                               , self._act_buffer1, self._act_buffer2, self._act_buffer3
                               ])

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
