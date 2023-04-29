import math
import gymnasium as gym


class Agent:
    """
    https://gymnasium.farama.org/environments/box2d/lunar_lander/
    observation = state of spacecraft - position and velocity
    The state is an 8-dimensional vector:
    1:2, the coordinates of the lander in x & y,
    3:4 linear velocities in x & y,
    5 angle,
    6:its angular velocity,
    7:8 two booleans that represent whether each leg is in contact with the ground or not.
    action_space = all possible moves spacecraft can make
    0: do nothing
    1: fire left orientation engine
    2: fire main engine
    3: fire right orientation engine
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        self.action_space = action_space
        self.observation_space = observation_space

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        The act method should take an observation and return an action - 1) fire left, 2) right, 3) main engine.
        """
        return self.action_space.sample()

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        The learn method should take an observation, a reward, a boolean indicating
        whether the episode has terminated, and a boolean indicating whether the episode was truncated:
        """
        # 1) Closer to landing pad
        x, y = self.observation_space.low[0], self.observation_space.low[1]
        o_x, o_y = observation[0], observation[1]

        dist_pad = math.sqrt(x**2 + y**2)
        dist_o_pad = math.sqrt(o_x**2 + o_y**2)

        if dist_o_pad < dist_pad:
            reward += 1

        # 2) Velocity
        x_v, y_v = self.observation_space.low[2], self.observation_space.low[3]
        o_x_v, o_y_v = observation[2], observation[3]

        v = math.sqrt(x_v**2 + y_v**2)
        v_o = math.sqrt(o_x_v**2 + o_y_v**2)

        if v < v_o:
            reward += 1

        # 3) is lander more tilted?
        angle = self.observation_space.low[4]

        if angle != 0:
            reward -= 1

        # 4) Leg(s) in contact?
        is_left_leg_touching, is_right_leg_touching = (
            self.observation_space.low[6],
            self.observation_space.low[7],
        )
        reward += 10 * (is_left_leg_touching + is_right_leg_touching)

        # 5) side engine
        self.action_space

        # 7) landing / termination
        if terminated:
            is_lander_body_touching_moon = not (
                is_left_leg_touching and is_right_leg_touching
            )
            is_lander_outside_viewport = x > 0

            is_not_succ_land = (
                is_lander_body_touching_moon or is_lander_outside_viewport
            )

            if is_not_succ_land:
                reward -= 100
            else:
                reward += 100

            reward = reward

        pass
