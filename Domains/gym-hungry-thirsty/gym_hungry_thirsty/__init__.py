from gym.envs.registration import register

register(
    id='hungry-thirsty-v0',
    entry_point='gym_hungry_thirsty.envs:HungryThirstyEnv',
)
