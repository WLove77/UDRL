from minigrid.wrappers import RGBImgObsWrapper

from env import FourRoomsEnv
from udrl import UDRL

# 创建环境实例
env = FourRoomsEnv()
env = RGBImgObsWrapper(env)

# Auto
# udrl_auto = UDRL(
#     env=env,
#     buffer_size=10000,
#     hidden_size=64,
#     learning_rate=1e-3,
#     return_scale=0.02,
#     horizon_scale=0.02,
#     batch_size=2,
#     n_updates_per_iter=150,
#     n_episodes_per_iter=15,
#     last_few=50,
#     mode='auto'
# )
# udrl_auto.train_and_plot(max_episodes=10)

udrl_manual = UDRL(
    env=env,
    buffer_size=10000,
    hidden_size=64,
    learning_rate=1e-3,
    return_scale=0.02,
    horizon_scale=0.02,
    batch_size=2,
    n_updates_per_iter=150,
    n_episodes_per_iter=15,
    last_few=50,
    mode='manual',
    filename="fourRoomRecord.pkl"
)

udrl_manual.train_and_plot(max_episodes=10)
