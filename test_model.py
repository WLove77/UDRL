from minigrid.wrappers import RGBImgObsWrapper

from env import FourRoomsEnv
from udrl import UDRL

# 创建环境实例
env = FourRoomsEnv()
env = RGBImgObsWrapper(env)

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
    filename="fourRoomRecorddata.pkl"
)
udrl_manual.train_and_plot(max_episodes=5)
udrl_manual.save_model('model_FourRoom_Mannul.pth')

#############################################################################################################################################################################

env = FourRoomsEnv(render_mode='human')
env = RGBImgObsWrapper(env)
udrl_manual.load_model('model_FourRoom_Mannul.pth')
udrl_manual.test_model(num_episodes=5,render=True)
