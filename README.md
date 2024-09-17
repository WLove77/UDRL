## UDRL with FourRooms Environment

## Introduction
This project implements a four-rooms environment using the Upside-Down Reinforcement Learning (UDRL) algorithm. The project contains several Python files, each responsible for a different function. The following is a detailed description of each file.

## File descriptions

### `env.py`
This file defines the `FourRoomsEnv` class, a custom environment that inherits from `MiniGridEnv`.
The main functions include:
1. initialize the environment parameters, such as grid size, maximum number of steps, and so on.
2. generate the grid layout of four rooms, including the positions of walls and doors.
3. defining the initial and target positions of the intelligences.
4. perform actions in the environment, calculate rewards, and record state transfer data.
5. provide a method for accessing the collected data.

### `udrl.py`
This file implements the core part of the UDRL algorithm and includes the following classes and methods:
1. `ReplayBuffer` class: used to store and manage the experience collected by the intelligences in the environment.
2. `BF` class: a behavioral function model that uses a convolutional neural network to predict the probability of an intelligent's action in a given state.
3. `UDRL` class: implements the main logic of the UDRL algorithm, including the warm-up step, training the behavioral function, generating new samples, and evaluating the performance of the intelligence.
   - `__init__` method: initializes each parameter of UDRL and chooses to perform warm-up or load data according to the mode.
   - `load_data_to_replay_buffer` method: loads data from a file into the experience buffer.
   - `warm_up` method: warms up and fills the experience buffer in automatic mode.
   - `sampling_exploration` method: samples exploratory commands from the buffer.
   - `select_time_steps` method: select random time steps for training.
   - `create_training_input` method: creates training input based on selected time steps.
   - `create_training_examples` method: create training examples.
   - `train_behavior_function` method: trains the behavior function model.
   - The `evaluate` method: evaluates the performance of the intelligence in the environment.
   - `generate_episode` method: generates new experience samples.
   - `run_upside_down` method: run the UDRL algorithm.
   - `train_and_plot` method: train the model and plot the results.

### `manual_game.py`
This file is used to manually manipulate the intelligences and record data. The main functions include:
1. `manual_play_and_record` function: allows the user to control the SmartBody using the keyboard and record the SmartBody's movements and state to a file.
2. setting up the Pygame window, which renders the environment and receives user input.

### `model_test.ipynb`
This file contains the code used to test the UDRL algorithm. The main functions include:
1. create UDRL instances and train them according to the selected mode (automatic or manual).
2. printing key metrics during training, such as rewards, losses, etc.
3. perform tests using pytest to ensure correctness and performance of the algorithm.

## Recorded data files

### `recorded_episodes.pkl`
This file stores empirical data recorded when manually manipulating the intelligences. The main contents include:
1. the state of the intelligences at each time step.
2. the actions performed by the intelligences at each time step.
3. the rewards obtained by the SmartBody at each time step.
This data is used to train the UDRL algorithm in manual mode.

The JuypterNotebook is ready to run and shows the results.

### `Conclusion`
UDRL does not work well enough in minigrid with four rooms and randomly generated locations, sometimes agent can find terminal easily, probably because minigird is a Markov environment
Try to apply UDRL to non-Markovian environments, let the agent be trained with our data, and let the behavior function approximate our Replay buffer, it may be more effective, similar to imitation learning, such as AlphaGo.
