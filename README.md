# reinforcement_learning_to_play_cartpole_game
A DQN model and an Actor-Critic model to paly cartpole games

## Introduction

the objective of reinforcement learning is to train an intelligent agent that is capable of interacting with an environment
intelligently. For instance, in the game of Go, the GO playing AI learns to excel in playing the game (the environment) 
through playing tens of thousands of game over and over again (usually with itself!) and discovers subtle strategies through
trial and error. At present, the two most popular classes of reinforcement learning algorithms are Q Learning and 
Policy Gradients. Q learning is a type of value iteration method aims at approximating the Q function, 
while Policy Gradients is a method to directly optimize in the action space. 
Please read the refered articles for more details about the concept of Deep-Q-Learning and Policy gradient.


## Methodology

### DQN
1. Build a neural network model to represent the agent and initialize with random values
2. Get the initial state from gym
3. The agent pick an action (random or the best action available) and get the reward and next state from gym
4. Experience replay: Save (state, action, reward, next_state, done) to a memory, sample from the memory, update reward and train the agent.

### Actor-Critic
1. Build an actor model to predict the action
2. Build a critic model to return a score of the action
3. Play the game and get reward, next_state and done signal from gym and remember (state, action, reward, next_state, done)
4. Experience replay: sample data from memory, update actor with the difference between new_value and original_value, update critic with decayed reward, and train both actor and critic.

## Result

### DQN
episode: 473/800, score: 499, e: 0.094</br>
episode: 474/800, score: 499, e: 0.094</br>
episode: 475/800, score: 499, e: 0.093</br>
episode: 476/800, score: 499, e: 0.093</br>
episode: 477/800, score: 337, e: 0.092</br>
episode: 478/800, score: 499, e: 0.092</br>
episode: 479/800, score: 499, e: 0.092</br>
episode: 480/800, score: 499, e: 0.091</br>
episode: 481/800, score: 499, e: 0.091</br>
episode: 482/800, score: 499, e: 0.09</br>
episode: 483/800, score: 499, e: 0.09</br>
episode: 484/800, score: 499, e: 0.089</br>
episode: 485/800, score: 499, e: 0.089</br>
episode: 486/800, score: 499, e: 0.088</br>
episode: 487/800, score: 499, e: 0.088</br>
episode: 488/800, score: 249, e: 0.088</br>
episode: 489/800, score: 499, e: 0.087</br>
episode: 490/800, score: 499, e: 0.087</br>
episode: 491/800, score: 499, e: 0.086</br>
episode: 492/800, score: 499, e: 0.086</br>

### Actor-Critic
episode: 576/800, score: 499, e: 0.056</br>
episode: 577/800, score: 499, e: 0.056</br>
episode: 578/800, score: 499, e: 0.056</br>
episode: 579/800, score: 499, e: 0.055</br>
episode: 580/800, score: 499, e: 0.055</br>
episode: 581/800, score: 499, e: 0.055</br>
episode: 582/800, score: 499, e: 0.055</br>
episode: 583/800, score: 499, e: 0.054</br>
episode: 584/800, score: 499, e: 0.054</br>
episode: 585/800, score: 499, e: 0.054</br>
episode: 586/800, score: 499, e: 0.054</br>
episode: 587/800, score: 499, e: 0.053</br>
episode: 588/800, score: 499, e: 0.053</br>
episode: 589/800, score: 499, e: 0.053</br>
episode: 590/800, score: 499, e: 0.052</br>
episode: 591/800, score: 499, e: 0.052</br>
episode: 592/800, score: 297, e: 0.052</br>
episode: 593/800, score: 287, e: 0.052</br>
episode: 594/800, score: 499, e: 0.051</br>
episode: 595/800, score: 499, e: 0.051</br>
episode: 596/800, score: 499, e: 0.051</br>

## References
https://flyyufelix.github.io/2017/10/12/dqn-vs-pg.html </br>
https://github.com/gregretkowski/notebooks/blob/master/ActorCritic-with-OpenAI-Gym.ipynb </br>
https://keon.io/deep-q-learning/ </br>
