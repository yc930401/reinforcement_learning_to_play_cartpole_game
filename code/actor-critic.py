import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 800


class ACAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.actor = self.actor_model()
        self.critic = self.critic_model()

    def actor_model(self):
        actor_model = Sequential()
        actor_model.add(Dense(128, init='lecun_uniform', input_dim=self.state_size, activation='relu'))
        actor_model.add(Dense(256, init='lecun_uniform', activation='relu'))
        actor_model.add(Dense(self.action_size, init='lecun_uniform', activation='linear'))
        actor_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        actor_model.summary()
        return actor_model

    def critic_model(self):
        critic_model = Sequential()
        critic_model.add(Dense(128, init='lecun_uniform', input_dim=self.state_size, activation='relu'))
        critic_model.add(Dense(256, init='lecun_uniform', activation='relu'))
        critic_model.add(Dense(1, init='lecun_uniform', activation='linear'))
        critic_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        critic_model.summary()
        return critic_model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        #print(self.memory[-1])

    def actor_act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        actions = self.actor.predict(state)
        return np.argmax(actions[0])  # returns action

    def critic_act(self, state):
        action_value = self.critic.predict(state)
        return action_value

    def actor_reply(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if not done:
                original_value = self.critic.predict(state)
                new_value = self.critic.predict(next_state)
                y = new_value-original_value
                target_f = self.actor.predict(state)
                target_f[0][action] = y
                self.actor.fit(state, target_f, epochs=2, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def critic_replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.critic.predict(next_state)[0]))
            self.critic.fit(state, [target], epochs=2, verbose=0)

    def load(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)

    def save(self, actor_path, critic_path):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = ACAgent(state_size, action_size)
    # agent.load("cartpole-ac.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # env.render()
            action = agent.actor_act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.critic_replay(batch_size)
            agent.actor_reply(batch_size)
        # if e % 10 == 0:
        #     agent.save("cartpole-ac.h5")