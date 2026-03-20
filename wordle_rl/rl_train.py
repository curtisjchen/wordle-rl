import torch
import numpy as np
import mlflow

from torch.distributions import Categorical

from .utils import load_words
from .env import WordleEnv
from .policy import PolicyNet


class RLTrainer:

    def __init__(self):

        guess_words = load_words("./data/words.txt")
        answer_words = load_words("./data/test_words.txt")

        self.guess_words = guess_words
        self.answer_words = answer_words

        self.env = WordleEnv(guess_words, answer_words)

        self.state_dim = 157
        self.action_dim = len(guess_words)

        self.policy = PolicyNet(self.state_dim, self.action_dim)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)

    def run_episode(self):

        state = torch.tensor(self.env.reset(), dtype=torch.float32)

        log_probs = []
        rewards = []

        for _ in range(6):

            logits = self.policy(state)

            dist = Categorical(logits=logits)

            action = dist.sample()

            guess = self.guess_words[action.item()]

            next_state, reward, done = self.env.step(guess)

            log_probs.append(dist.log_prob(action))
            rewards.append(reward)

            state = torch.tensor(next_state, dtype=torch.float32)

            if done:
                break

        return log_probs, rewards

    def update(self, log_probs, rewards):

        gamma = 0.99

        returns = []
        R = 0

        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)

        loss = 0

        for log_prob, R in zip(log_probs, returns):
            loss -= log_prob * R

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, episodes=50000):

        mlflow.set_experiment("wordle_rl")

        with mlflow.start_run():

            mlflow.log_param("episodes", episodes)
            mlflow.log_param("lr", 1e-4)

            for ep in range(episodes):

                log_probs, rewards = self.run_episode()

                loss = self.update(log_probs, rewards)

                total_reward = sum(rewards)

                mlflow.log_metric("reward", total_reward, step=ep)
                mlflow.log_metric("loss", loss, step=ep)

                if ep % 500 == 0:

                    print(
                        "episode",
                        ep,
                        "reward",
                        total_reward
                    )
                
                if ep % 5000 == 0 and ep > 0:
                    torch.save(self.policy.state_dict(), f"entropy_model_{ep}.pt")

            torch.save(self.policy.state_dict(), "entropy_model.pt")