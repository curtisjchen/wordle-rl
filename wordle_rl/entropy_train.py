import torch
import numpy as np
import mlflow

from .utils import load_words, feedback, filter_candidates, encode_state
from .policy import PolicyNet


class EntropyTrainer:

    def __init__(self):

        guess_words = load_words("./data/words.txt")
        answer_words = load_words("./data/test_words.txt")

        self.guess_words = guess_words
        self.answer_words = answer_words

        self.state_dim = 157
        self.action_dim = len(guess_words)

        self.policy = PolicyNet(self.state_dim, self.action_dim)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)

    def entropy_score(self, guess, candidates):

        partitions = {}

        for w in candidates:

            p = feedback(guess, w)

            partitions.setdefault(p, 0)
            partitions[p] += 1

        probs = np.array(list(partitions.values())) / len(candidates)

        return -(probs * np.log(probs)).sum()

    def best_guess(self, candidates):

        scores = []

        for g in self.guess_words[:2000]:

            scores.append(self.entropy_score(g, candidates))

        return self.guess_words[np.argmax(scores)]

    def generate_state(self):

        size = np.random.randint(5,100)

        candidates = list(np.random.choice(self.answer_words,size))

        state = encode_state(candidates)

        target = self.best_guess(candidates)

        action = self.guess_words.index(target)

        return state, action

    def train(self, steps=20000):

        mlflow.set_experiment("wordle_entropy")

        with mlflow.start_run():

            mlflow.log_param("steps", steps)

            for step in range(steps):

                state, action = self.generate_state()

                state = torch.tensor(state, dtype=torch.float32)

                logits = self.policy(state)

                loss = torch.nn.functional.cross_entropy(
                    logits.unsqueeze(0),
                    torch.tensor([action])
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                mlflow.log_metric("loss", loss.item(), step=step)

                if step % 500 == 0:

                    print("step", step, "loss", loss.item())
            
                
                if step % 5000 == 0 and step > 0:
                    torch.save(self.policy.state_dict(), f"./models/entropy_model_{step}.pt")

            torch.save(self.policy.state_dict(), "./models/entropy_model.pt")