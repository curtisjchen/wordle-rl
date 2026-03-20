from wordle_rl.rl_train import RLTrainer

trainer = RLTrainer()
trainer.train(episodes=50000)

torch.save(self.policy.state_dict(), "entropy_model.pt")