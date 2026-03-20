from wordle_rl.entropy_train import EntropyTrainer

trainer = EntropyTrainer()
trainer.train(steps=20000)