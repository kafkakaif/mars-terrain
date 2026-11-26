from datasets import load_dataset

train = load_dataset("hassanjbara/AI4MARS", split="train")
print(train)
print(train[0].keys())