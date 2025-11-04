from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data/wikiqa", exist_ok=True)

print("📥 Loading WikiQA via Hugging Face datasets...")
dataset = load_dataset("wiki_qa")

train_df = pd.DataFrame(dataset["train"])
train_df.to_csv("data/wikiqa/wikiqa_clean.csv", index=False)

print("✅ Saved cleaned dataset to data/wikiqa/wikiqa_clean.csv")
print(train_df.head())
