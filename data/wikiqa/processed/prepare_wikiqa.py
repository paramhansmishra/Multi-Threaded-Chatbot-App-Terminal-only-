import pandas as pd

# Load the preprocessed WikiQA dataset
df = pd.read_csv("C:/Prog/CHATBOT/data/wikiqa/wikiqa_clean.csv")

# Keep only the rows marked as correct answers
df = df[df["label"] == 1]

# Select and rename relevant columns
df = df[["question", "answer"]]
df.columns = ["question", "answer"]

# Remove duplicates and empty rows
df = df.drop_duplicates().dropna()

# Save to a clean CSV for the bot
output_path = "C:/Prog/CHATBOT/data/wikiqa/wikiqa_final.csv"
df.to_csv(output_path, index=False)
print(f"✅ Saved {len(df)} question-answer pairs to {output_path}")
