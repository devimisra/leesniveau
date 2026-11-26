import pandas as pd

print("Downloading dataset from Hugging Face...")
df = pd.read_json("hf://datasets/UniversalCEFR/elg_cefr_nl/elg-cefr-nl.json")

# 1. Preview the data in your terminal
print("Data loaded successfully! Here are the first 5 rows:")
print(df.head())

print(f"\nTotal number of Dutch texts found: {len(df)}")

# 2. Save a local copy for Phase 1 training
output_filename = "../data/local_elg_cefr_nl.json"
df.to_json(output_filename, orient="records")
print(f"\nSaved local copy to: {output_filename}")