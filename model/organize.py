import json
import os
import pandas as pd

# Directory with JSON files
json_dir = "../data-extractor/data"
data_list = []

# Loop through JSON files
for file in os.listdir(json_dir):
    if file.endswith(".json"):
        with open(os.path.join(json_dir, file), 'r') as f:
            data = json.load(f)
            if "O" in data.get("species", []):
                data_list.append(data)

# Create DataFrame
df = pd.DataFrame(data_list)
print(df.columns)  # Inspect available fields

