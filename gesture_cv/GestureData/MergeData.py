import pandas as pd

df1 = pd.read_csv("gesture_data1.csv")
df2 = pd.read_csv("gesture_data2.csv")

full = pd.concat([df1, df2], ignore_index=True)

# optional: shuffle so training isn't biased toward one session
full = full.sample(frac=1, random_state=42).reset_index(drop=True)

full.to_csv("gesture_data_full.csv", index=False)

print("Merged into gesture_data_full.csv")
print("Unique classes:", sorted(full["label"].unique()))
print("Total samples:", len(full))
