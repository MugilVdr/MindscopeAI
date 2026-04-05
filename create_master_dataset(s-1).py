import pandas as pd
import os

# =========================
# FILE PATHS (EDIT IF NEEDED)
# =========================
GOEMOTIONS_PATH = "C:\\Users\\kaviya\\OneDrive\\Desktop\\Minscope AI Phase 2\\data_csv\\mindscop ai dataset 1_go_emotions_dataset.csv"  # change name if different
SUICIDE_PATH = "C:\\Users\\kaviya\\OneDrive\\Desktop\\Minscope AI Phase 2\\data_csv\\minscope ai dataset 2_Suicide_Detection.csv"

OUTPUT_PATH = "data_csv/mindscope_master_dataset.csv"

# =========================
# LOAD DATASETS
# =========================
print("Loading datasets...")
go_df = pd.read_csv(GOEMOTIONS_PATH)
suicide_df = pd.read_csv(SUICIDE_PATH)

# =========================
# CLEAN GOEMOTIONS DATASET
# =========================
print("Cleaning GoEmotions dataset...")

# Keep only text + emotion columns (drop id)
go_df = go_df.drop(columns=["id"], errors="ignore")

# Function to map multi-label emotions → mental_state
def map_emotion(row):
    if row["sadness"] == 1 or row["grief"] == 1 or row["remorse"] == 1:
        return "Depression"
    elif row["fear"] == 1 or row["nervousness"] == 1:
        return "Anxiety"
    elif row["anger"] == 1 or row["annoyance"] == 1:
        return "Stress"
    elif row["joy"] == 1 or row["love"] == 1 or row["gratitude"] == 1 or row["optimism"] == 1:
        return "Positive"
    elif row["neutral"] == 1:
        return "Neutral"
    else:
        return None  # drop unclear samples

# Apply mapping
go_df["mental_state"] = go_df.apply(map_emotion, axis=1)

# Keep only needed columns
go_df = go_df[["text", "mental_state"]]

# Remove null labels
go_df = go_df.dropna()

print("GoEmotions cleaned:", go_df.shape)

# =========================
# CLEAN SUICIDE DATASET
# =========================
print("Cleaning Suicide dataset...")

# Drop unwanted index column
suicide_df = suicide_df.drop(columns=["Unnamed: 0"], errors="ignore")

# Map suicide classes → mental states
def map_suicide(label):
    if label == "suicide":
        return "Depression"
    else:
        return "Neutral"

suicide_df["mental_state"] = suicide_df["class"].apply(map_suicide)

# Keep only needed columns
suicide_df = suicide_df[["text", "mental_state"]]

# Remove null text
suicide_df = suicide_df.dropna(subset=["text"])

print("Suicide dataset cleaned:", suicide_df.shape)

# =========================
# MERGE DATASETS
# =========================
print("Merging datasets...")

master_df = pd.concat([go_df, suicide_df], ignore_index=True)

# Final text cleaning
master_df["text"] = master_df["text"].str.lower()
master_df["text"] = master_df["text"].str.replace(r"http\S+", "", regex=True)
master_df["text"] = master_df["text"].str.replace(r"[^a-zA-Z\s]", "", regex=True)

# Remove duplicates
master_df = master_df.drop_duplicates(subset=["text"])

# Save master dataset
master_df.to_csv(OUTPUT_PATH, index=False)

print("✅ Master Dataset Created Successfully!")
print("Final Shape:", master_df.shape)
print("Saved at:", OUTPUT_PATH)