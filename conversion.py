import os
import zipfile
import pandas as pd

# ==============================
# GIVE YOUR 3 ZIP FILE PATHS HERE
# ==============================
zip_files = [
    r"D:\Mindscope_AI_phase2.0\raw data(zip)\mindscop ai dataset 1.zip",
    r"D:\Mindscope_AI_phase2.0\raw data(zip)\dataset for mindscop AI.zip",
    r"D:\Mindscope_AI_phase2.0\raw data(zip)\minscope ai dataset 2.zip"
]

# Output folders
extract_folder = "extracted_data"
output_folder = "data_csv"

os.makedirs(extract_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# ==============================
# EXTRACT + CONVERT
# ==============================
for zip_path in zip_files:

    zip_name = os.path.basename(zip_path).replace(".zip", "")
    current_extract_path = os.path.join(extract_folder, zip_name)

    os.makedirs(current_extract_path, exist_ok=True)

    # Step 1: Extract ZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(current_extract_path)

    print(f"✅ Extracted: {zip_name}")

    # Step 2: Convert files inside ZIP to CSV
    for root, dirs, files in os.walk(current_extract_path):
        for file in files:
            file_path = os.path.join(root, file)

            try:

                # TSV → CSV
                if file.endswith(".tsv"):
                    df = pd.read_csv(file_path, sep="\t")
                    new_name = zip_name + "_" + file.replace(".tsv", ".csv")
                    df.to_csv(os.path.join(output_folder, new_name), index=False)
                    print(f"Converted TSV → CSV: {new_name}")

                # CSV → copy clean
                elif file.endswith(".csv"):
                    df = pd.read_csv(file_path)
                    new_name = zip_name + "_" + file
                    df.to_csv(os.path.join(output_folder, new_name), index=False)
                    print(f"Processed CSV: {new_name}")

                # JSON → CSV
                elif file.endswith(".json"):
                    df = pd.read_json(file_path)
                    new_name = zip_name + "_" + file.replace(".json", ".csv")
                    df.to_csv(os.path.join(output_folder, new_name), index=False)
                    print(f"Converted JSON → CSV: {new_name}")

            except Exception as e:
                print(f"⚠ Error in file {file}: {e}")

print("\n✅ All ZIP datasets extracted and converted successfully!")