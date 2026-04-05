import zipfile
import os

# ZIP file location
zip_path = r"D:\Mindscope_AI_phase2.0\raw data(zip)\Mindscope AI(face dataset).zip"

# Extract location inside project
extract_to = r"D:\Mindscope_AI_phase2.0\extracted_data\face_dataset"

# Create folder if not exists
os.makedirs(extract_to, exist_ok=True)

# Extract ZIP
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("✅ Dataset extracted successfully to:", extract_to)