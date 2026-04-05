import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset paths
train_dir = "extracted_data/face_dataset/train"
test_dir = "extracted_data/face_dataset/test"

# Image size (FER standard = 48x48, but 64x64 gives better accuracy)
IMG_SIZE = (64, 64)
BATCH_SIZE = 32

# Training Data Generator (with augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.1  # 10% validation from train
)

# Test Data Generator (NO augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load Training Data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Validation Data
val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Test Data
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("✅ Data Loaded Successfully")
print("Classes:", train_data.class_indices)