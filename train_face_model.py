from face_data_loader import train_data, val_data
from face_cnn_model import model

EPOCHS = 10

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Save trained model
model.save("models/face_emotion_model.keras")

print("✅ Face Emotion Model Trained & Saved Successfully!")