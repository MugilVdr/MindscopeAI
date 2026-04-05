import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load dataset
df = pd.read_csv("data_csv/mindscope_master_dataset.csv")

# Remove rows where text is missing
df = df.dropna(subset=["text"])

# Remove rows where mental_state is missing
df = df.dropna(subset=["mental_state"])

# Remove rows where text is empty
df = df[df["text"].str.strip() != ""]
print("Dataset size after cleaning:", len(df))
# Reduce dataset size
df = df.sample(n=100000, random_state=42)

print("Dataset size used for training:", len(df))

# Basic text cleaning
def clean_text(text):
    
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)

texts = df["text"]
labels = df["mental_state"]

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# Convert text to numerical features
vectorizer = TfidfVectorizer(max_features=3000)

X = vectorizer.fit_transform(texts)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build model
model = Sequential()

model.add(Dense(256, activation="relu", input_shape=(3000,)))
model.add(Dense(128, activation="relu"))
model.add(Dense(y_train.shape[1], activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Save model
model.save("models/text_mental_model.keras")

# Save vectorizer
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

# Save encoder
pickle.dump(encoder, open("models/label_encoder.pkl", "wb"))

print("Text mental model training completed")