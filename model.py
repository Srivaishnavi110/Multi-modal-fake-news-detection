"""Step 1: Data Loading and Preparation"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("fakeddit_sampled.csv")

# Preview the dataset
print(df.head())

# Split dataset into features (X) and target variable (y)
X = df[['title', 'image_url']]
y = df['label']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Step 2: Text Preprocessing"""
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize and fit TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=1000)
X_train_text = vectorizer.fit_transform(X_train['title']).toarray()
X_test_text = vectorizer.transform(X_test['title']).toarray()

# Print the shape of the vectorized text data
print(f"Vectorized train text shape: {X_train_text.shape}")
print(f"Vectorized test text shape: {X_test_text.shape}")

"""Step 3: Image Preprocessing"""
import requests
from PIL import Image
from io import BytesIO
import numpy as np

def download_image(url):
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        print(e)
        return None

def preprocess_image(image, target_size=(224, 224)):
    if image is None:
        # Return a zero array with 3 channels for missing images
        return np.zeros(target_size + (3,))
    image = image.convert('RGB')  # Convert to RGB
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize
    return image_array

# Example applying the functions (simplified for demonstration)
X_train_images = []
X_test_images = []

for url in X_train['image_url']:
    raw_image = download_image(url)
    processed_image = preprocess_image(raw_image)
    if processed_image is not None:
        X_train_images.append(processed_image)
    else:
        # Append a zeros array if the image couldn't be downloaded/preprocessed
        X_train_images.append(np.zeros((224, 224, 3)))

# Assuming X_train_images is your list of image arrays
all_same_shape = all(img.shape == (224, 224, 3) for img in X_train_images)
if not all_same_shape:
    print("Not all images have the same shape.")

# Proceed with conversion if all shapes are consistent
if all_same_shape:
    X_train_images = np.array(X_train_images)

# Repeat for X_test
for url in X_test['image_url']:
    raw_image = download_image(url)
    processed_image = preprocess_image(raw_image)
    if processed_image is not None:
        X_test_images.append(processed_image)
    else:
        X_test_images.append(np.zeros((224, 224, 3)))

X_test_images = np.array(X_test_images)

"""Step4 : Model Definition, Compilation, and Training"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Flatten, Conv2D, MaxPooling2D

# Define the model architecture
text_input = Input(shape=(X_train_text.shape[1],), name='text_input')
text_dense = Dense(256, activation='relu')(text_input)

image_input = Input(shape=(224, 224, 3), name='image_input')
conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(image_input)
maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
flatten = Flatten()(maxpool1)
image_dense = Dense(256, activation='relu')(flatten)

concat = Concatenate()([text_dense, image_dense])
output = Dense(1, activation='sigmoid')(concat)
model = Model(inputs=[text_input, image_input], outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([X_train_text, X_train_images], y_train, epochs=10, batch_size=32, validation_split=0.2)
model.save('my_fakeddit_model.h5')

"""Step 5: Evaluation"""
# Evaluate the model on the test set
evaluation = model.evaluate([X_test_text, X_test_images], y_test)
print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")

