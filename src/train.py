import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, concatenate, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import pickle
from tqdm import tqdm
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint

# Download NLTK resources
nltk.download('punkt')

# Load captions file
captions_file = r'C:\Users\RISHABH\OneDrive\Desktop\image caption generator\image_captioning_project\data\flickr8k_captions.txt'
captions = open(captions_file, 'r').read()

# Create a dictionary mapping image names to captions
def load_captions(captions):
    mapping = {}
    for line in captions.strip().split('\n'):
        tokens = line.strip().split(',')
        if len(tokens) < 2:
            continue
        image_id, caption = tokens[0], tokens[1]
        image_id = image_id.split('#')[0]
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
    return mapping

all_captions_mapping = load_captions(captions)
all_captions_mapping.pop('image', None)  # Remove header
print(f"Total images: {len(all_captions_mapping)}")

# Clean captions
import string
def clean_captions(captions_mapping):
    table = str.maketrans('', '', string.punctuation)
    for img_id, captions in captions_mapping.items():
        for i, caption in enumerate(captions):
            caption = caption.lower()
            caption = caption.translate(table)
            caption = caption.strip()
            caption = ' '.join([word for word in caption.split() if len(word) > 1])
            caption = 'startseq ' + caption + ' endseq'
            captions[i] = caption

clean_captions(all_captions_mapping)

# Tokenizer
all_captions = []
for captions in all_captions_mapping.values():
    all_captions.extend(captions)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary Size: {vocab_size}")

# Maximum length of a caption
max_length = max(len(caption.split()) for caption in all_captions)
print(f"Maximum caption length: {max_length}")

# Preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = preprocess_input(img)
    return img

# Create CNN model for feature extraction
def create_cnn_model():
    inputs = Input(shape=(224, 224, 3))
    x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = x
    model = Model(inputs, outputs, name='CNN_Model')
    return model

cnn_model = create_cnn_model()
cnn_model.summary()

# Extract features for all images
def extract_features(cnn_model, images_directory, captions_mapping):
    features = {}
    for img_name in tqdm(captions_mapping.keys(), desc="Extracting image features"):
        img_path = os.path.join(images_directory, img_name)
        img = preprocess_image(img_path)
        img = np.expand_dims(img, axis=0)
        feature = cnn_model.predict(img, verbose=0)
        img_id = img_name.split('.')[0]
        features[img_id] = feature[0]
    return features

# Extract image features
features = extract_features(cnn_model, r'C:\Users\RISHABH\OneDrive\Desktop\image caption generator\image_captioning_project\data\flickr8k_images', all_captions_mapping)
print(f"Extracted features for {len(features)} images")

# Create sequences for training
def create_sequences(tokenizer, max_length, captions_list, image_id, features):
    X1, X2, y = [], [], []
    for caption in captions_list:
        seq = tokenizer.texts_to_sequences([caption])[0]
        for i in range(1, len(seq)):
            in_seq = seq[:i]
            out_seq = seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(features)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

# Prepare sequences for the model
print("Preparing sequences...")
X1, X2, y = [], [], []

for img_id, captions_list in tqdm(all_captions_mapping.items(), desc="Preparing sequences"):
    img_key = img_id.split('.')[0]  # Extract image ID without extension
    if img_key in features:  # Ensure the feature exists
        feature = features[img_key]
        xi1, xi2, yi = create_sequences(tokenizer, max_length, captions_list, img_key, feature)
        X1.extend(xi1)
        X2.extend(xi2)
        y.extend(yi)
    else:
        print(f"Feature not found for image: {img_id}")

X1 = np.array(X1)
X2 = np.array(X2)
y = np.array(y)

print(f"X1 shape: {X1.shape}, X2 shape: {X2.shape}, y shape: {y.shape}")

# LSTM model
def create_lstm_model(vocab_size, max_length):
    inputs1 = Input(shape=(256,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    decoder1 = concatenate([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

lstm_model = create_lstm_model(vocab_size, max_length)
lstm_model.summary()
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam')

# Define checkpoint
checkpoint = ModelCheckpoint('models/lstm_model.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

# Train the model
lstm_model.fit([X1, X2], y, epochs=50, batch_size=128, callbacks=[checkpoint], verbose=1)

# Save the tokenizer, max_length, and features for evaluation
with open('models/tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('models/max_length.pkl', 'wb') as handle:
    pickle.dump(max_length, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('models/features.pkl', 'wb') as handle:
    pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the CNN model
cnn_model.save('models/cnn_model.h5')
