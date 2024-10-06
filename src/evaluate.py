import pickle
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

# Load tokenizer
with open('models/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load max_length
with open('models/max_length.pkl', 'rb') as handle:
    max_length = pickle.load(handle)

# Load extracted features
with open('models/features.pkl', 'rb') as handle:
    features = pickle.load(handle)

# Load models
cnn_model = load_model(r'C:\Users\RISHABH\OneDrive\Desktop\image caption generator\image_captioning_project\models\cnn_model.h5')
lstm_model = load_model(r'C:\Users\RISHABH\OneDrive\Desktop\image caption generator\image_captioning_project\models\lstm_model.h5')

# Generate captions using the trained LSTM model
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        # Convert input text to sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        
        # Retrieve the word corresponding to the predicted word index
        word = tokenizer.index_word.get(yhat)
        if word is None:
            print(f"Warning: Word index {yhat} not found in tokenizer index")
            break
        
        # Append the word to the caption
        in_text += ' ' + word
        
        # Stop if end of sequence is predicted
        if word == 'endseq':
            break
    return in_text

# Load captions
captions_file = r'C:\Users\RISHABH\OneDrive\Desktop\image caption generator\image_captioning_project\data\flickr8k_captions.txt'
captions_mapping = {}
with open(captions_file, 'r') as f:
    for line in f.readlines():
        tokens = line.strip().split(',')
        img_id, caption = tokens[0], tokens[1]
        img_id = img_id.split('#')[0]
        if img_id not in captions_mapping:
            captions_mapping[img_id] = []
        captions_mapping[img_id].append(caption)

# Clean and prepare captions
def clean_captions(captions_mapping):
    import string
    table = str.maketrans('', '', string.punctuation)
    for img_id, captions in captions_mapping.items():
        for i, caption in enumerate(captions):
            caption = caption.lower()
            caption = caption.translate(table)
            caption = caption.strip()
            captions[i] = 'startseq ' + caption + ' endseq'

clean_captions(captions_mapping)

# Evaluate model
actual, predicted = [], []

# Check if each image has a feature and evaluate it
for img_id, captions_list in tqdm(captions_mapping.items(), desc="Evaluating BLEU scores"):
    img_key = img_id.split('.')[0]  # Extract image ID without extension

    # Ensure the feature exists for the image
    if img_key in features:
        feature = features[img_key]
        
        # Check feature shape (debugging)
        print(f"Feature shape for image {img_id}: {np.array([feature]).shape}")

        # Generate the predicted caption
        y_pred = generate_caption(lstm_model, tokenizer, np.array([feature]), max_length)

        # Append actual and predicted captions for BLEU evaluation
        actual.append([caption.split() for caption in captions_list])
        predicted.append(y_pred.split())
    else:
        print(f"Feature for image {img_id} not found, skipping.")
        continue

# BLEU Scores
bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
print(f'BLEU-1: {bleu1}')
print(f'BLEU-2: {bleu2}')
