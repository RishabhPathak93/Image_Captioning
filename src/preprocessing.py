import os
import numpy as np
import string
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import nltk
nltk.download('punkt')

# Load and clean captions
def load_captions(captions_file):
    mapping = {}
    with open(captions_file, 'r') as f:
        for line in f:
            tokens = line.strip().split(',')
            if len(tokens) < 2:
                continue
            image_id, caption = tokens[0], tokens[1]
            image_id = image_id.split('#')[0]
            if image_id not in mapping:
                mapping[image_id] = []
            mapping[image_id].append(caption)
    return mapping

def clean_captions(captions_mapping):
    table = str.maketrans('', '', string.punctuation)
    for img_id, captions in captions_mapping.items():
        for i, caption in enumerate(captions):
            caption = caption.lower().translate(table).strip()
            caption = ' '.join([word for word in caption.split() if len(word) > 1])
            captions[i] = 'startseq ' + caption + ' endseq'

# Preprocess images
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = preprocess_input(img)
    return img
