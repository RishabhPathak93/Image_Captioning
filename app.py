import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved models and tokenizer
cnn_model = tf.keras.models.load_model('models/cnn_model.h5')
lstm_model = tf.keras.models.load_model('models/lstm_model.h5')

# Manually compile the model to avoid warnings (necessary for LSTM)
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy')

# Load the tokenizer
with open('models/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load max_length
with open('models/max_length.pkl', 'rb') as handle:
    max_length = pickle.load(handle)

# Preprocess the image for CNN model
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img)
    if img.shape[-1] == 4:  # If the image has an alpha channel (transparency), remove it
        img = img[..., :3]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img

# Updated generate_caption function
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        # Ensure the photo (image features) has the correct shape
        if photo.shape[-1] != 256:
            photo = np.expand_dims(photo, axis=0)  # Add batch dimension if missing

        # Predict the next word in the sequence
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)  # Get index of the predicted word

        word = tokenizer.index_word.get(yhat)  # Get the word from the index
        if word is None:
            break

        in_text += ' ' + word  # Append the predicted word to the input sequence

        # If the word is 'endseq', stop the generation
        if word == 'endseq':
            break

    # Remove 'startseq' and 'endseq' from the final caption
    final_caption = in_text.split(' ', 1)[1].rsplit(' ', 1)[0]

    return final_caption

# Streamlit app interface
st.title("🖼️ Image Caption Generator")

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image for CNN
    st.write("Processing image...")
    preprocessed_image = preprocess_image(image)
    
    # Extract image features using the CNN model
    image_features = cnn_model.predict(preprocessed_image, verbose=0)
    st.write(f"Extracted Features Shape: {image_features.shape}")  # Debugging

    # Generate caption using the LSTM model
    st.write("Generating caption...")
    caption = generate_caption(lstm_model, tokenizer, image_features, max_length)
    
    # Display the generated caption
    st.subheader("Generated Caption")
    st.write(f"**{caption}**")
