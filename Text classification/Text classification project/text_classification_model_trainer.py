"""----------------------------------------------------------------------------
well come, this is amiriiw, this is a simple project about text classification.
    in this file we will train our text classification model.
----------------------------------------------------------"""
import pickle  # https://docs.python.org/3/library/pickle.html
import numpy as np  # https://numpy.org/devdocs/user/absolute_beginners.html
import pandas as pd  # https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html
import tensorflow as tf  # https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html
from sklearn.model_selection import train_test_split  # https://scikit-learn.org/stable/user_guide.html
from tensorflow.keras.preprocessing.text import Tokenizer  # https://www.tensorflow.org/guide/keras
from tensorflow.keras.preprocessing.sequence import pad_sequences  # https://www.tensorflow.org/guide/keras
"""-----------------------------------------------------------------------------------------------------"""


class ModelTrainer:
    def __init__(self, vocab_size=10000, max_length=100, embedding_dim=16):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.label_tokenizer = Tokenizer()
        self.model = None
        self.num_classes = None

    def load_data(self, file_path, test_size=0.2):
        data = pd.read_csv(file_path)
        texts = data['Clean_Text'].astype(str).values
        labels = data['Emotion'].values
        self.label_tokenizer.fit_on_texts(labels)
        label_sequences = self.label_tokenizer.texts_to_sequences(labels)
        self.num_classes = len(self.label_tokenizer.word_index) + 1  # Set num_classes
        labels_encoded = tf.keras.utils.to_categorical(np.array(label_sequences) - 1, num_classes=self.num_classes)
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels_encoded, test_size=test_size, random_state=42, stratify=labels_encoded
        )
        self.tokenizer.fit_on_texts(train_texts)
        train_sequences = self.tokenizer.texts_to_sequences(train_texts)
        test_sequences = self.tokenizer.texts_to_sequences(test_texts)
        train_padded = pad_sequences(train_sequences, maxlen=self.max_length, padding='post', truncating='post')
        test_padded = pad_sequences(test_sequences, maxlen=self.max_length, padding='post', truncating='post')
        return train_padded, test_padded, train_labels, test_labels

    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self, train_data, train_labels, test_data, test_labels, epochs=10, batch_size=32):
        history = self.model.fit(
            train_data, train_labels, epochs=epochs, validation_data=(test_data, test_labels), batch_size=batch_size
        )
        return history

    def save_model(self, file_path):
        self.model.save(file_path)

    def save_tokenizers(self, tokenizer_path='tokenizer.pickle', label_tokenizer_path='label_tokenizer.pickle'):
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        with open(label_tokenizer_path, 'wb') as f:
            pickle.dump(self.label_tokenizer, f)


if __name__ == "__main__":
    trainer = ModelTrainer()
    train_data_padded, test_data_padded, train_labels_encoded, test_labels_encoded = trainer.load_data('dataset.csv')
    trainer.build_model()
    trainer.train_model(train_data_padded, train_labels_encoded, test_data_padded, test_labels_encoded)
    trainer.save_model('emotion_classification_model.h5')
    trainer.save_tokenizers()
"""-----------------------"""
