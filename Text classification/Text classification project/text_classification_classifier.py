"""----------------------------------------------------------------------------
well come, this is amiriiw, this is a simple project about text classification.
    in this file we will classify the texts from the file's.
---------------------------------------------------------"""
import pickle  # https://docs.python.org/3/library/pickle.html
import numpy as np  # https://numpy.org/devdocs/user/absolute_beginners.html
import tensorflow as tf  # https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html
from tensorflow.keras.preprocessing.sequence import pad_sequences  # https://www.tensorflow.org/guide/keras
"""-----------------------------------------------------------------------------------------------------"""


class EmotionClassifier:
    def __init__(self, model_path='emotion_classification_model.h5', tokenizer_path='tokenizer.pickle', label_tokenizer_path='label_tokenizer.pickle', max_length=100):
        self.model = tf.keras.models.load_model(model_path)
        self.max_length = max_length
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        with open(label_tokenizer_path, 'rb') as handle:
            self.label_tokenizer = pickle.load(handle)
        self.index_to_label = {v: k for k, v in self.label_tokenizer.word_index.items()}

    def predict_emotion(self, sentence):
        sequence = self.tokenizer.texts_to_sequences([sentence])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_length, padding='post', truncating='post')
        prediction = self.model.predict(padded_sequence)
        predicted_label = self.index_to_label[np.argmax(prediction) + 1]
        return predicted_label

    def classify_sentences(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()
        sentences = text.split('.')
        emotion_files = {}
        for sentence in sentences:
            if sentence.strip():
                emotion = self.predict_emotion(sentence.strip())
                if emotion not in emotion_files:
                    emotion_files[emotion] = open(f'{emotion}.txt', 'a', encoding='utf-8')
                emotion_files[emotion].write(sentence.strip() + '\n')
        for file in emotion_files.values():
            file.close()


if __name__ == "__main__":
    classifier = EmotionClassifier()
    classifier.classify_sentences('text.txt')
"""---------------------------------------"""
