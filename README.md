# Text Classification Project

Welcome to the **Text Classification Project**! This project is designed to train a model for classifying texts based on their emotional content and then using it to categorize new texts into corresponding emotional categories.

## Overview

This project consists of two main components:

1. **text_classification_model_trainer.py**: This script is responsible for training a TensorFlow model to classify texts based on their emotional content. The model is trained using a dataset of labeled text samples.
2. **text_classification_classifier.py**: This script uses the trained model to classify new text samples into predefined emotional categories and save the results into separate files.

## Libraries Used

The following libraries are used in this project:

- **[tensorflow](https://www.tensorflow.org/)**: TensorFlow is used for building and training the text classification model.
- **[numpy](https://numpy.org/)**: NumPy is used for numerical operations and handling arrays.
- **[pandas](https://pandas.pydata.org/)**: Pandas is used for data manipulation and analysis.
- **[scikit-learn](https://scikit-learn.org/)**: Scikit-learn is used for splitting the dataset into training and test sets.
- **[pickle](https://docs.python.org/3/library/pickle.html)**: Pickle is used for saving and loading the tokenizer objects.

## Detailed Explanation

### `text_classification_model_trainer.py`

This script is the core of the project, responsible for training the text classification model. The key components of the script are:

- **ModelTrainer Class**: This class handles the training process for the text classification model. It includes methods to load the dataset, tokenize the text, build the model, train the model, and save both the trained model and tokenizers.
  - **load_data() Method**: Loads the dataset from a CSV file, tokenizes the texts and labels, and splits them into training and test sets.
  - **build_model() Method**: Defines and compiles a Bidirectional LSTM model architecture for text classification.
  - **train_model() Method**: Trains the model on the prepared dataset, using a specified number of epochs and batch size.
  - **save_model() Method**: Saves the trained model to a file.
  - **save_tokenizers() Method**: Saves the text and label tokenizers to files.

### `text_classification_classifier.py`

This script uses the trained model to classify new text samples and save the results into separate files based on the predicted emotions. The key components of the script are:

- **EmotionClassifier Class**: This class loads the trained model and tokenizers, and provides methods to classify text samples.
  - **predict_emotion() Method**: Predicts the emotional category of a given sentence using the trained model.
  - **classify_sentences() Method**: Classifies each sentence in a text file and writes them to separate files based on the predicted emotion.

### How It Works

1. **Model Training**:
    - The `text_classification_model_trainer.py` script reads the dataset from a CSV file.
    - The texts and labels are tokenized, and the model is trained to classify the texts into different emotional categories.
    - The trained model and tokenizers are saved for later use.

2. **Text Classification**:
    - The `text_classification_classifier.py` script loads the trained model and tokenizers.
    - It reads a text file, predicts the emotional category for each sentence, and writes the sentences into separate files based on the predicted emotions.

### Dataset

The dataset used for training the model can be accessed via this [Dataset](https://drive.google.com/drive/folders/1Sp0bBe-qSXct9LfpZO_cM3SaGaJQWYfg?usp=sharing).

## Installation and Setup

To use this project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/amiriiw/text_classification
    cd text_classification
    ```

2. Install the required libraries:

    ```bash
    pip install tensorflow numpy pandas scikit-learn
    ```

3. Prepare your dataset (a CSV file with 'Clean_Text' and 'Emotion' columns).

4. Train the model:

    ```bash
    python text_classification_model_trainer.py
    ```

5. Classify new texts:

    ```bash
    python text_classification_classifier.py
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
