# Fake News Detector

This repository contains a Jupyter Notebook (`Fake_News_Detector.ipynb`) that experiments with and creates a working framework for training a fine-tuned model to detect fake news articles using a small dataset.

## Project Overview

The goal of this project is to develop a model that can accurately distinguish between real and fake news articles. The process involves:

-   Loading and formatting data.
-   Computing statistical analysis of the data.
-   Experimenting with training parameters.
-   Training a RoBERTa-based model and saving it.

## Notebook Contents

The Jupyter Notebook (`Fake_News_Detector.ipynb`) includes the following key steps:

1.  **Data Loading and Preprocessing:**
    -      Loading the dataset from CSV files.
    -      Adding labels to differentiate between real and fake articles.
    -      Combining the datasets into a single DataFrame.
    -      Performing Exploratory Data Analysis (EDA) to understand the data distribution.
    -      Filtering articles based on character count.
    -   Preprocessing text data by removing special characters, stopwords, and lemmatizing words.
2.  **Tokenization:**
    -      Tokenizing the preprocessed text using the RoBERTa tokenizer.
3.  **Model Training:**
    -      Preparing the data for training using PyTorch Datasets and DataLoaders.
    -      Fine-tuning a pre-trained RoBERTa model for sequence classification.
    -      Setting up the optimizer and loss function.
    -      Implementing a training and validation loop.
    -      Saving model checkpoints during training.
    -   Plotting training and validation metrics.
4.  **Results and Future Improvements:**
    -   Analyzing the training results, including loss, F1 score, and accuracy.
    -   Discussing potential improvements for future iterations, such as:
        -   Including more recent data.
        -   Expanding the variety of topics.
        -   Incorporating true articles from diverse sources.
        -   Optimizing model size for faster inference.
        -   Deploying the model as a web application.

## Dependencies

To run the notebook, you will need the following Python libraries:

-   `pandas`
-   `matplotlib`
-   `seaborn`
-   `nltk`
-   `transformers`
-   `torch`
-   `scikit-learn`
-   `kagglehub`

You can install these libraries using pip:

```bash
pip install pandas matplotlib seaborn nltk transformers torch scikit-learn kagglehub
```

You will also need to download the necessary NLTK resources:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## Dataset

The dataset used in this project is the "Fake and Real News Dataset" from Kaggle. You can download it using the kagglehub library.
The download cell is within the notebook.

## Usage

1.  Clone the repository.
2.  Install the required dependencies, including an appropriate version of cuda.
3.  Ensure there is sufficient storage space on the local drive to save model checkpoints, around 20gb. 
4.  Open the Jupyter Notebook (`Fake_News_Detector.ipynb`).
5.  Run the cells in the notebook to load, process, and train the model.

## Results

The best model produced by the notebook was the second checkpoint, which had a validation accuracy of 99.9% and a loss value below 0.005. See graphs of training results below.

## Future Work

Future iterations of this project will focus on:

-   Expanding the dataset with more recent and diverse data.
-   Optimizing the model for better performance and efficiency.
-   Deploying the model as a web application.

## Acknowledgments

-   The "Fake and Real News Dataset" from Kaggle.
-   The RoBERTa pre-trained model from Hugging Face Transformers.
-   The NLTK library for natural language processing.
