# README.md

# Reddit Big Data Analysis Project

This repository contains a comprehensive analysis of Reddit data using various machine learning techniques and tools. The project is divided into multiple components, including data collection, preprocessing, sentiment analysis, and model training using Spark and TensorFlow. Below are the details of each file and its purpose within the project.

## Files and Directories
### pyspark_sentiment_analysis.ipynb
This notebook handles the sentiment analysis using PySpark and TensorFlow.

1. **Library Imports**:
   - Imports necessary libraries including `numpy`, `pandas`, `tensorflow`, and `pyspark`.

2. **Spark Session Setup**:
   - Configures and initializes a Spark session for handling data.

3. **Data Loading and Preprocessing**:
   - Loads data from a CSV file and preprocesses it for training the sentiment analysis model.

4. **Tokenization and Padding**:
   - Tokenizes the text data and pads the sequences to a fixed length.

5. **Model Building and Training**:
   - Builds and trains a GRU-based neural network for sentiment analysis using TensorFlow.

6. **Model Evaluation**:
   - Evaluates the model's performance on the test dataset and prints accuracy.

7. **Model Saving**:
   - Saves the trained model to a file for later use.

8. **Prediction Function**:
   - Defines a function to predict the sentiment of a given text using the trained model.

### reddit_project.ipynb
This notebook demonstrates the end-to-end pipeline for collecting, processing, and analyzing Reddit data.

1. **Data Retrieval**:
   - Connects to MongoDB to retrieve stored Reddit data.

2. **Data Transformation**:
   - Transforms and preprocesses the data for analysis.

3. **Pipeline Setup**:
   - Sets up a Spark ML pipeline for tokenization, stop word removal, TF-IDF, and classification.

4. **Model Training and Evaluation**:
   - Trains and evaluates different classifiers (Logistic Regression, Random Forest, Decision Tree) on the preprocessed data.

5. **Visualization**:
   - Visualizes the distribution of sentiments and subreddits using `matplotlib`.

### api.py
This Flask application serves as an API for predicting the sentiment of a given text using a pre-trained TensorFlow model.

1. **Flask Setup**:
   - Initializes a Flask application.

2. **Model Loading**:
   - Defines functions to load data, tokenizer, and the pre-trained sentiment analysis model.

3. **Sentiment Prediction Endpoint**:
   - Provides a `/sentiment` endpoint to predict the sentiment of a given text. 

4. **Model Initialization**:
   - Loads and prepares the model and tokenizer when the endpoint is first called.
### simulator.ipynb
This notebook simulates the data collection and preprocessing pipeline. It includes the following steps:

1. **Library Imports**:
   - Imports necessary libraries like `pyspark`, `nltk`, `numpy`, and `pandas`.
   - Downloads NLTK data for text preprocessing.

2. **Spark Session Setup**:
   - Configures and initializes a Spark session for handling large datasets efficiently.

3. **API Keys and Configuration**:
   - Loads API keys and MongoDB connection strings from JSON files.

4. **Data Loading and Display**:
   - Loads subreddit data from a CSV file into a Spark DataFrame and displays it.

5. **Random Subreddit Selection**:
   - Adds a random column to the DataFrame and selects a random subreddit.

6. **Reddit Data Collection**:
   - Uses the `praw` library to fetch data from Reddit based on the randomly selected subreddit.

7. **Text Preprocessing**:
   - Defines functions to preprocess text by removing punctuation, converting to lowercase, and removing stopwords.

8. **Sentiment Analysis**:
   - Sends text data to a sentiment analysis API and appends the sentiment score to the data.

9. **Data Transformation**:
   - Applies text preprocessing and transformation to the collected data.

10. **Database Insertion**:
    - Inserts the processed data into a MongoDB collection.

İşte `Setup and Installation` bölümünün güncellenmiş hali:

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/reddit-big-data-analysis.git
   cd reddit-big-data-analysis
   ```

2. **Create and activate the conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate reddit_big_data_analysis
   ```

3. **Download NLTK data**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

4. **Set up Spark and Hadoop**:
   - Ensure that Apache Spark and Hadoop are installed and configured on your system.

5. **Run the notebooks**:
   - Use Jupyter Notebook or Jupyter Lab to run the provided `.ipynb` files.

6. **Run the Flask API**:
   ```bash
   python api.py
   ```

Bu adımları izleyerek gerekli bağımlılıkları yükleyebilir ve projeyi çalıştırabilirsiniz.
## Usage

1. **Data Collection**:
   - Run `simulator.ipynb` to collect and preprocess Reddit data.

2. **Sentiment Analysis**:
   - Use the `/sentiment` endpoint provided by the Flask API to predict sentiment for new text inputs.

3. **Data Analysis and Visualization**:
   - Run `reddit_project.ipynb` to analyze and visualize the collected data.

