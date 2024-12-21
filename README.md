# Text Classification using Naive Bayes and TF-IDF

This project demonstrates text classification using a Naive Bayes classifier along with TF-IDF vectorization. The goal is to categorize text documents into predefined topics such as Finance, Sports, Technology, and Politics. The script uses Natural Language Processing (NLP) techniques for text preprocessing, including tokenization and stopword removal.

## Requirements

- Python 3.x
- NLTK
- Scikit-learn

### Install Required Libraries

To install the necessary libraries, you can use the following command:

```bash
pip install nltk scikit-learn

```
# Code Explanation

## 1. Imports

```bash 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

```

These libraries are used for text preprocessing (NLTK) and machine learning (Scikit-learn).

## 2. Downloading NLTK Data
Before running the script, we download necessary NLTK data for stopwords and tokenization 

```bash 
nltk.download('stopwords')
nltk.download('punkt')

```

## 3. Dataset
A small dataset is provided with texts and their corresponding topics.

```bash 
texts = [
    "The stock market is down today.",
    "Cristiano Ronaldo scored a hat-trick last night.",
    "The latest iPhone has impressive features.",
    "New policies were introduced in the government meeting.",
    "Tesla's stock price surged after the new product announcement.",
    "The Olympic Games are exciting to watch.",
]
topics = ["Finance", "Sports", "Technology", "Politics", "Finance", "Sports"]

```

## 4. Preprocessing
Text data is preprocessed by removing stopwords and non-alphanumeric characters.

```bash 
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    return " ".join(filtered_words)

```

## 5. TF-IDF Vectorization
The TfidfVectorizer is used to convert the text data into numerical features

```bash 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_texts)

```

## 6. Train/Test Split
The dataset is split into training and testing sets using Scikit-learn's train_test_split


```bash 
X_train, X_test, y_train, y_test = train_test_split(X, topics, test_size=0.3, random_state=42)

```

## 7. Model Training
A Multinomial Naive Bayes model is used for classification.

```bash 
model = MultinomialNB()
model.fit(X_train, y_train)

```

## 8. Evaluation
The model is evaluated using the classification_report from Scikit-learn, which provides precision, recall, and F1-score.


```bash 
predictions = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, predictions))

```

# How to Run
1. Clone the repository:
    ```bash 
        git clone https://github.com/wesimosiuoa/topic_classification_nlp.git
    ```
2. Navigate to the project directory:

    ```bash 
        cd topic_classification_nlp.git
    ```
3. Install dependencies:
     ```bash 
        pip install -r requirements.txt
    ```

4. Run the script:
     ```bash 
        python text_classification.py
    ```

# Output
The output will be a classification report showing the performance of the model on the test dataset

```bash 
Classification Report:
               precision    recall  f1-score   support

       Finance       1.00      1.00      1.00         1
        Sports       1.00      1.00      1.00         1
    Technology       1.00      1.00      1.00         1
      Politics       1.00      1.00      1.00         1

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4

```