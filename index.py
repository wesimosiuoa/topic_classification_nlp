import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Example dataset
texts = [
    "The stock market is down today.",
    "Cristiano Ronaldo scored a hat-trick last night.",
    "The latest iPhone has impressive features.",
    "New policies were introduced in the government meeting.",
    "Tesla's stock price surged after the new product announcement.",
    "The Olympic Games are exciting to watch.",
]
topics = ["Finance", "Sports", "Technology", "Politics", "Finance", "Sports"]

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    return " ".join(filtered_words)

# Apply preprocessing
preprocessed_texts = [preprocess_text(text) for text in texts]

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_texts)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, topics, test_size=0.3, random_state=42)

# Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, predictions))
