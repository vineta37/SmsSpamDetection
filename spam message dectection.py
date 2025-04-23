import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
file_path = "C:\\Users\\vinet\\Downloads\\archive\\spam.csv"
df = pd.read_csv(file_path, encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Display dataset shape and first few rows
print("Dataset Loaded:")
print(df.shape)
print(df.head(), "\n")

# Encode labels: ham -> 0, spam -> 1
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.strip()
    return text

df['cleaned_message'] = df['message'].apply(preprocess_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['cleaned_message'])
y = df['label_num']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training using Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Sample prediction
sample_sms = ["Congratulations! You’ve won a $1000 gift card. Reply to claim now.", 
              "Hey, are we still meeting for lunch today?"]
sample_cleaned = [preprocess_text(s) for s in sample_sms]
sample_vect = vectorizer.transform(sample_cleaned)
predictions = model.predict(sample_vect)

print("\nSample Predictions:")
for sms, pred in zip(sample_sms, predictions):
    print(f"Message: {sms}\nPrediction: {'Spam' if pred == 1 else 'ham'}\n")
