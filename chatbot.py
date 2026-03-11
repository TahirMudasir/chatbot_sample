import json
import random
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
def preprocess(text):
    # Tokenize and lowercase
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join(tokens)
# Load intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Prepare patterns and responses
patterns = []
tags = []
responses = {}
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern.lower())
        tags.append(intent['tag'])
    responses[intent['tag']] = intent['responses']

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(patterns)

print(" ChatBot is ready! Type 'quit' to exit.")
print("-" * 40)

while True:
    message = input("You: ").lower()
    
    if message in ['quit', 'exit', 'bye']:
        print("Bot: Goodbye! 👋")
        break
    
    # Convert message to TF-IDF vector
    msg_vec = vectorizer.transform([message])
    pattern_vecs = vectorizer.transform(patterns)
    
    # Find best match
    similarities = cosine_similarity(msg_vec, pattern_vecs).flatten()
    best_match_idx = np.argmax(similarities)
    best_match_tag = tags[best_match_idx]
    
    # Respond if good match
    if similarities[best_match_idx] > 0.3:  # Threshold
        response = random.choice(responses[best_match_tag])
        print(f"Bot: {response}")
    else:
        print("Bot: Sorry, I didn't understand that. Try asking about greetings, student help, or say goodbye!")
