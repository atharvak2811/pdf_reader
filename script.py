import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text
def preprocess_text(text):
    sentences = sent_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    preprocessed_sentences = []
    stop_words = set(stopwords.words("english"))

    for sentence in sentences:
        words = nltk.word_tokenize(sentence.lower())
        words = [lemmatizer.lemmatize(word) for word in words if word.isalnum()]
        words = [word for word in words if word not in stop_words]
        preprocessed_sentences.append(" ".join(words))

    return preprocessed_sentences

# Read the PDF file
def read_pdf(file_name):
    pdf_text = ""
    with open(file_name, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            pdf_text += page.extract_text()
    return pdf_text

# Process text and create TF-IDF vectors
def process_and_vectorize_text(text):
    preprocessed_sentences = preprocess_text(text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    return preprocessed_sentences, vectorizer, tfidf_matrix

# Function to find the most relevant sentence
def find_most_relevant_sentence(question, text, vectorizer, tfidf_matrix):
    question = preprocess_text(question)
    question_tfidf = vectorizer.transform(question)
    cosine_similarities = cosine_similarity(question_tfidf, tfidf_matrix)
    max_similarity_index = cosine_similarities.argmax()
    return text[max_similarity_index]

# Main function
if __name__ == "__main__":
    file_name = r'C:\Users\hp 2082\Desktop\pdf_reader\input.pdf' 
    pdf_text = read_pdf(file_name)
    sentences, vectorizer, tfidf_matrix = process_and_vectorize_text(pdf_text)

    while True:
        user_question = input("Ask a question (or type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break

        answer = find_most_relevant_sentence(user_question, sentences, vectorizer, tfidf_matrix)
        print("Answer:", answer)

