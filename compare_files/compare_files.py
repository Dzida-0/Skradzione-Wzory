from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(file1, file2):
    # Read the files
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        text1 = f1.read()
        text2 = f2.read()
    
    # Create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    
    # Compute cosine similarity
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity
 
f1 = "C:/Users/m.kramarz/Downloads/data (1)/data/g0pA_taska.txt"
f2 = "C:/Users/m.kramarz/Downloads/compare_files/md9.txt"
 
similarity_score = calculate_similarity(f1, f2)
print(f"Similarity: {similarity_score * 100:.2f}%")