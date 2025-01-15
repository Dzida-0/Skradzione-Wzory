import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_file(file_path):
    """Reads the content of a file and returns it as a string."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        sys.exit(1)

def check_plagiarism(file1_path, file2_path):
    """Compares two files for similarity using cosine similarity."""
    # Read the content of both files
    text1 = read_file(file1_path)
    text2 = read_file(file2_path)

    # Create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Convert the texts into TF-IDF matrices
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Calculate cosine similarity
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    return similarity_score[0][0]

if __name__ == "__main__":
    # Example usage
    file1_path = "C:/Users/m.kramarz/Downloads/compare_files/md10.txt"
    file2_path = "C:/Users/m.kramarz/Downloads/compare_files/md9.txt"

    similarity = check_plagiarism(file1_path, file2_path)

    print(f"Similarity Score: {similarity * 100:.2f}%")

    if similarity > 0.8:
        print("Warning: High similarity detected! Possible plagiarism.")
    elif similarity > 0.5:
        print("Moderate similarity detected. Further review recommended.")
    else:
        print("Low similarity. Files are likely not plagiarized.")
