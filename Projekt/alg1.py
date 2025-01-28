from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
import subprocess

def calculate_similarity(text1, text2):
    """Compares two files for similarity using cosine similarity."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    return similarity_score[0][0]


def check_plagiarism(similarity):
    if similarity > 0.8:
        print("Warning: High similarity detected! Possible plagiarism.")
    elif similarity > 0.5:
        print("Moderate similarity detected. Further review recommended.")
    else:
        print("Low similarity. Files are likely not plagiarized.")

def divide_text(text, parts):
    part_length = len(text) // parts
    return [text[i * part_length:(i + 1) * part_length] for i in range(parts)] + [text[parts * part_length:]] if parts > 1 else [text]

def partial_similarity(plik1,plik2,parts):
    divided = divide_text(plik1, parts)
    similarity_scores_text = [calculate_similarity(part, plik2) for part in divided]
    for i, score in enumerate(similarity_scores_text, 1):
        print(f"Part {i} Similarity: {score * 100:.2f}%\n")


#Odczytywanie zawartości plików i ich podział na wzory matematyczne oraz tekst
def read_file(f1):
    try:
        with open(f1, 'r', encoding='utf-8') as f:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            konwerter = os.path.join(current_dir, "Konwerter.py")
            wynik1 = subprocess.run(['python', konwerter, f1], capture_output=True, text=True)            
       
            output_lines = wynik1.stdout.strip().split("###################")

            plain_text_1 = output_lines[0]  
            formulas_1 = output_lines[1]
        return plain_text_1,formulas_1
    except FileNotFoundError:
        print(f"Plik nie został znaleziony! Ścieżka: {f1}")

  

f1 = sys.argv[1]
f2 = sys.argv[2]
parts=int(sys.argv[3])

plain_text_1,formulas_1=read_file(f1)
plain_text_2,formulas_2=read_file(f2)

sim_text= calculate_similarity(plain_text_1, plain_text_2)
sim_mat = calculate_similarity(formulas_1, formulas_2)

print(f"Podobieństwo tekstów: {sim_text * 100:.2f}%\n")
check_plagiarism(sim_text)
partial_similarity(plain_text_1,plain_text_2,parts)
print()
print()
print(f"Podobieństwo wzorów matematycznych: {sim_mat * 100:.2f}%\n")
check_plagiarism(sim_mat)
partial_similarity(formulas_1,formulas_2,parts)


