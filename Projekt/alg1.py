from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
import subprocess

def calculate_similarity(text1, text2):
    
    # Create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    
    # Compute cosine similarity
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity
 

#Odczytywanie zawartości plików i ich podział na wzory matematyczne oraz tekst
f1 = sys.argv[1]
f2 = sys.argv[2]

try:
    with open(f1, 'r', encoding='utf-8') as f:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        konwerter = os.path.join(current_dir, "Konwerter.py")
        wynik1 = subprocess.run(['python', konwerter, f1], capture_output=True, text=True)            
        output_lines = wynik1.stdout.strip().split("###################")

        plain_text_1 = output_lines[0]  
        formulas_1 = output_lines[1]  
except FileNotFoundError:
    print(f"Plik nie został znaleziony! Ścieżka: {f1}")

try:
    with open(f2, 'r', encoding='utf-8') as f:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        konwerter = os.path.join(current_dir, "Konwerter.py")
        wynik2 = subprocess.run(['python', konwerter, f2], capture_output=True, text=True)            
        output_lines2 = wynik2.stdout.strip().split("###################")

        plain_text_2 = output_lines2[0]  
        formulas_2 = output_lines2[1]  
except FileNotFoundError:
    print(f"Plik nie został znaleziony! Ścieżka: {f1}")
  
    
sim_text= calculate_similarity(plain_text_1, plain_text_2)
 
sim_mat = calculate_similarity(formulas_1, formulas_2)

print(f"Podobieństwo tekstów: {sim_text * 100:.2f}%\n")
print(f"Podobieństwo wzorów matematycznych: {sim_mat * 100:.2f}%")