from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
import subprocess
import time
from report_generator1 import ReportGenerator

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity

def divide_text(text, parts):
    part_length = len(text) // parts
    return [text[i * part_length:(i + 1) * part_length] for i in range(parts)] + [text[parts * part_length:]] if parts > 1 else [text]

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


if __name__ == "__main__":
    # File paths

    f1 = sys.argv[1]
    f2 = sys.argv[2]
    parts=int(sys.argv[3])
    n1=sys.argv[4]
    n2=sys.argv[5]
    zamienniki = {
        'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n',
        'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z',
        'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 'Ł': 'L', 'Ń': 'N',
        'Ó': 'O', 'Ś': 'S', 'Ź': 'Z', 'Ż': 'Z'
        }
    n1=''.join(zamienniki.get(znak, znak) for znak in n1)
    n2=''.join(zamienniki.get(znak, znak) for znak in n2)
          

    plain_text_1,formulas_1=read_file(f1)
    plain_text_2,formulas_2=read_file(f2)

    f2=f2.replace('\\','/')
    f1=f1.replace('\\','/')

    # Get number of parts from the user
    if not (1 <= parts <= 10):
        print("Invalid number of parts. Please choose between 1 and 10.")
    else:
        divided_texts = divide_text(plain_text_1, parts)
        divided_mat=divide_text(formulas_1, parts)
        # Measure execution time
        start_time = time.time()
        similarity_scores_text = [calculate_similarity(part, plain_text_2) for part in divided_texts]
        similarity_scores_mat = [calculate_similarity(part, formulas_2) for part in divided_mat]
        
        exec_time = time.time() - start_time

        # Prepare data for the report
        report_data = {
            'date': int(time.time()),
            'file': f"{n1} vs {n2}",
            'exec_time': f"{exec_time:.2f} seconds",
            'numbers_tekst': similarity_scores_text,
            'numbers_mat': similarity_scores_mat
        }

        # Generate the LaTeX report
        generator = ReportGenerator()
        generator.generate_latex_report(report_data)
        generator.open_latex_report()