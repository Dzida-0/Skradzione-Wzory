from TexSoup import TexSoup
import re
import sys

# Określenie ścieżki pliku
file_path = sys.argv[1]
#print('Tu konwerter')


# Funkcja do przetwarzania pliku LaTeX na tekst
def latex_to_plain_text_texsoup(latex_content):
    soup = TexSoup(latex_content)
    plain_text = []
    for text in soup.text:
        plain_text.append(text)

    return ' '.join(plain_text).strip()
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        latex_content = f.read()
        soup = TexSoup(latex_content)

    my1_math=re.findall(r'\\begin{align\*}.*?\\end{align\*}', latex_content, re.DOTALL)
    my2_math=re.findall(r'\$\$.*?\$\$', latex_content)

    smieci1=re.findall(r'\$\$\\begin{array}.*?\\end{array}\$\$', latex_content, re.DOTALL)
    smieci2=re.findall(r'\\begin{tabular}.*?\\end{tabular}', latex_content, re.DOTALL)
    smieci3=re.findall(r'\$\$\s*\\begin{array}.*?\\end{array}\s*\$\$', latex_content, re.DOTALL)

    # Łączenie wyników
    all_my_math = my1_math + my2_math
    smieci=smieci1+smieci2+smieci3+all_my_math

    pattern = '|'.join([re.escape(sentence) for sentence in smieci ])

    # Usuwamy te zdania z tekstu
    text_cleaned = re.sub(pattern, '', latex_content)
    plain_text = latex_to_plain_text_texsoup(text_cleaned)

    print(f"Tekts: {plain_text}")
    print('###################')
    print(f"Wzory: {str(all_my_math)}")
    
except FileNotFoundError:
    print(f"Błąd: Plik '{file_path}' nie został znaleziony.")
except Exception as e:
    print(f"Wystąpił błąd: {e}")



