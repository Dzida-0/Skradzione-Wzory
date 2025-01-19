# Określenie ścieżki pliku
file_path = r'C:\Users\NowyLOMBARD\Desktop\Kasia\Studia\Inżynieria\dane\md12 algebry boolea.lect'  # Upewnij się, że plik znajduje się w tym samym katalogu lub podaj pełną ścieżkę
name='\\md12.txt'
from TexSoup import TexSoup

# Funkcja do przetwarzania pliku LaTeX na tekst
def latex_to_plain_text_texsoup(latex_content):
    # Parsowanie LaTeX za pomocą TexSoup
    soup = TexSoup(latex_content)

    # Wyodrębnianie tekstu i zamiana polskich znaków
    plain_text = []
    for text in soup.text:
        plain_text.append(text)

    return ' '.join(plain_text).strip()


try:
    with open(file_path, 'r', encoding='utf-8') as f:
        latex_content = f.read()
    

    

except FileNotFoundError:
    print(f"Błąd: Plik '{file_path}' nie został znaleziony.")
except Exception as e:
    print(f"Wystąpił błąd: {e}")

import re

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

# Oczyszczony tekst
plain_text = latex_to_plain_text_texsoup(text_cleaned)


# Ścieżka do katalogu i nazwa pliku
output_file_path =r"C:\Users\NowyLOMBARD\Desktop\Kasia\Studia\Inżynieria\matematyczne" + name

# Zawartość do zapisania
content_to_save = str(all_my_math)

# Zapis do pliku
try:
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(content_to_save)
    print(f"Plik został zapisany w: {output_file_path}")
except Exception as e:
    print(f"Wystąpił błąd podczas zapisywania pliku: {e}")


output_file_path =r"C:\Users\NowyLOMBARD\Desktop\Kasia\Studia\Inżynieria\teksty" + name

# Zawartość do zapisania
content_to_save = plain_text

# Zapis do pliku
try:
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(content_to_save)
    print(f"Plik został zapisany w: {output_file_path}")
except Exception as e:
    print(f"Wystąpił błąd podczas zapisywania pliku: {e}")