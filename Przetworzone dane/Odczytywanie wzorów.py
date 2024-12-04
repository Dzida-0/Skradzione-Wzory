file_path = r"C:\Users\NowyLOMBARD\Desktop\Kasia\Studia\Inżynieria\matematyczne\md2.txt"  # Upewnij się, że plik znajduje się w tym samym katalogu lub podaj pełną ścieżkę

with open(file_path, 'r', encoding='utf-8') as f:
        zawartosc = f.read()

wzory=zawartosc.split('\'')
wzory= [wzor for wzor in wzory if wzor !=', ']
wzory=wzory[1:-1]
for w in wzory:
   print(w)
