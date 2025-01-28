import streamlit as st
import subprocess
import tempfile
import os
import time
import matplotlib.pyplot as plt
from report_generator1 import ReportGenerator
from model2 import Model

# Edycja koloru tła i stylów
st.markdown(
      """
    <style>

    

    /* Dodanie paska między kolumnami */
    .stColumn:first-child {
        border-right: 4px solid #ffffff; /* Biały pasek po prawej stronie pierwszej kolumny */
        padding-right: 15px; /* Odstęp wewnętrzny dla estetyki */
    }
    .stColumn:last-child {
        padding-left: 15px; /* Odstęp dla drugiej kolumny */
    }
    </style>
    """, unsafe_allow_html=True
)


# Tytuł aplikacji
st.title("Skradzione Wzory")

# Opis aplikacji
st.write("Sprawdź, czy w plikach występuje plagiat.")

# Kolumny dla dwóch sekcji
col1, col2 = st.columns(2)

# Sekcja dla algorytmu 1

with col1:
    
    st.subheader("Algorytm 1: Porównanie dwóch plików")
    plik1 = st.file_uploader("Wybierz plik1", type=["tex"], key="plik1")
    plik2 = st.file_uploader("Wybierz plik2", type=["tex"], key="plik2")

    liczba1 = st.slider(
    'Wybierz na ile sekcji ma być dzielony dokument',  
    min_value=1,      
    max_value=10,     
    value=1,         
    step=1             
    )
    
    if st.button("Oblicz dla Algorytmu 1"):
        st.session_state.result1 = None
        
        if plik1 and plik2:
            st.write(f"Uruchamianie Algorytmu 1 dla {liczba1} sekcji dokumentu...")
            t1=time.time()
            
            with tempfile.NamedTemporaryFile(delete=False, mode='wb') as tmp1, tempfile.NamedTemporaryFile(delete=False, mode='wb') as tmp2:
                tmp1.write(plik1.getbuffer())
                tmp2.write(plik2.getbuffer())
                tmp1_path = tmp1.name
                tmp2_path = tmp2.name
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            alg1_path = os.path.join(current_dir, "alg1.py")
            
            result1 = subprocess.run(['python', alg1_path, tmp1_path, tmp2_path, str(liczba1)], capture_output=True, text=True)
            
            if result1.stdout:
                st.session_state.result1 = result1.stdout.strip()
                st.write(f" \n {result1.stdout.strip()}")
            if result1.stderr:
                #st.error(f"Błąd Algorytmu 1: Jeden z plików zawiera niepoprawny kod LaTex")
                st.error(f'Błąd: {result1.stderr}')
            os.remove(tmp1_path)
            os.remove(tmp2_path)

            t2=time.time()
            st.write(f"Czas pracy: {t2-t1:.2f}s")

        else:
            st.error("Proszę załadować oba pliki do porównania.")

    if st.button("Wygeneruj szczegółowy raport Latex dla Alg1"):
            if st.session_state.result1 :
                st.write('Generowanie raportu LaTeX...')
                with tempfile.NamedTemporaryFile(delete=False, mode='wb') as tmp1, tempfile.NamedTemporaryFile(delete=False, mode='wb') as tmp2:
                    tmp1.write(plik1.getbuffer())
                    tmp2.write(plik2.getbuffer())
                    tmp1_path = tmp1.name
                    tmp2_path = tmp2.name
            
                current_dir = os.path.dirname(os.path.abspath(__file__))
                raport1_path = os.path.join(current_dir, "raport1.py")
            
                raport1 = subprocess.run(['python', raport1_path, tmp1_path, tmp2_path, str(liczba1),plik1.name,plik2.name], capture_output=True, text=True)


                #if raport1.stdout:
                 #   st.write(f"{raport1.stdout.strip()}")
                #if raport1.stderr:
                #st.error(f"Błąd Algorytmu 1: Jeden z plików zawiera niepoprawny kod LaTex")
                 #   st.error(f'Błąd: {raport1.stderr}')

                os.remove(tmp1_path)
                os.remove(tmp2_path)

                st.write("Wygenerowano raport LaTeX na podstawie wyników algorytmów.")
            else:
                st.write("Najpierw wygeneruj wyniki algorytmu")   


# Sekcja dla algorytmu 2
with col2:
    
    st.subheader("Algorytm 2: Porównanie pliku z bazą danych")
    plik3 = st.file_uploader("Wybierz plik3", type=["tex"], key="plik3")

    liczba2 = st.slider(
    'Wybierz na ile sekcji ma być dzielony dokument',  
    min_value=1,      
    max_value=5,     
    value=1,         
    step=1             
    )
    
    if st.button("Oblicz dla Algorytmu 2"):
        t1=time.time()
        st.session_state.result2 = None
        
        if plik3:
            st.write("Uruchamianie Algorytmu 2...")
            
            with tempfile.NamedTemporaryFile(delete=False, mode='wb') as tmp3:
                tmp3.write(plik3.getbuffer())
                tmp3_path = tmp3.name
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            alg2_path = os.path.join(current_dir, "alg2.py")
            
            result2 = subprocess.run(['python', alg2_path, tmp3_path,str(liczba2),plik3.name], capture_output=True, text=True)
            
            if result2.stdout:
                st.session_state.result2 = result2.stdout.strip()
                st.write(f" \n {result2.stdout.strip()}")
            
            #if result2.stderr:
            #    st.error(f"Błąd Algorytmu 2: {result2.stderr}")   
            os.remove(tmp3_path)
            t2=time.time()
            st.write(f"Czas pracy: {t2-t1:.2f}s")

        else:
            st.error("Proszę załadować plik do porównania z bazą danych.")
        
    if st.button("Wygeneruj szczegółowy raport Latex dla Alg2"):
        if st.session_state.result2 and plik3:
            st.write('Generowanie raportu LaTeX...')

            with tempfile.NamedTemporaryFile(delete=False, mode='wb') as tmp3:
                tmp3.write(plik3.getbuffer())
                tmp3_path = tmp3.name

            current_dir = os.path.dirname(os.path.abspath(__file__))
            raport2_path = os.path.join(current_dir, "raport2.py")
            
            raport2 = subprocess.run(['python', raport2_path, tmp3_path, plik3.name, str(liczba2)], capture_output=True, text=True)
            #if raport2.stdout:
             #   st.write(f" \n {raport2.stdout.strip()}")
            
            #if raport2.stderr:
             #   st.error(f"Błąd Raport 2: {raport2.stderr}")   
            
            os.remove(tmp3_path)

            st.write("Wygenerowano raport LaTeX na podstawie wyników algorytmów.")
        else:
            st.write("Najpierw wygeneruj wyniki algorytmu")

