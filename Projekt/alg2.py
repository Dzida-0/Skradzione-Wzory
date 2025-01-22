from model2 import Model
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mod = Model()

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "saved_models", "test_generate_21_f_256_1024_o256_e20")
mod.load_model(model_path)

f1 = sys.argv[1]

d = mod.predict_from_latex(f1)
print(f'{((sum(d.values())/len(d))*100):.2f}')

cos=r"C:\Users\NowyLOMBARD\Desktop\Kasia\Studia\Inżynieria\Projekt\saved_models\test_generate_21_f_256_1024_o256_e20.h5"