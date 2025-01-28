import matplotlib.pyplot as plt
from report_generator2 import ReportGenerator
from model2 import Model
import sys
import os

mod=Model()

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "saved_models", "test_generate_21_f_256_1024_o256_e20")
mod.load_model(model_path)
f1=sys.argv[1]
name=sys.argv[2]
n=int(sys.argv[3])

f1=f1.replace('\\','/')
#print(f1)

d = mod.predict_from_latex(f1,name, n)

r = ReportGenerator()
r.generate_latex_report(mod.get_report_data())
r.open_latex_report()

#r.generate_html_report(mod.get_report_data())
#r.open_html_report()