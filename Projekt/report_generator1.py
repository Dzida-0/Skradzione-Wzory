import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import subprocess

class ReportGenerator:
    def __init__(self, tex_pattern='report_pattern1.tex', output_folder="reports"):
        self.tex_pattern = tex_pattern
        self.output_folder = output_folder
        self.output_file_name = "report1.tex"
        self.report_latex_exists = False
        os.makedirs(self.output_folder, exist_ok=True)

    def generate_latex_report(self, data):
      to_put = {
          "date": time.strftime("%d.%m.%Y", time.localtime(data['date'])),
          "hour": time.strftime("%H:%M", time.localtime(data['date'])),
          "file_name": data['file'].replace('_', '\\_').replace("\\", "/"),
          "exec_time": data['exec_time'],
          "proc_plagiat": 100*(sum(data['numbers_tekst']))/ (len(data['numbers_tekst'])),

      }

    # Generate dynamic content for similarity scores
      score_lines_tekst = ""
      score_lines_mat = ""
      for i, score in enumerate(data['numbers_tekst'], start=1):
            score_lines_tekst += f"\\item Part {i}: {score * 100:.2f}% \n"
      for i, score in enumerate(data['numbers_mat'], start=1):
            score_lines_mat += f"\\item Part {i}: {score * 100:.2f}% \n"

    # Replace placeholders in the LaTeX template
      current_dir = os.path.dirname(os.path.abspath(__file__))
      tex_path = os.path.join(current_dir, self.tex_pattern)
      print(tex_path)

      with open(tex_path, 'r') as f:
            tex = f.read()

      for key, value in to_put.items():
          tex = tex.replace(f"Input-{key}", str(value))

    # Replace detailed similarity scores dynamically
      tex = tex.replace("TEXT", score_lines_tekst.strip())
      tex = tex.replace("FORMULAS", score_lines_mat.strip())

      output_path = os.path.join(current_dir, self.output_folder, self.output_file_name)
      with open(output_path, 'w') as f:
         f.write(tex)

      print(f"LaTeX report saved to {output_path}")
      self.output_folder=os.path.join(current_dir, self.output_folder)
      try:
            result = subprocess.run(
                ["pdflatex","-interaction=nonstopmode", "-halt-on-error", self.output_file_name],
                cwd= self.output_folder,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("Standard Output:", result.stdout)
            print("Standard Error:", result.stderr)
      except subprocess.CalledProcessError as e:
            print(f"Błąd podczas uruchamiania pdflatex: {e}")
            print("Standard Output:", e.stdout)
            print("Standard Error:", e.stderr)
            
      except subprocess.CalledProcessError as e:
            print(f"Error occurred during LaTeX compilation: {e}")     
      self.report_latex_exists = True

    def open_latex_report(self):
        try:
            os.startfile(os.path.join(self.output_folder,os.path.splitext(self.output_file_name)[0] + ".pdf"))
        except Exception as e:
            print(f"Failed to open PDF: {e}")
