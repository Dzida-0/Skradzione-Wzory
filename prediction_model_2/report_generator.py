import time
import zipfile
import subprocess
import os
import matplotlib.pyplot as plt
import seaborn as sns
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
import webbrowser
import shutil
import numpy as np


class ReportGenerator:
    def __init__(self):
        self.report_latex_exists = False
        self.report_html_exists = False
        self.tex_pattern = 'report_pattern.tex'
        self.tex_output_folder = "reports"
        self.tex_png_output_folder = "elements"
        self.tex_out_file_name = "report.tex"
        self.html_pattern = 'report_pattern.ipynb'
        self.html_out_file_name = "report.html"

    def generate_plots_for_latex(self, data):
        plt.plot([i for i in range(len(data))], data)
        plt.savefig(os.path.join(self.tex_output_folder, self.tex_png_output_folder, "plot0.png"))
        ###
        plt.figure(figsize=(8, 6))
        sns.histplot(data, bins=8, kde=True, color='blue')
        plt.xlim(min(data), max(data))
        plt.xlabel('a', fontsize=12)
        plt.ylabel('b', fontsize=12)
        plt.title('c', fontsize=14)
        plt.savefig(os.path.join(self.tex_output_folder, self.tex_png_output_folder, "plot1.png"))

    def generate_latex_report(self, data):
        to_put = {
            "date": time.strftime("%d.%m.%Y", time.localtime(data['date'])),
            "hour": time.strftime("%H:%M", time.localtime(data['date'])),
            "file_name": data['file'].replace('_', '\\_'),
            "exec_time": data['exec_time'],
            "proc_plagiat": sum((data['numbers'])) / len(data['numbers'])
        }
        self.report_latex_exists = False
        with open(self.tex_pattern, 'r') as f:
            tex = f.read()
        for input, text in to_put.items():
            tex = tex.replace(f"Input-{input}", str(text))
        for i in range(2):
            tex = tex.replace(f"Plot-nr-{i}", self.tex_png_output_folder + f"/plot{i}.png")
        os.makedirs(os.path.join(self.tex_output_folder, self.tex_png_output_folder), exist_ok=True)
        self.generate_plots_for_latex(data['numbers'])
        with open(os.path.join(self.tex_output_folder, self.tex_out_file_name), 'w') as f:
            f.write(tex)
        # To PDF
        try:
            result = subprocess.run(
                ["pdflatex", self.tex_out_file_name],
                cwd=self.tex_output_folder,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error occurred during LaTeX compilation: {e}")
        self.report_latex_exists = True

    def generate_html_report(self, data):

        # ["24.01.2025","13:34","test.tex","green", "40.0", "123", "54" ],
        with open(self.html_pattern, 'r', encoding='utf-8') as f:
            notebook_content = nbformat.read(f, as_version=4)

        data_to_pass = {"data": {
            "date": time.strftime("%d.%m.%Y", time.localtime(data['date'])),
            "hour": time.strftime("%H:%M", time.localtime(data['date'])),
            "file_name": data['file'],
            "exec_time": data['exec_time'],
            "proc_plagiat": sum((data['numbers'])) / len(data['numbers']),
            "row": data['row']
        },
            "numbers": data['numbers']}
        notebook_content.cells[1].source = f"data = {data_to_pass}"

        execute_preprocessor = ExecutePreprocessor(timeout=600, kernel_name='python3')
        try:
            execute_preprocessor.preprocess(notebook_content, {'metadata': {'path': './'}})
        except Exception as e:
            print(f"Error during execution: {e}")
            return

        html_exporter = HTMLExporter()
        html_exporter.exclude_input = True
        html_exporter.exclude_input_prompt = True  #

        (html_body, _) = html_exporter.from_notebook_node(notebook_content)
        os.makedirs(self.tex_output_folder, exist_ok=True)
        with open(os.path.join(self.tex_output_folder, self.html_out_file_name), 'w', encoding='utf-8') as f:
            f.write(html_body)

    def open_latex_report(self):
        try:
            os.startfile(os.path.join(self.tex_output_folder, os.path.splitext(self.tex_out_file_name)[0] + ".pdf"))
        except Exception as e:
            print(f"Failed to open PDF: {e}")

    def open_html_report(self):
        webbrowser.open(os.path.join(self.tex_output_folder, self.html_out_file_name))

    def download_latex_report(self, path):
        if self.report_latex_exists:
            try:
                folder_to_zip = self.tex_png_output_folder
                folder_path = os.path.join(self.tex_output_folder, folder_to_zip)
                with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in [self.tex_out_file_name, os.path.splitext(self.tex_out_file_name)[0] + ".pdf"]:
                        full_file_path = os.path.join(self.tex_output_folder, file_path)
                        if os.path.exists(full_file_path):
                            zip_element = os.path.relpath(full_file_path, start=self.tex_output_folder)
                            zipf.write(full_file_path, zip_element)
                    if os.path.exists(folder_path):
                        for root, _, files in os.walk(folder_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                zip_element = os.path.relpath(file_path, start=self.tex_output_folder)
                                zipf.write(file_path, zip_element)
            except Exception as e:
                print(f"Error while zipping folder: {e}")

    def download_html_report(self, path):
        shutil.copy(os.path.join(self.tex_output_folder, self.html_out_file_name), path)
