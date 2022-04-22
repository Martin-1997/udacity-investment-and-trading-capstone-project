# https://nbconvert.readthedocs.io/en/latest/execute_api.html

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

notebook_filename = "Untitled.ipynb"

with open(notebook_filename) as f:
    nb = nbformat.read(f, as_version=4)

print(f"Notebook opened successfully!")

try:
    ep = ExecutePreprocessor(timeout=None, kernel_name='python3', allow_errors=True) #  
    ep.preprocess(nb, {'metadata': {'path': './'}})

    print(f"Notebook porcessed successfully!")

    with open('executed_notebook.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

except RuntimeError as e:
    print("Runtime error occured:")
    print(str(e))

print("Script ended.")

