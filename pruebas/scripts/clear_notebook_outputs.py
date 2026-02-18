import nbformat
import sys
from pathlib import Path
p = Path(r'c:\Users\BISITE-NEL\Desktop\pruebas\compare_synthetic_vs_real_report.ipynb')
nb = nbformat.read(str(p), as_version=4)
for cell in nb.cells:
    if 'outputs' in cell:
        cell['outputs'] = []
    if 'execution_count' in cell:
        cell['execution_count'] = None
nbformat.write(nb, str(p))
print('Cleared outputs for', p)
