import json
import sys
import traceback
from pathlib import Path

nb_path = Path('notebooks/compare_article_vs_1024seq.ipynb')
if not nb_path.exists():
    print('Notebook not found:', nb_path)
    sys.exit(1)

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# collector for DataFrame-like objects to save
collected = {}

def display(obj, name=None):
    try:
        import pandas as _pd
        if isinstance(obj, _pd.DataFrame):
            n = name or f'df_{len(collected)}'
            collected[n] = obj
            print(f"[DISPLAY] DataFrame {n} shape={obj.shape}")
            # save CSV for inspection
            obj.to_csv(f"notebooks/outputs/{n}.csv", index=False)
            return
    except Exception:
        pass
    print(repr(obj))

# ensure outputs dir
Path('notebooks/outputs').mkdir(parents=True, exist_ok=True)

global_vars = {
    'np': __import__('numpy'),
    'pd': __import__('pandas'),
    'display': display,
}

# also import scipy functions lazily

errors = []
for i, cell in enumerate(nb.get('cells', []), start=1):
    if cell.get('cell_type') != 'code':
        continue
    src = cell.get('source', [])
    if isinstance(src, list):
        code = '\n'.join(src)
    else:
        code = src
    print(f'--- Executing cell {i} ---')
    print('--- CODE START ---')
    print(code)
    print('--- CODE END ---')
    try:
        exec(code, global_vars)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error in cell {i}: {e}\n{tb}")
        errors.append({'cell': i, 'error': str(e), 'traceback': tb})
        # stop execution on first error to allow targeted fixes
        break

# summary
print('\nExecution finished. Collected outputs:')
for k in collected:
    print('-', k, '->', f'notebooks/outputs/{k}.csv')

if errors:
    print('\nErrors occurred:')
    for e in errors:
        print(e['cell'], e['error'])
    sys.exit(2)
else:
    print('No errors')
    sys.exit(0)
