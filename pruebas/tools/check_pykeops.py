import importlib
try:
    m = importlib.import_module('pykeops')
    print('pykeops version:', getattr(m, '__version__', 'unknown'))
except Exception as e:
    print('pykeops import failed:', repr(e))
