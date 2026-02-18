import os
reals=[]
synt=[]
others=[]
root='.'
for dirpath,dirnames,files in os.walk(root):
    for f in files:
        if not f.lower().endswith('.npy'): continue
        p=os.path.join(dirpath,f)
        path_norm=p.replace('\\','/')
        name=f.lower()
        # heuristics
        if '/npy_output/' in path_norm or path_norm.startswith('./p10') or '/p10/' in path_norm or '/p100' in path_norm:
            reals.append(p)
            continue
        if '1024seq' in name or 'synth' in name or 'synt' in name or ('metricas' in path_norm and (('af' in name) or ('nsr' in name))):
            synt.append(p)
            continue
        # common generated dataset names
        if name.endswith('1024x3000x1.npy') or 'processed' in name or 'selected' in name:
            # likely real (generated from real WFDB records)
            reals.append(p)
            continue
        # fallback: treat files inside PULSOVITAL/Metricas that contain 'seq' as synthetic
        if '/PULSOVITAL/Metricas/'.lower() in path_norm.lower() and 'seq' in name:
            synt.append(p)
            continue
        others.append(p)
# print summary
print('REAL_COUNT:', len(reals))
for x in reals[:200]: print(x)
print('\nSYNTH_COUNT:', len(synt))
for x in synt[:200]: print(x)
print('\nOTHER_COUNT:', len(others))
for x in others[:200]: print(x)
# Optionally save to CSV
out='compare_out_test/npy_classification.csv'
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out,'w',encoding='utf-8') as fh:
    fh.write('path,classification\n')
    for x in reals: fh.write(f'"{x}",real\n')
    for x in synt: fh.write(f'"{x}",synthetic\n')
    for x in others: fh.write(f'"{x}",other\n')
print('\nWROTE', out)
