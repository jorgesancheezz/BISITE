import csv
import random
import os

random_seed = 42
n_per_class = 300

in_csv = os.path.join('PULSOVITAL', 'npy_output', 'processed_records.csv')
out_csv = os.path.join('PULSOVITAL', 'npy_output', 'selected_records_600.csv')

records = []
with open(in_csv, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        records.append(r)

nsr = [r for r in records if r.get('tag','').upper()=='NSR']
af = [r for r in records if r.get('tag','').upper()=='AF']

random.seed(random_seed)
sel_nsr = random.sample(nsr, min(n_per_class, len(nsr)))
sel_af = random.sample(af, min(n_per_class, len(af)))
selected = sel_nsr + sel_af

with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['record','tag','out_folder']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in selected:
        writer.writerow({'record': r.get('record',''), 'tag': r.get('tag',''), 'out_folder': r.get('out_folder','')})

print(f"Input: {in_csv}")
print(f"Found NSR: {len(nsr)}, AF: {len(af)}")
print(f"Selected NSR: {len(sel_nsr)}, AF: {len(sel_af)} -> written to {out_csv}")
