import csv, json, glob, os

def read_kv_csv(path):
    out={}
    with open(path, encoding='utf-8') as fh:
        for row in csv.reader(fh):
            if not row: continue
            if row[0].strip().lower()=='metric': continue
            k=row[0].strip()
            v=','.join(row[1:]).strip() if len(row)>1 else ''
            out[k]=v
    return out

def read_json(path):
    with open(path, encoding='utf-8') as fh:
        j=json.load(fh)
    out={}
    def flatten(prefix, obj):
        if isinstance(obj, dict):
            for k,v in obj.items():
                flatten(f"{prefix}{k}_", v)
        else:
            out[prefix[:-1]]=str(obj)
    flatten('', j)
    return out

# candidate files
pairs={'AF':{}, 'NSR':{}}
# quick CSVs
af_quick = glob.glob('compare_out_quick_AF/*quick_summary.csv')
nsr_quick = glob.glob('compare_out_quick_NSR/*quick_summary.csv')
# best jsons
af_best = glob.glob('compare_out_best_AF/*best_summary.json')
nsr_best = glob.glob('compare_out_best_NSR/*best_summary.json')
# processed summaries (more detailed) if exist
af_proc = glob.glob('compare_out_processed/*AF*_summary.csv')
nsr_proc = glob.glob('compare_out_processed_NSR/*_summary.csv') + glob.glob('compare_out_processed/*NSR*_summary.csv')

if af_proc:
    pairs['AF']['processed']=af_proc[0]
if nsr_proc:
    pairs['NSR']['processed']=nsr_proc[0]
if af_quick:
    pairs['AF']['quick']=af_quick[0]
if nsr_quick:
    pairs['NSR']['quick']=nsr_quick[0]
if af_best:
    pairs['AF']['best']=af_best[0]
if nsr_best:
    pairs['NSR']['best']=nsr_best[0]

# load metrics preferring processed > best > quick
metrics_by_group={'AF':{}, 'NSR':{}}
for tag in ['AF','NSR']:
    # processed
    if 'processed' in pairs[tag]:
        try:
            metrics_by_group[tag].update(read_kv_csv(pairs[tag]['processed']))
        except Exception:
            pass
    # best json
    if 'best' in pairs[tag]:
        try:
            metrics_by_group[tag].update(read_json(pairs[tag]['best']))
        except Exception:
            pass
    # quick csv
    if 'quick' in pairs[tag]:
        try:
            metrics_by_group[tag].update(read_kv_csv(pairs[tag]['quick']))
        except Exception:
            pass

# union of keys
all_keys=set()
for v in metrics_by_group.values():
    all_keys.update(v.keys())
all_keys = sorted(all_keys)

out_dir='compare_out_test'
os.makedirs(out_dir, exist_ok=True)
out_path=os.path.join(out_dir,'final_metrics_AF_NSR.csv')
with open(out_path,'w',newline='',encoding='utf-8') as fh:
    writer=csv.writer(fh)
    writer.writerow(['metric','AF','NSR'])
    for k in all_keys:
        a = metrics_by_group['AF'].get(k,'')
        n = metrics_by_group['NSR'].get(k,'')
        writer.writerow([k,a,n])
print('WROTE', out_path)
# print head
with open(out_path, encoding='utf-8') as fh:
    for i,line in enumerate(fh):
        if i<200:
            print(line.strip())
        else:
            break
