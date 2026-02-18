import csv,glob,os
files=sorted([f for f in glob.glob('**/*_summary.csv', recursive=True)])
if not files:
    print('No summary CSVs found')
    raise SystemExit(0)
# Column names derived from file basename without extension
cols=[]
metric_values={} # metric -> {col: value}
for f in files:
    name=os.path.splitext(os.path.basename(f))[0]
    col=name
    cols.append(col)
    try:
        with open(f, newline='', encoding='utf-8') as fh:
            reader=csv.reader(fh)
            for row in reader:
                if not row: continue
                metric=row[0]
                # value: if row length>1, take second column; if more, join with '|'
                val=''
                if len(row)>=2:
                    if len(row)==2:
                        val=row[1]
                    else:
                        val='|'.join(row[1:])
                metric_values.setdefault(metric, {})[col]=val
    except Exception as e:
        print('Error reading',f,e)
# Write combined CSV
out='compare_out_test/combined_all_metrics.csv'
os.makedirs(os.path.dirname(out), exist_ok=True)
all_metrics=sorted(metric_values.keys())
with open(out,'w',newline='',encoding='utf-8') as fh:
    writer=csv.writer(fh)
    writer.writerow(['metric']+cols)
    for m in all_metrics:
        row=[m]
        for c in cols:
            row.append(metric_values.get(m,{}).get(c,''))
        writer.writerow(row)
print('WROTE',out)
# print head
with open(out,'r',encoding='utf-8') as fh:
    for i,line in enumerate(fh):
        if i<40:
            print(line.rstrip())
        else:
            break
