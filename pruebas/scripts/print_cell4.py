import json
nb=json.load(open('notebooks/compare_article_vs_1024seq.ipynb','r',encoding='utf-8'))
print('CELL 4 SOURCE:')
for i,line in enumerate(nb['cells'][3]['source'],start=1):
    print(i,repr(line))
