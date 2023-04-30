import pandas as pd
import csv
def isnan(string):
    return string!=string
df=pd.read_csv("All_triples.tsv",sep="\t",header=None)
df.columns=["node1","label","node2"]
node1=df['node1']
label_=df['label']
node2=df['node2']
entities_mapping=dict()
properties_mapping=dict()
for (i,j,k) in zip(node1,label_,node2):
    entities_mapping[i]=i
    entities_mapping[k]=k
    properties_mapping[j]=j
col_names=["id","label"]
df=pd.read_csv("entities.tsv",sep="\t",header=None)
df.columns=col_names
id=df['id']
label=df['label']
for i,j in zip(id,label):
    if(isnan(j)==False and j[-3:][0]=="@"):
      #print(j)
      n=len(j)
      j=j[:-3]
    if(j=="Not found"):
      j=""
    entities_mapping[i]=j
df=pd.read_csv("properties.tsv",sep="\t",header=None)
df.columns=col_names
id=df['id']
label=df['label']
for i,j in zip(id,label):
    properties_mapping[i]=j
print(len(entities_mapping))
print(len(properties_mapping))

with open('demo_final.tsv', 'wt') as out_file:
  tsv_writer = csv.writer(out_file, delimiter='\t')
  for (i,j,k) in zip(node1,label_,node2):
      tsv_writer.writerow([entities_mapping[i],properties_mapping[j],entities_mapping[k]])