import pandas as pd
import numpy as np
import csv
mapping =dict()
def isnan(string):
    if(string!=string):
       return True
r =pd.read_csv("All_triples.tsv",sep = '\t')
Q =r["node1"]
P= r["label"]
r=pd.read_csv("rhs.tsv",sep='\t')
Q2=r["node2"]
Qnodes =set()
Pnodes =set()
C=0
for i in Q:
    Qnodes.add(i)
    mapping[i]="Not found"
print(len(Qnodes))
for i in P:
    Pnodes.add(i)
    mapping[i]="Not found"
print(len(Pnodes))
for i in Q2:
    Qnodes.add(i)
    mapping[i]="Not found"
print(len(Qnodes))
r=pd.read_csv("../nodefile.tsv",sep='\t')
value =r["id"]
ids=r["label"]
for (i,j) in zip(value,ids):
    if(isnan(j)):
       C=C+1
    mapping[i]=j
print(C)
with open('entities.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(["id","value"])
    for i in Qnodes:
        tsv_writer.writerow([i,mapping[i]])
with open('properties.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(["id","value"])
    for i in Pnodes:
        tsv_writer.writerow([i,mapping[i]])