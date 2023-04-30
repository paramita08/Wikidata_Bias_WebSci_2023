import json
import pandas as pd
import numpy as np
import operator
def get_dataset(URL):
  dataset = pd.read_csv(URL,sep="\t",header=None)
  dataset.columns = ['subject', 'predicate', 'object']
  dataset=dataset.dropna()
  dataset=dataset.drop_duplicates()
  print(dataset.shape)
  return dataset
def get_professions(dataset):
  professions = list(set(dataset[dataset["predicate"] == "'occupation'"]["object"]))
  print(len(professions))
  return professions
def get_humans(dataset):
  humans = list(set(dataset[(dataset["predicate"] == "'instance of'") & (dataset["object"]=="'human'")]["subject"]))
  print(len(humans))
  return humans
def get_profession_and_gender_data(dataset):
  profession_data = dataset[dataset["predicate"] == "'occupation'"].reset_index().drop(columns = ["index"])
  gender_data = dataset[dataset["predicate"] == "'sex or gender'"].reset_index().drop(columns = ["index"])
  human_data=dataset[(dataset["predicate"] == "'instance of'") & (dataset["object"]=="'human'")].reset_index().drop(columns = ["index"])
  profession_data.columns = ["subject", "profession_rel", "profesion"]
  gender_data.columns = ["subject", "gender_rel", "gender"]
  human_data.columns = ["subject", "instance", "human"]
  profession_data = profession_data.drop_duplicates()
  gender_data = gender_data.drop_duplicates()
  human_data=human_data.drop_duplicates()
  profession_and_human_data = pd.merge(profession_data, human_data, on = "subject", how = "inner")
  profession_and_gender_data = pd.merge(profession_and_human_data, gender_data, on = "subject", how = "inner")
  return profession_and_gender_data , profession_and_human_data
def get_humans_with_one_prof(profession_and_human_data):
  humans_with_one_prof = list(set(profession_and_human_data["subject"]))
  print(len(humans_with_one_prof))
  return humans_with_one_prof
def get_male_and_female_entities(profession_and_gender_data):
  male_total_list=list(set(profession_and_gender_data[profession_and_gender_data["gender"] == "'male'"]["subject"].values))
  female_total_list=list(set(profession_and_gender_data[profession_and_gender_data["gender"] == "'female'"]["subject"].values))
  print(len(male_total_list))
  print(len(female_total_list))
  return male_total_list,female_total_list
def get_professions_count(professions,profession_and_gender_data):
    profession_count = {}
    male_profession_count={}
    female_profession_count={}
    male_profession_count_list=[]
    female_profession_count_list=[]
    profession_set=set()
    k=0
    for i in professions:
      profession_set.add(i)
      profession_count[i] = profession_and_gender_data[profession_and_gender_data["profesion"] == i].shape[0]
      count=profession_and_gender_data[(profession_and_gender_data["profesion"] == i) & (profession_and_gender_data["gender"] == "'male'")].shape[0]
      male_profession_count[i]=count
      male_profession_count_list.append(count)
      count=profession_and_gender_data[(profession_and_gender_data["profesion"] == i) & (profession_and_gender_data["gender"] == "'female'")].shape[0]
      female_profession_count[i]=count
      female_profession_count_list.append(count)
    return profession_count,male_profession_count,female_profession_count
def get_filtered_profession_set(professions,male_profession_count_sort,female_profession_count_sort,c1,c2):
    profession_set_male=set(professions)
    print(len(professions))
    for i in male_profession_count_sort:
        if i[1]<c1:
          profession_set_male.remove(i[0])
    print(len(profession_set_male))
    profession_set_female=set(professions)
    for i in female_profession_count_sort:
        if i[1]<c2:
          profession_set_female.remove(i[0])
    print(len(profession_set_female))
    return profession_set_female,profession_set_male
URL = '../../HULK/KG_bias/bias_calculation/demographics_All.tsv'
wikidata = pd.read_csv(URL,sep="\t",header=None)
wikidata.columns = ['subject', 'predicate', 'object']
wikidata =wikidata.dropna()
print("read wikidata")
URL="../../HULK/KG_bias/bias_calculation/Demographies/All/Demographies/India/India_final.tsv"
dataset=get_dataset(URL)
professions1=get_professions(dataset)
professions2=get_professions(wikidata)
professions = list(set(professions1) & set(professions2))
print(len(professions))
profession_and_gender_data, profession_and_human_data=get_profession_and_gender_data(dataset)
for i in  ["'terrorist'","'boxer'" ,"'racing automobile driver'"]:
  count=profession_and_gender_data[(profession_and_gender_data["profesion"] == i) & (profession_and_gender_data["gender"] == "'female'")]["subject"].values  
  print(count)
#humans_with_one_prof = get_humans_with_one_prof(profession_and_human_data)
#with open("Russia/humans_filtered.json","w+") as f:
#  json.dump(humans_with_one_prof,f)
profession_count_1,male_profession_count_1,female_profession_count_1=get_professions_count(professions,profession_and_gender_data)
male_profession_count_sort_1= sorted(male_profession_count_1.items(), key=operator.itemgetter(1), reverse =True)
female_profession_count_sort_1= sorted(female_profession_count_1.items(), key=operator.itemgetter(1), reverse =True)
profession_set_female_1,profession_set_male_1 = get_filtered_profession_set(professions,male_profession_count_sort_1,female_profession_count_sort_1,0,0)
profession_set_1=list(set(profession_set_female_1) & set(profession_set_male_1))
print(len(profession_set_1))
with open("../Professions/India_professions_f.json","w+") as f:
     json.dump(profession_set_1,f)	
