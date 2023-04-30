import json
import pandas as pd
import torch
import numpy as np
import torch
from ampligraph.utils import save_model, restore_model

def preprocess(dataset_path, model_path,entire_dataset_path):
  col_names = ["head" ,"rel", "tail"]
  df = pd.read_csv(entire_dataset_path,sep="\t",names=col_names,header =None)
  df=df.dropna()
  df=df.drop_duplicates()
  all_human_entities1 = list(set(df[(df["rel"]=="'instance of'") & (df["tail"]=="'human'")]["head"].values))
  dataset = pd.read_csv(dataset_path,sep="\t",names=col_names,header =None)
  dataset=dataset.dropna()
  dataset=dataset.drop_duplicates()
  human_entities = list(set(dataset[(dataset["rel"]=="'instance of'") & (dataset["tail"]=="'human'")]["head"].values))
  all_human_entities=list(set(all_human_entities1)&set(human_entities))
  print(len(all_human_entities))
  print(len(human_entities))
  model = restore_model(model_path)
  human_embeddings = torch.tensor(model.get_embeddings(all_human_entities, embedding_type = 'entity'))
  male_emb = torch.tensor(model.get_embeddings("'male'", embedding_type = 'entity'))
  female_emb = torch.tensor(model.get_embeddings("'female'", embedding_type = 'entity'))
  gender_emb = torch.tensor(model.get_embeddings("'sex or gender'", embedding_type = 'relation'))
  all_human_dataset = pd.DataFrame(all_human_entities, columns=["human"])
  gmf2id = {"gender":0, "male": 1, "female":2 }
  human2id = {j: i for i, j in enumerate(all_human_entities)}
  gmf_embeddings = torch.stack((gender_emb, male_emb, female_emb), dim = 0)

  return all_human_dataset, human2id, human_embeddings, gmf2id, gmf_embeddings
