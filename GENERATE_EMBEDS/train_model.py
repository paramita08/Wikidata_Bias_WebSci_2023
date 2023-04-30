from ampligraph.evaluation import train_test_split_no_unseen
from ampligraph.latent_features import TransE,ComplEx,DistMult
from ampligraph.utils import save_model, restore_model
from ampligraph.evaluation import evaluate_performance, mr_score, hits_at_n_score, mrr_score
import pandas as pd
import numpy as np
import random

def display_aggregate_metrics(ranks):
    print('Mean Rank:', mr_score(ranks))
    print('Mean Reciprocal Rank:', mrr_score(ranks))
    print('Hits@1:', hits_at_n_score(ranks, 1))
    print('Hits@10:', hits_at_n_score(ranks, 10))
    print('Hits@20:', hits_at_n_score(ranks, 20))

def train_model(model_name,path_to_dataset,e,saved_model_path):
  columns = ['subject', 'predicate', 'object']
  dataset = pd.read_csv(path_to_dataset, sep = "\t", names = columns, header = None)
  dataset = dataset.dropna()

  X_train, X_test = train_test_split_no_unseen(dataset.values,10000, seed=0)

  print(len(X_train))
  print(len(X_test))
  if model_name=="TransE":
     model =TransE(k=100, epochs=e, eta=10, loss='multiclass_nll',
                  initializer='xavier', initializer_params={'uniform': False},
                  regularizer='LP', regularizer_params= {'lambda': 0.0001, 'p': 3},
                  optimizer= 'sgd', optimizer_params= {'lr': 0.0001},
                  seed= 0, batches_count= 100, verbose=True)
  if model_name=="ComplexE":
     model = ComplEx(k=100, eta=10, epochs=e, batches_count=256, seed=0,
         optimizer='sgd', optimizer_params={'lr': 0.01}, loss=Loss, loss_params={},
         regularizer=Regularizer, regularizer_params=reg_params, initializer='xavier', initializer_params={'uniform': False}, verbose=True)
  if model_name=="DistMult":
     model=DistMult(k=100, eta=10, epochs=e, batches_count=100, seed=0,
        embedding_model_params={'corrupt_sides': ['s,o'], 'negative_corruption_entities': 'all', 'normalize_ent_emb': False},
        optimizer='sgd', optimizer_params={'lr': 0.0005}, loss='nll', loss_params={},
        regularizer=None, regularizer_params={}, initializer='xavier',
        initializer_params={'uniform': False}, verbose=False)

  model.fit(X_train)
  save_model(model, saved_model_path)
 

def eval_model(saved_model_path, path_to_dataset,  n):
  model = restore_model(saved_model_path)

  columns = ['subject', 'predicate', 'object']
  dataset = pd.read_csv(path_to_dataset, sep = "\t", names = columns, header = None)
  dataset.columns=columns
  dataset = dataset.dropna()
  X_train, X_test = train_test_split_no_unseen(dataset.values, 500, seed=0)
  X_filter=np.concatenate([X_train,X_test],0)
  corruption_subset_o = random.sample(list(dataset["object"].values), n)
  corruption_subset_s = random.sample(list(dataset["subject"].values), n)
  print(X_test.shape)
  ranks_o = evaluate_performance(X_test,
                             model=model,
                             filter_triples=X_filter,
                             corrupt_side = 'o',entities_subset=corruption_subset_o)
  ranks_s = evaluate_performance(X_test,
                             model=model,
                             filter_triples=X_filter,
                             corrupt_side = 's',entities_subset=corruption_subset_s)


  print("object side corruption")
  display_aggregate_metrics(ranks_o)

  print("subject side corruption")
  display_aggregate_metrics(ranks_s)