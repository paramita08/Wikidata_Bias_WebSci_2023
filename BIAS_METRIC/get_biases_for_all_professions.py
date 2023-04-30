from ampligraph.utils import save_model, restore_model
import json
import torch
from grad_descent_for_bias import model_c
import torch.optim as optim
from grad_descent_for_bias import load_ckp
import numpy as np

def get_biases(l, r, human2id_path, ckpt_path, professions_path, embeddings_path,dimension):
  f = open(human2id_path, "r")
  human2id = json.load(f)
  dummy_hum_embeds = torch.zeros(len(human2id), dimension)
  dummy_gmf_embeds = torch.zeros(3, dimension)
  model = model_c(len(human2id), 3,"cuda", dummy_hum_embeds, dummy_gmf_embeds, dim = dimension)
  optimizer = optim.SGD(model.parameters(), lr=0.01)
  loaded_model, loaded_optimizer, start_epoch, train_loss_min = load_ckp(ckpt_path, model, optimizer)
  updated_human_embeds = loaded_model.human_embeds.weight
  print(updated_human_embeds.size())
  print(updated_human_embeds[0])

  model = restore_model(embeddings_path)
  original_human_embeds = torch.tensor(model.get_embeddings(list(human2id.keys()), embedding_type = 'entity'))
  print(original_human_embeds.size())
  print(original_human_embeds[0])

  profession_emb = torch.tensor(model.get_embeddings("'occupation'", embedding_type = 'relation'))

  f = open(professions_path, "r")
  professions = json.load(f)
  biases = {}
  cnt = 0
  for occ in professions[l:r]:
    try:
      occ_emb = torch.tensor(model.get_embeddings(occ, embedding_type = 'entity'))
      grad_p = (updated_human_embeds + profession_emb - occ_emb).norm(p = 1, dim = 1) - (original_human_embeds + profession_emb - occ_emb).norm(p = 1, dim = 1)
      bias_score = torch.mean(grad_p)
      biases[occ] = bias_score
      cnt += 1
      print("total cnt of biases", cnt)
    except IndexError:
      print("index out of range")
      pass

  return biases
