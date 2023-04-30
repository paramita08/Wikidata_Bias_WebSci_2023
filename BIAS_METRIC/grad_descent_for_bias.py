import os
import json
import pickle
import argparse
from collections import Counter
import pdb
from torch.utils import data
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
##from torch.utils import tensorboard
class FB15KDataset(data.Dataset):
    """Dataset implementation for handling FB15K and FB15K-237."""
    def __init__(self, data, human_to_idx, gmf_to_idx):
        self.human2id = human_to_idx
        self.gmf2id = gmf_to_idx
        self.data = [row["human"] for index, row in data.iterrows()]

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.data)

    def __getitem__(self, index):
        gender_id = self._to_idx("gender", self.gmf2id)
        male_id = self._to_idx("male", self.gmf2id)
        female_id = self._to_idx("female", self.gmf2id)
        human = self.data[index]
        human_id = self._to_idx(human, self.human2id)
        return gender_id, male_id, female_id, human_id

    @staticmethod
    def _to_idx(key, mapping_dict):
        try:
            return mapping_dict[key]
        except KeyError:
            return len(mapping)


class model_c(nn.Module):

    def __init__(self, entity_count, relation_count, device, human_embeddings, gmf_embeddings, norm=1, dim=100):
        super(model_c, self).__init__()
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.device = device
        self.norm = norm
        self.dim = dim
        self.human_embeds = nn.Embedding.from_pretrained(human_embeddings, freeze = False)
        self.gmf_embeds = nn.Embedding.from_pretrained(gmf_embeddings, freeze = True)

    def forward(self, male_triplets, female_triplets, relative):
        if relative == "male":
            negative_m_theta = self._distance(female_triplets) - self._distance(male_triplets)
        if relative == "female":
            negative_m_theta = self._distance(male_triplets) - self._distance(female_triplets)
        return negative_m_theta, self._distance(male_triplets), self._distance(female_triplets)

    def _distance(self, triplets):
        assert triplets.size()[1] == 3
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]
        
        e_s = self.human_embeds(heads)
        e_p = self.gmf_embeds(relations)
        e_o = self.gmf_embeds(tails)

        #return (self.human_embeds(heads) + self.gmf_embeds(relations) - self.gmf_embeds(tails)).norm(p=self.norm,dim=1)
        return torch.sum(e_s * e_p * e_o, 1)

def save_ckp(state, is_best, checkpoint_path):
    f_path = checkpoint_path
    torch.save(state, f_path)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['train_loss_min']
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

def train(seed, human2id, gmf2id, human_embeddings, gmf_embeddings, train_df, batch_size, use_gpu, learning_rate, checkpoint_path, dimension, epochs, best_loss_input, ifSave, relative):
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda') if use_gpu else torch.device('cpu')

    train_set = FB15KDataset(train_df, human2id, gmf2id)
    train_generator = data.DataLoader(train_set, batch_size=batch_size)
    model = model_c(len(human2id), len(gmf2id), device, human_embeddings, gmf_embeddings, dim = dimension)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    human_embeddings =  human_embeddings.to(device)
    start_epoch_id = 1
    step = 0
    best_loss = best_loss_input
    prev_loss = 0.0
    '''
    train_losses = []
    male_distances_list = []
    female_distances_list = []
    one_entity_male_distances = []
    one_entity_female_distances = []
    '''
    for epoch_id in range(start_epoch_id, epochs + 1):
        print("Starting epoch: ", epoch_id)
        train_loss = 0.0
        #accu_male_distances = 0.0
        #accu_female_distances = 0.0
        model.train()

        for batch_id, (gender_batch, male_batch, female_batch, human_batch) in enumerate(train_generator):
            male_triplets = torch.stack((human_batch, gender_batch, male_batch), dim = 1)
            female_triplets = torch.stack((human_batch, gender_batch, female_batch), dim = 1)
            male_triplets = male_triplets.to(device)
            female_triplets = female_triplets.to(device)

            optimizer.zero_grad()

            negative_m_theta, male_distances, female_distances = model(male_triplets, female_triplets, relative)
            negative_m_theta.mean().backward()

            ##print(negative_m_theta)

            optimizer.step()
            step += 1

            ##train_loss += ((1 / (batch_id + 1)) * (negative_m_theta.mean().data - train_loss))
            train_loss += (negative_m_theta.mean().data)
            #accu_male_distances += (male_distances.mean().data)
            #accu_female_distances += (female_distances.mean().data)

        train_loss = train_loss/len(train_generator)
        '''
        accu_male_distances = accu_male_distances/len(train_generator)
        accu_female_distances = accu_female_distances/len(train_generator)

        train_losses.append(train_loss.cpu().numpy().reshape(1,1).item())
        male_distances_list.append(accu_male_distances.cpu().numpy().reshape(1,1).item())
        female_distances_list.append(accu_female_distances.cpu().numpy().reshape(1,1).item())

        with torch.no_grad():
          x = model._distance(torch.tensor([[0, 0, 1]]).to(device))
          y = model._distance(torch.tensor([[0, 0, 2]]).to(device))
        one_entity_male_distances.append(x)
        one_entity_female_distances.append(y)
        '''
        print("The train loss is", train_loss.item())

        checkpoint = {
            'epoch': epoch_id + 1,
            'train_loss_min': train_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        if best_loss >= train_loss:
          best_loss = train_loss
          if ifSave == True:
            save_ckp(checkpoint, True, checkpoint_path)

        if (prev_loss - train_loss) == 0:
           break
        prev_loss = train_loss
        #print(torch.equal(model.human_embeds.weight, human_embeddings))

    return model
