# Diversity matters: Robustness of bias measurements in Wikidata
The work accepted at ```ACM WebSci 2023``` is contributed by Paramita Das, Sai Keerthana Karnam, Anirban Panda, Bhanu Prakash Reddy Guda, Soumya Sarkar and Animesh Mukherjee. 
## Dataset Preperation
------------------
We used [kgtk](https://kgtk.readthedocs.io/en/latest/) to prepare the dataset. All the required files are added in DATASET folder. 
```
wiki_dump = dataset_path
kgtk import-wikidata -i $wiki_dumo --node nodefile.tsv --edge edgefile.tsv --qual qualfile.tsv

# **Node file containes the labels of all entities and relations in the wikidata**
# Edge file contains all the triples in the wikidata along with their descriptions. We remove all the columns and store only the head,relation and tail entity. This file is saved as Final_triples_QP.tsv

```
### Collection of Demography datasets


First we extract all the humans from the entire dataset
```
# P31 - instance of and Q5 - human
kgtk filter -i Final_triples_QP.tsv -p ";P31;Q5" > humans.tsv
```
Following are the commands to extract each demographic dataset. Let us say Qnode_demo is the Qnode entity of a particular demography. The entities and relations whose labels are not present in nodefile.tsv are found using [qwikidata](https://pypi.org/project/qwikidata/) python package. 
```
kgtk filter -i Final_triples_QP.tsv -p ";P21;Qnode_demo" > humans.tsv
kgtk join --right-file citizens.tsv --left-file ../humans.tsv > humans.tsv
kgtk join --right-file humans.tsv --left-file Final_triples_QP.tsv > All_triples.tsv
kgtk filter -i All_triples.tsv --regex --match-type fullmatch -p ';;Q[0-9].*' > rhs.tsv
# The All_triples.tsv file contain all the triples in format <Qnode,Pnode,Qnode>
python find_labels.py 
# This command helps to label the entities and relations present in the demography dataset.
python write_labels.py
# This command prepares the final demographic dataset. 
```

### Generation of the knowledge graph embedding
------------------
To generate the embeddings we used [Ampligraph](https://github.com/Accenture/AmpliGraph) library. The files for this are in GENERATE_EMBEDS folder. Following are the commands to train the model and evaluate it's performance.

```
cd Generate_embeddings
from train_model import train_model,eval_model
train_model(model_name,path_to_dataset,num_epochs,saved_model_path)
eval_model(saved_model_path, path_to_dataset,n)
```

### Bias measurement in KG embedding
------------------
The entire code for the bias measurement is present in the folder BIAS_METRIC. Following are the commands to run the bias measurement for each of the demography.
```
import json
import operator
from preprocess import preprocess
from grad_descent_for_bias import train
from find_bias import find_bias
from get_biases_for_all_professions import get_biases
# Preprocessing to get the humans,occupations and gender entities embeddings
all_human_dataset, human2id, human_embeddings, gmf2id, gmf_embeddings = preprocess(demographic_dataset_path, model_path , entire_dataset_path)
f=open(human2id_path,"w")
json.dump(human2id,f)
# Updating the embeddings using gradient descent
model = train(seed, human2id, gmf2id, human_embeddings, gmf_embeddings, train_df, batch_size, use_gpu, learning_rate, checkpoint_path, dimension, epochs, best_loss_input, ifSave, relative_gender)
# Finding the bias scores and sorting in decreasing order to rank the professions
find_bias(human2id_path, ckpt_path, professions_path, embeddings_path, dimension, path_to_save_bias_scores)

```
