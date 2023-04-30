from get_biases_for_all_professions_own import get_biases
import operator
import json

def find_bias(human2id_path, ckpt_path, professions_path, embeddings_path,dimension,path_to_save_bias_scores)
    for i in [0,500,1000,1500]:
        biases = get_biases(i, i+500,human2id_path, ckpt_path, professions_path, embeddings_path ,dimension)
        biases = {k: v.detach().numpy().reshape(1,1).item() for k, v in biases.items()}
        sorted_tuples = sorted(biases.items(), key=operator.itemgetter(1), reverse = True)
        f=open(path_to_save_bias_scores+"/biases"+str(i)+".json","w")
        json.dump(sorted_tuples,f)
        del biases
        del sorted_tuples
    biases=dict()
    for i in [0,500,1000,1500]:
        f=open(path_to_save_bias_scores+"/biases"+str(i)+".json","r")
        biases2=json.load(f)
        biases2=dict(biases2)
        biases.update(biases2)
        print(len(biases))
    sorted_tuples = sorted(biases.items(), key=operator.itemgetter(1), reverse = True)
    s=[i for i,j in sorted_tuples]
    print(s)
