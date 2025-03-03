import json 
from anytree import Node, RenderTree
import tensorflow_hub as hub
import torch

def read_json(file_path):
    with open(file_path, 'r') as file: data = json.load(file)
    return data

def json2tree(input, model): # TREE MUST HAVE ONLY 2 LAYERS 
    root = Node("Root")
    for K1 in input.keys(): 
        child1 = Node(K1, parent=root)
        for K2 in input[K1].keys(): 
            child2 = Node(K2, parent=child1)
            desc = Node(input[K1][K2], parent=child2)
    
    for pre, fill, node in RenderTree(root): node.vect = torch.tensor(model([str])[0].numpy())
    return root


def score(a, b, model): 

    tree_a = json2tree(a,model)


    return a + b 




if __name__ == "__main__": 
    # model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    # embed = hub.load(model_url)

    # # TEST
    # sentence = "Google's Universal Sentence Encoder generates 512-dimensional embeddings."
    # embedding = embed([sentence])[0].numpy()
    # embedding = torch.tensor(embedding)
    # print(embedding.shape)
    # print("\n")

    collection = read_json("positives.json")

    print(collection.keys())
    Foods = collection["Foods"]
    Countries = collection["Countries"]
    Leaders = collection["Leaders"]
    print("\t",Foods.keys())
    print("\t",Countries.keys())
    print("\t",Leaders.keys())

    score(Foods, Countries)


    print("over")