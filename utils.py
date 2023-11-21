import copy
import os
import random
from math import cos, pi
import numpy as np
import torch

def setdiff2d_set(arr1, arr2):
    set1 = set(map(tuple, arr1))
    set2 = set(map(tuple, arr2))
    return np.array(list(set1 - set2))

def f_x(epoch,summit):
    if epoch>=10 and epoch <=30:
        y = 0.05 * cos(pi*(epoch+10)/40) + summit
    else:
        y = summit-0.05
    return y

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class Utils:
    def __init__(self, num_clients, local_path="./models/local_items/", server_path="./models/central/"):
        self.epoch = 0
        self.num_clients = num_clients
        self.local_path = local_path
        self.server_path = server_path

    @staticmethod
    def load_pytorch_client_model(path):
        # Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
        return torch.jit.load(path)

    def get_user_models(self, loader):#获得的用户模型是不含有用户嵌入的参数的，即local_item这里的参数不含有用户嵌入的参数。但是local里的参数含有用户嵌入
        # get clients models from "./models/local_items/"
        models = []
        for client_id in range(self.num_clients):
            models.append({'model': loader(self.local_path + "dp" + str(client_id) + ".pt")})
        return models

    def get_previous_federated_model(self):
        self.epoch += 1
        return torch.jit.load(self.server_path + "server" + str(self.epoch - 1) + ".pt")

    def save_federated_model(self, model):
        torch.jit.save(model, self.server_path + "server" + str(self.epoch) + ".pt")

    # after each epoch the framework saves the result in "models/server"

    def federate(self):
        client_models = self.get_user_models(self.load_pytorch_client_model)
        server_model = self.get_previous_federated_model()  # get the last model saved by the last epoch
        if len(client_models) == 0:
            self.save_federated_model(server_model)  # if there's no models for clients then save the last readed model
            return
        n = len(client_models)

        # deep copy :a copy of the object is copied into another object.
        # It means that any changes made to a copy of the object do not reflect in the original object.
        # Access the model and optimizer state_dict

        # model 'mlp_item_embeddings.weight', 'gmf_item_embeddings.weight', 'mlp.0.weight' , 'mlp.0.bias',
        # 'mlp.2.weight', 'mlp.2.bias' , 'mlp.4.weight', 'mlp.4.bias' , 'gmf_out.weight', 'gmf_out.bias',
        # 'mlp_out.weight', 'mlp_out.bias', 'output_logits.weight', 'output_logits.bias'

        # agg function (server weights+client weights)/#clients
        server_new_dict = copy.deepcopy(client_models[0]['model'].state_dict())
        for i in range(1, len(client_models)):
            client_dict = client_models[i]['model'].state_dict()
            for k in client_dict.keys():
                server_new_dict[k] += client_dict[k]
        for k in server_new_dict.keys():
            server_new_dict[k] = server_new_dict[k] / n
        server_model.load_state_dict(server_new_dict)
        self.save_federated_model(server_model)
