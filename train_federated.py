import random
from train_single import MatrixLoader
import numpy as np
import torch
from tqdm import tqdm
from ncf_model import NeuralCollaborativeFiltering
from dataloader import MovielensDatasetLoader
from ncf_server_model import ServerNeuralCollaborativeFiltering
from svdpp_model import SVDPP
from svdpp_server_model import ServerSVDPP
from train_single import NCFTrainer
from utils import Utils, seed_everything
from metrics import compute_metrics, compute_metrics_5
import argparse
import logging
class FederatedNCF:
    def __init__(self,
                 data_loader: MovielensDatasetLoader,
                 num_clients=50,
                 user_per_client_range=(1, 5),
                 model="ncf",
                 mode="baseline_noisy",
                 aggregation_epochs=50,
                 local_epochs=10,
                 batch_size=128,
                 latent_dim=32,
                 lr=0.001,
                 client_data_thresh=1,
                 seed=0):
        self.seed = seed
        seed_everything(seed)
        self.data_loader = data_loader
        self.test_set = data_loader.latest_ratings
        self.ui_matrix = self.data_loader.ratings
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_clients = num_clients
        self.latent_dim = latent_dim
        self.user_per_client_range = user_per_client_range
        self.mode = mode
        self.model = model
        self.aggregation_epochs = aggregation_epochs
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.client_data_thresh = client_data_thresh
        self.thresh = 0.1
        self.clients = self.generate_clients()
        self.model_optimizers = [torch.optim.Adam(client.model.parameters(), lr=lr) for client in self.clients]
        self.utils = Utils(self.num_clients)
        self.hrs = []
        self.ndcg = []
        self.loss = []
        self.val_best_hit = 0
        self.val_best_ndcg = 0

    def generate_clients(self):
        clients = []
        random_integers = []
        while len(random_integers) < self.num_clients:
            num = random.randint(0, len(self.ui_matrix)-1)
            client_matrix = self.ui_matrix[num]
            client_clean_num = np.argwhere(client_matrix >= self.client_data_thresh)
            if len(client_clean_num) > 18:
                random_integers.append(num)
        for i in random_integers:
            users = random.randint(self.user_per_client_range[0], self.user_per_client_range[1])
            clients.append(NCFTrainer(user_ids=list(range(i, i + users)),
                                      data_loader=self.data_loader,
                                      epochs=self.local_epochs,
                                      batch_size=self.batch_size,
                                      mode=self.mode,
                                      model=self.model,
                                      thresh=self.thresh,
                                      latent_dim=self.latent_dim))
        return clients

    def single_round(self, epoch=0):
        single_round_results = {key: [] for key in ["num_users", "loss"]}
        bar = tqdm(enumerate(self.clients), total=self.num_clients)
        for client_id, client in bar:
            results = client.train(self.model_optimizers[client_id], epoch)
            for k, i in results.items():
                single_round_results[k].append(i)
            printing_single_round = {"epoch": epoch}
            printing_single_round.update({k: round(sum(i) / len(i), 2) for k, i in single_round_results.items()})
            model = torch.jit.script(client.model.to(torch.device("cpu")))
            torch.jit.save(model, "./models/local/dp" + str(client_id) + ".pt")
            bar.set_description(str(printing_single_round))
        self.loss.append(single_round_results["loss"])
        bar.close()

    def extract_item_models(self):
        for client_id in range(self.num_clients):
            model = torch.jit.load("./models/local/dp" + str(client_id) + ".pt")
            if self.model == 'ncf':
                item_model = ServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
            elif self.model == 'svdpp':
                item_model = ServerSVDPP(item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
            else:
                raise ValueError('can not found a server model matched')
            item_model.set_weights(model)
            item_model = torch.jit.script(item_model.to(torch.device("cpu")))
            torch.jit.save(item_model, "./models/local_items/dp" + str(client_id) + ".pt")

    def train(self):
        if self.model == 'ncf':
            server_model = ServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1],predictive_factor=self.latent_dim)
        elif self.model == 'svdpp':
            server_model = ServerSVDPP(item_num=self.ui_matrix.shape[1],predictive_factor=self.latent_dim)
        else:
            raise ValueError('can not found a server model matched')
        server_model = torch.jit.script(server_model.to(torch.device("cpu")))
        torch.jit.save(server_model, "./models/central/server" + str(0) + ".pt")
        for epoch in range(self.aggregation_epochs):
            server_model = torch.jit.load("./models/central/server" + str(epoch) + ".pt",
                                          map_location=self.device)
            _ = [client.model.to(self.device) for client in self.clients]
            _ = [client.model.load_server_weights(server_model) for client in self.clients]
            self.single_round(epoch=epoch)
            self.extract_item_models()
            self.utils.federate()
            self.test(epoch)

    def test(self, epoch):
        i = epoch + 1
        server_model = torch.jit.load("./models/central/server" + str(i) + ".pt", map_location=self.device)
        if self.model == 'ncf':
            moni_client_model = NeuralCollaborativeFiltering(self.ui_matrix.shape[0],self.ui_matrix.shape[1],self.latent_dim).to(self.device)
        elif self.model == 'svdpp':
            moni_client_model = SVDPP(self.ui_matrix.shape[0],self.ui_matrix.shape[1],self.latent_dim).to(self.device)
        else:
            raise ValueError('please choose a base model')
        moni_client_model.load_server_weights(server_model)

        hit_10_total = 0
        ndcg_10_total = 0
        hit_5_total = 0
        ndcg_5_total = 0
        for user in range(self.ui_matrix.shape[0]):
            if user == 3597 or user == 1846:
                continue
            user_ids=[user]
            current_ui_matrix = self.data_loader.get_ui_matrix(user_ids)
            loader = MatrixLoader(current_ui_matrix, dataloader=self.data_loader, user_ids=user_ids)
            test_batch = loader.get_test_batch().to(self.device)
            hr, ndcg = compute_metrics(model=moni_client_model, test_batch=test_batch, device=self.device)
            hr_5, ndcg_5 = compute_metrics_5(model=moni_client_model, test_batch=test_batch, device=self.device)
            hit_10_total += hr
            ndcg_10_total += ndcg
            hit_5_total += hr_5
            ndcg_5_total += ndcg_5

        i = i - 1
        print(hit_10_total,ndcg_10_total,hit_5_total,ndcg_5_total,i)
        logging.info(f"第{epoch}个epoch执行测试评估: ndcg@10: {ndcg_10_total:.2f}, hits@10: {hit_10_total:.2f}，ndcg@5: {ndcg_5_total:.2f}, hits@5: {hit_5_total:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='path of ratings.dat file')
    parser.add_argument('--num_clients', type=int, default=600, help='the numbers of clients')
    parser.add_argument('--model', type=str, default='ncf', help='the mode of model, you can choose from both ncf and gmf')
    parser.add_argument('--mode', type=str, default='baseline_clean', help='choose from baseline_clean, baseline_noisy, self_training')
    parser.add_argument('--global_epochs', type=int, default=100, help='global epochs for aggregation')
    parser.add_argument('--local_epochs', type=int, default=5, help='global epochs for training of local epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='learing rate')
    parser.add_argument('--filename', type=str, default='./ml-1m/ratings.dat')
    parser.add_argument('--client_data_thresh', type=int, default=1,
                        help='how many data in each client, 1 for noisy,  3 for clean and self-training')
    args = parser.parse_args()

    if args.dataset == 'ml-100k':
        args.filename = './ml-100k/ratings.dat'
    elif args.dataset == 'ml-1m':
        args.filename = './ml-1m/ratings.dat'
    elif args.dataset == 'hetrec-ml':
        args.filename = './hetrec-ml/ratings.dat'
    else:
        raise ValueError('dataset is not found')

    logging.basicConfig(filename='metrics_info.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Program started with arguments: {}'.format(args))

    dataloader = MovielensDatasetLoader(filename=args.filename)
    seeds = {117623077}
    for s in seeds:
        fncf = FederatedNCF(
            data_loader=dataloader,
            num_clients=args.num_clients,
            user_per_client_range=[1,1],
            model=args.model,
            mode=args.mode,
            aggregation_epochs=args.global_epochs,
            local_epochs=args.local_epochs,
            batch_size=64,
            latent_dim=12,
            lr=args.lr,
            client_data_thresh=args.client_data_thresh,
            seed=s
        )
        fncf.train()
