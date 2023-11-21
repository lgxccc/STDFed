import numpy as np
import torch
from dataloader import MovielensDatasetLoader
from ncf_model import NeuralCollaborativeFiltering
from svdpp_model import SVDPP
from utils import f_x, setdiff2d_set

class MatrixLoader:
    def __init__(self,
                 ui_matrix,
                 dataloader: MovielensDatasetLoader = None,
                 default=None,
                 user_ids=None,
                 thresh=0.1):  # data with a score of threshold and above is valid training data
        self.ui_matrix = ui_matrix
        self.positives = np.argwhere(self.ui_matrix >= thresh)
        self.negatives = np.argwhere(self.ui_matrix == 0)
        for i, usr_id in enumerate(user_ids):
            self.positives[self.positives[:, 0] == i, 0] = usr_id
            self.negatives[self.negatives[:, 0] == i, 0] = usr_id
        self.relabel = False
        self.user_ids = user_ids
        self.dataloader = dataloader
        if user_ids:
            test_interactions = np.array(
                [[usr_id, dataloader.latest_ratings[usr_id]["item_id"]] for usr_id in user_ids])
            val_interactions = np.array(
                [[usr_id, dataloader.val_set[usr_id]["item_id"]] for usr_id in user_ids])
            latest_interactions = np.concatenate((test_interactions, val_interactions),axis=0)
            mask = np.array([not np.array_equal(row, rows_to_remove_i) for row in self.positives for rows_to_remove_i in
                             latest_interactions]).reshape(self.positives.shape[0], latest_interactions.shape[0]).all(
                axis=1)
            self.positives = self.positives[mask]
        if default is None:
            self.default = np.array([[0, 0]]), np.array([0])
        else:
            self.default = default

    def delete_indexes(self, indexes, arr="pos"):
        if arr == "pos":
            self.positives = np.delete(self.positives, indexes, 0)
        else:
            self.negatives = np.delete(self.negatives, indexes, 0)

    def get_batch(self, batch_size):
        if self.positives.shape[0] < batch_size // 4 or self.negatives.shape[0] < batch_size - batch_size // 4:
            return torch.tensor(self.default[0]), torch.tensor(self.default[1])
        try:
            pos_indexes = np.random.choice(self.positives.shape[0], batch_size // 4, replace=False)
            neg_indexes = np.random.choice(self.negatives.shape[0], batch_size - batch_size // 4, replace=False)
            pos = self.positives[pos_indexes]
            neg = self.negatives[neg_indexes]
            self.delete_indexes(pos_indexes, "pos")
            self.delete_indexes(neg_indexes, "neg")
            batch = np.concatenate((pos, neg), axis=0)
            if batch.shape[0] != batch_size:
                return torch.tensor(self.default[0]), torch.tensor(self.default[1]).float()
            np.random.shuffle(batch)
            y = np.array([self.dataloader.ratings[i][j] for i, j in batch])
            return batch, y
        except Exception as exp:
            print(exp)
            return torch.tensor(self.default[0]), torch.tensor(self.default[1]).float()

    def get_test_batch(self):
        user_id = np.random.choice(self.user_ids)
        pos = np.array([user_id, self.dataloader.latest_ratings[user_id]["item_id"]])
        neg_indexes = np.random.choice(self.negatives.shape[0], 99)
        neg = self.negatives[neg_indexes]
        batch = np.concatenate((pos.reshape(1, -1), neg), axis=0)
        return torch.tensor(batch)

    def get_all_train(self):
        pos = self.positives
        y = np.array([self.dataloader.ratings[i][j] for i, j in pos])
        return pos,y


class NCFTrainer: #初始化客户端时会创建该类的实例
    def __init__(self,
                 data_loader: MovielensDatasetLoader,
                 user_ids,
                 epochs,
                 batch_size,
                 mode,
                 model,
                 thresh,
                 latent_dim=32):
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.loader = None
        self.thresh = thresh
        self.user_ids = user_ids
        self.relabel = False
        self.mode = mode
        self.unlabeled = np.zeros((0, 2))
        self.data_loader = data_loader
        self.ui_matrix = self.data_loader.get_ui_matrix(self.user_ids)
        self.initialize_loader()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if model == 'ncf':
            self.model = NeuralCollaborativeFiltering(self.data_loader.ratings.shape[0], self.data_loader.ratings.shape[1],
                                                self.latent_dim).to(self.device)
            self.sever_model = NeuralCollaborativeFiltering(self.data_loader.ratings.shape[0], self.data_loader.ratings.shape[1],
                                                       self.latent_dim).to(self.device)
        elif model == 'svdpp':
            self.model = SVDPP(self.data_loader.ratings.shape[0], self.data_loader.ratings.shape[1],
                                                      self.latent_dim).to(self.device)
            self.sever_model = SVDPP(self.data_loader.ratings.shape[0],self.data_loader.ratings.shape[1],
                                                            self.latent_dim).to(self.device)
        else:
            raise ValueError('please specify base model')

    def initialize_loader(self):
        self.loader = MatrixLoader(self.ui_matrix, dataloader=self.data_loader, user_ids=self.user_ids, thresh=self.thresh)

    def train_batch(self, x, y, optimizer):
        self.model.train()
        optimizer.zero_grad()
        y_ = self.model(x)
        loss = torch.nn.functional.binary_cross_entropy(y_, y)  # compute the loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item(), y_.detach()

    def train_model(self, optimizer, epoch_global, epochs_local=None):
        epoch = 0
        server_model = torch.jit.load("./models/central/server" + str(epoch_global) + ".pt", map_location=self.device)
        self.sever_model.load_server_weights(server_model)
        progress = {"epoch": [], "loss": []}
        running_loss, prev_running_loss = 0, 0
        if epochs_local is None:
            epochs_local = self.epochs
        steps, prev_steps, prev_epoch, count, step_total = 0, 0, 0, 0, 0
        while epoch < epochs_local:
            x, y = self.loader.get_batch(self.batch_size)
            if x.shape[0] < self.batch_size:
                prev_running_loss = running_loss
                running_loss = 0
                prev_steps = steps
                step_total += steps
                epoch += 1
                self.initialize_loader()
                x, y = self.loader.get_batch(self.batch_size)
            y[y <= 0.1] = 0
            y[y > 0.1] = 1
            x = torch.tensor(x).int()
            y = torch.tensor(y).float()
            x, y = x.to(self.device), y.to(self.device)
            loss, y_ = self.train_batch(x, y, optimizer)
            running_loss += loss
            if epoch != 0 and steps == 0:
                results = {"epoch": prev_epoch, "loss": prev_running_loss / (prev_steps + 1)}
            else:
                results = {"epoch": prev_epoch, "loss": running_loss / (steps + 1)}
            if prev_epoch != epoch:
                progress["epoch"].append(results["epoch"])
                progress["loss"].append(results["loss"])
                prev_epoch += 1

        if self.mode == 'self_training':  # denoising process
            if epoch_global >= 10 and not self.relabel:  # warm up for 10 rounds
                self.initialize_loader()
                all_train, all_train_label = self.loader.get_all_train()
                all_train = torch.tensor(all_train).int().to(self.device)
                with torch.no_grad():
                    client_pred_train = self.model(all_train)
                    server_pred_train = self.sever_model(all_train)
                    client_pred_train = client_pred_train.to('cpu').numpy()
                    server_pred_train = server_pred_train.to('cpu').numpy()
                    clean_confid_idx = np.where((client_pred_train > f_x(epoch_global,0.5)) & (server_pred_train > f_x(epoch_global,0.4)))[0]
                    noisy_confid_idx = np.where((client_pred_train < f_x(epoch_global,0.4)) & (server_pred_train < f_x(epoch_global,0.2)))[0]
                    if len(clean_confid_idx) + len(noisy_confid_idx) >= 16:  # only if confidence samples >= 16, relabeling is conducted
                        self.loader.ui_matrix.fill(0)
                        neg_confid = self.loader.positives[noisy_confid_idx]
                        if len(neg_confid) >0: #relabel noisy samples identified by the dual-network
                            for item in neg_confid:
                                row_idx, col_idx = item
                                self.ui_matrix[0, col_idx] = 0.1
                        pos_confid = self.loader.positives[clean_confid_idx]
                        confid_all = np.concatenate((pos_confid, neg_confid), axis=0)
                        self.unlabeled = setdiff2d_set(self.loader.positives, confid_all)
                        self.unlabeled = self.unlabeled.reshape(-1,2)
                        if len(confid_all) >0:
                            for item in pos_confid:
                                row_idx, col_idx = item
                                self.ui_matrix[0, col_idx] = 5
                        self.relabel = True
                        r_results = {"num_users": self.ui_matrix.shape[0]}
                        r_results.update({i: results[i] for i in ["loss"]})
                        return r_results, progress

            if epoch_global >= 10 and self.relabel:
                unlabel_train = self.unlabeled
                if len(unlabel_train) > 0:
                    unlabel_train = torch.tensor(unlabel_train).int().to(self.device)
                    with torch.no_grad():
                        client_pred_train = self.model(unlabel_train).to('cpu').numpy()
                        server_pred_train = self.sever_model(unlabel_train).to('cpu').numpy()
                        clean_confid_idx = np.where((client_pred_train > f_x(epoch_global,0.5)) & (server_pred_train > f_x(epoch_global,0.4)))[0]#server
                        unlabel_train = unlabel_train.to('cpu').numpy()
                        if len(clean_confid_idx) > 0:
                            clean_confid_data = unlabel_train[clean_confid_idx]
                            for item in clean_confid_data:
                                row_idx, col_idx = item
                                self.ui_matrix[0, col_idx] = 5
                            self.unlabeled = setdiff2d_set(self.unlabeled, clean_confid_data)
                        noisy_confid_idx = np.where((client_pred_train < f_x(epoch_global,0.4)) & (server_pred_train < f_x(epoch_global,0.2)))[0] #client
                        if len(noisy_confid_idx) >0:
                            noisy_confid_data = unlabel_train[noisy_confid_idx]
                            for item in noisy_confid_data:
                                row_idx, col_idx = item
                                self.ui_matrix[0, col_idx] = 0.1
                            self.unlabeled = setdiff2d_set(self.unlabeled, noisy_confid_data)

            if epoch_global >= 30 and self.relabel:  # final revision
                unlabel_train = self.unlabeled
                for item in unlabel_train:
                    row_idx, col_idx = item
                    self.loader.ui_matrix[0, col_idx] = 5
                self.unlabeled = np.zeros((0, 2))
            self.initialize_loader()

        r_results = {"num_users": self.ui_matrix.shape[0]}
        r_results.update({i: results[i] for i in ["loss"]})
        return r_results, progress

    def train(self, ncf_optimizer, epoch, return_progress=False):
        if isinstance(self.model, NeuralCollaborativeFiltering):
            self.model.join_output_weights()
        results, progress= self.train_model(ncf_optimizer, epoch)
        if return_progress:
            return results, progress
        else:
            return results


