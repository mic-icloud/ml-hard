import math

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler

from typing import List

import utils


class ListNet(torch.nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # укажите архитектуру простой модели здесь
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_input_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_1)
        return logits


class Solution:
    def __init__(self, n_epochs: int = 5, listnet_hidden_dim: int = 30,
                 lr: float = 0.001, ndcg_top_k: int = 10):
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs

        self.model = self._create_model(
            self.num_input_features, listnet_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
            X_test, y_test, self.query_ids_test) = self._get_data()
        # используйте методы _scale_features_in_query_groups для X_train и X_test.
        # поместить X_train и X_test, y_train и y_test в torch.FloatTensor
        self.X_train = torch.FloatTensor(self._scale_features_in_query_groups(X_train, self.query_ids_train))
        self.y_train = torch.FloatTensor(y_train)
        self.X_test = torch.FloatTensor(self._scale_features_in_query_groups(X_test, self.query_ids_test))
        self.y_test = torch.FloatTensor(y_test)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        unique_query_ids = np.unique(inp_query_ids)
        scaled_features = np.zeros_like(inp_feat_array)
        
        for qid in unique_query_ids:
            group_indices = np.where(inp_query_ids == qid)[0]
            scaler = StandardScaler()
            scaled_features[group_indices] = scaler.fit_transform(inp_feat_array[group_indices])
        
        return scaled_features

    def _create_model(self, listnet_num_input_features: int,
                      listnet_hidden_dim: int) -> torch.nn.Module:
        torch.manual_seed(0)
        # создайте модель ListNet
        net = ListNet(listnet_num_input_features, listnet_hidden_dim)
        return net

    def fit(self) -> List[float]:
        # обучение модели в течение n_epochs
        ndcgs = []
        for _ in range(self.n_epochs):
            self._train_one_epoch()
            ndcgs.append(self._eval_test_set())
        return  ndcgs

    def _calc_loss(self, batch_ys: torch.FloatTensor,
                   batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        # 
        return utils.listnet_kl_loss(batch_ys, batch_pred)

    def _train_one_epoch(self) -> None:
        self.model.train()
        unique_query_ids = np.unique(self.query_ids_train)
        for qid in unique_query_ids:
            group_indices = np.where(self.query_ids_train == qid)[0]
            X_batch = self.X_train[group_indices]
            y_batch = self.y_train[group_indices]

            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self._calc_loss(y_batch, y_pred)
            loss.backward()
            self.optimizer.step()

    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            ndcgs = []
            # допишите ваш код здесь
            ids = np.unique(self.query_ids_test)
            for qid in ids:
                group_indices = np.where(self.query_ids_test == qid)[0]
                X_batch = self.X_test[group_indices]
                y_batch = self.y_test[group_indices]
                y_pred = self.model(X_batch)
                try:
                    ndcgs.append(self._ndcg_k(y_batch, y_pred, self.ndcg_top_k))
                except Exception:
                    ndcgs.append(0)
            return np.mean(ndcgs)

    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int) -> float:
        # вызовите функцию ndcg из utils.py. взять первые ndcg_top_k элементов
        return utils.ndcg(ys_true[:ndcg_top_k], ys_pred[:ndcg_top_k], gain_scheme='exp2')


net = Solution()
ndcgs = net.fit() 
print(ndcgs)