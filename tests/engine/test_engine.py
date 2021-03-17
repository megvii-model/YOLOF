# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest
import torch
from torch import nn


from cvpods.engine import SimpleRunner
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, length=100):
        self.data_list = torch.rand(length, 3, 3)

    def __getitem__(self, index):
        return self.data_list[index]


class SimpleModel(nn.Sequential):
    def forward(self, x):
        return {"loss": x.sum() + sum([x.mean() for x in self.parameters()])}


class TestTrainer(unittest.TestCase):
    def test_simple_trainer(self, device="cpu"):
        device = torch.device(device)
        model = SimpleModel(nn.Linear(10, 10)).to(device)

        class DataLoader:
            def __len__(self):
                return 10000

            def __iter__(self):
                while True:
                    yield torch.rand(3, 3).to(device)

        trainer = SimpleRunner(model, DataLoader(), torch.optim.SGD(model.parameters(), 0.1))
        trainer.max_epoch = None
        trainer.train(0, 0, 10)
        return trainer

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_simple_trainer_cuda(self):
        self.test_simple_trainer(device="cuda")
