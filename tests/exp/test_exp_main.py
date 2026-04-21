# Code from: https://github.com/Ladbaby/PyOmniTS
import unittest

import torch.nn as nn

from exp.exp_main import Exp_Main
from utils.configs import get_configs


class TestExpMain(unittest.TestCase):

    def test_exp_main(self):
        configs = get_configs(args=[])
        exp = Exp_Main(configs)
        self.assertIsNotNone(exp.configs)
        self.assertIsNotNone(exp.device)

    def test_select_optimizer_uses_adam_for_qshnet(self):
        configs = get_configs(args=[
            "--model_name", "QSHNet",
            "--model_id", "QSHNet",
            "--dataset_name", "USHCN",
            "--dataset_id", "USHCN",
        ])
        exp = Exp_Main(configs)

        optimizer = exp._select_optimizer(nn.Linear(2, 2))

        self.assertEqual(type(optimizer).__name__, "Adam")
