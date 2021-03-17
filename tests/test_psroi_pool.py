import unittest

import numpy as np

import torch
from torch.autograd import gradcheck

from cvpods.layers.psroi_pool import PSROIPool


class PSROIPoolTest(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_psroipool_forward_cuda(self):
        input = torch.randn(2, 21 * 7 * 7, 5, 7, requires_grad=True).float()
        rois = torch.from_numpy(
            np.array([
                [0.0000, 350.6689, 211.0240, 779.0886, 777.7496],
                [0.0000, 744.0627, 277.4919, 988.4307, 602.7589],
                [1.0000, 350.6689, 211.0240, 779.0886, 777.7496],
                [1.0000, 744.0627, 277.4919, 988.4307, 602.7589],
            ])
        ).float()

        pool = PSROIPool((7, 7), 1 / 160.0, 7, 21)
        input = input.cuda()
        rois = rois.cuda()
        out = pool(input, rois)
        assert out.shape == (4, 21, 7, 7), out.shape

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_psroipool_backward_cuda(self):
        input = torch.randn(2, 21 * 7 * 7, 5, 7, requires_grad=True).double()
        rois = torch.from_numpy(
            np.array([
                [0.0000, 350.6689, 211.0240, 779.0886, 777.7496],
                [0.0000, 744.0627, 277.4919, 988.4307, 602.7589],
                [1.0000, 350.6689, 211.0240, 779.0886, 777.7496],
                [1.0000, 744.0627, 277.4919, 988.4307, 602.7589],
            ])
        ).double()

        pool = PSROIPool((7, 7), 1 / 160.0, 7, 21)
        input = input.cuda()
        rois = rois.cuda()
        func = lambda x: pool(x, rois).mean()  # noqa
        gradcheck(func, input, atol=3e-5)


if __name__ == "__main__":
    unittest.main()
