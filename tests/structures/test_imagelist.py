# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
from typing import List, Tuple
import torch

from cvpods.structures import ImageList
from cvpods.utils.env import TORCH_VERSION


class TestImageList(unittest.TestCase):
    def test_imagelist_padding_shape(self):
        ret = ImageList.from_tensors([torch.ones((3, 15, 20), dtype=torch.float32)], 4).tensor
        self.assertEqual(list(ret.shape), [1, 3, 16, 20], str(ret.shape))

        ret = ImageList.from_tensors(
            [
                torch.ones((3, 25, 20), dtype=torch.float32),
                torch.ones((3, 10, 10), dtype=torch.float32),
            ], 4
        ).tensor
        self.assertEqual(list(ret.shape), [2, 3, 28, 20], str(ret.shape))

    # TODO: Make cvpods.structures.ImageList scriptable
    @unittest.skip("Don't support scriptable cvpods.structures.ImageList")
    @unittest.skipIf(TORCH_VERSION < (1, 6), "Insufficient pytorch version")
    def test_imagelist_scriptability(self):
        image_nums = 2
        image_tensor = torch.randn((image_nums, 10, 20), dtype=torch.float32)
        image_shape = [(10, 20)] * image_nums

        def f(image_tensor, image_shape: List[Tuple[int, int]]):
            return ImageList(image_tensor, image_shape)

        ret = f(image_tensor, image_shape)
        ret_script = torch.jit.script(f)(image_tensor, image_shape)

        self.assertEqual(len(ret), len(ret_script))
        for i in range(image_nums):
            self.assertTrue(torch.equal(ret[i], ret_script[i]))


if __name__ == "__main__":
    unittest.main()
