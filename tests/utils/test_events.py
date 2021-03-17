# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import os
import tempfile
import unittest

from cvpods.utils import EventStorage, JSONWriter


class TestEventWriter(unittest.TestCase):
    def testScalar(self):
        with tempfile.TemporaryDirectory(
            prefix="cvpods_tests"
        ) as dir, EventStorage() as storage:
            json_file = os.path.join(dir, "test.json")
            writer = JSONWriter(json_file)
            for k in range(60):
                storage.put_scalar("key", k, smoothing_hint=False)
                if (k + 1) % 20 == 0:
                    writer.write()
                storage.step()
            writer.close()
            with open(json_file) as f:
                data = [json.loads(line) for line in f]
                self.assertTrue([int(k["key"]) for k in data] == [19, 39, 59])

    @unittest.skip("EventStorage doesn't support mismatched period writing")
    def testScalarMismatchedPeriod(self):
        with tempfile.TemporaryDirectory(
            prefix="cvpods_tests"
        ) as dir, EventStorage() as storage:
            json_file = os.path.join(dir, "test.json")

            writer = JSONWriter(json_file)
            for k in range(60):
                if k % 17 == 0:  # write in a differnt period
                    storage.put_scalar("key2", k, smoothing_hint=False)
                storage.put_scalar("key", k, smoothing_hint=False)
                if (k + 1) % 20 == 0:
                    writer.write()
                storage.step()
            writer.close()
            with open(json_file) as f:
                data = [json.loads(line) for line in f]
                print([int(k.get("key2", 0)) for k in data])
                print([int(k.get("key", 0)) for k in data])
                print([int(k.get("iteration", 0)) for k in data])
                self.assertTrue([int(k.get("key2", 0)) for k in data] == [17, 0, 34, 0, 51, 0])
                self.assertTrue([int(k.get("key", 0)) for k in data] == [0, 19, 0, 39, 0, 59])
                self.assertTrue([int(k["iteration"]) for k in data] == [17, 19, 34, 39, 51, 59])
