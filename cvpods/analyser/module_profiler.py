#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved
# pylint: disable=W0613

import functools
from collections import OrderedDict, defaultdict, namedtuple
from tabulate import tabulate

import numpy as np

import torch.autograd.profiler as tprofiler

Trace = namedtuple("Trace", ["path", "module"])
Measure = namedtuple("Measure", ["self_cpu_total", "cpu_total", "cuda_total", "hits"])
ModuleInfo = namedtuple(
    "ModuleInfo", ["type", "self_cpu_total", "cpu_total", "cuda_total", "hits"]
)


def walk_modules(module, name=None, path=()):
    """Generator. Walks through a PyTorch Module and outputs Trace tuples"""
    if not name:
        name = module.__class__.__name__
    named_children = list(module.named_children())
    path = path + (name,)
    if len(named_children) == 0:
        yield Trace(".".join(path), module)
    # recursively walk into all submodules
    for name, child_module in named_children:
        yield from walk_modules(child_module, name=name, path=path)


class Profile(object):
    """
    Layer by layer profiling of Pytorch models, using the Pytorch autograd profiler.
    """

    def __init__(self, module, enabled=True, use_cuda=False, paths=None, with_mapping=True):
        """
        Args:
            model:
            enabled:
            use_cuda:
            paths:
            with_mapping:
        """
        self._module = module
        self.enabled = enabled
        self.use_cuda = use_cuda
        self.paths = paths
        self.with_mapping = with_mapping

        self.entered = False
        self.exited = False
        self.traces = ()
        self.trace_profile_events = defaultdict(list)

    def __enter__(self):
        if not self.enabled:
            return self
        if self.entered:
            raise RuntimeError("torchprof profiler is not reentrant")
        self.entered = True
        self._forwards = {}  # store the original forward functions
        self.traces = tuple(map(self._hook_trace, walk_modules(self._module)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        tuple(map(self._remove_hook_trace, self.traces))
        del self._forwards  # remove unnecessary forwards
        self.exited = True

    def __str__(self):
        if self.exited:
            return repr(traces_to_display(
                self.traces, self.trace_profile_events,
                self.use_cuda, paths=self.paths
            ))
        return "<unfinished torchprof.profile>"

    def __call__(self, *args, **kwargs):
        return self._module(*args, **kwargs)

    def _hook_trace(self, trace):
        path, module = trace
        if (self.paths is not None and path in self.paths) or (self.paths is None):
            _forward = module.forward
            self._forwards[path] = _forward

            @functools.wraps(_forward)
            def wrap_forward(*args, **kwargs):
                with tprofiler.profile(use_cuda=self.use_cuda) as prof:
                    res = _forward(*args, **kwargs)
                event_list = prof.function_events
                event_list.populate_cpu_children()
                # each profile call should be contained in its own list
                self.trace_profile_events[path].append(event_list)
                return res

            module.forward = wrap_forward
        return trace

    def _remove_hook_trace(self, trace):
        path, module = trace
        if (self.paths is not None and path in self.paths) or (self.paths is None):
            module.forward = self._forwards[path]


class InfoTable():

    def __init__(self, headers, data, average=False):
        assert len(headers) == len(data), "headers and data are not matched"
        self.headers = headers
        self.info = {key: value for key, value in zip(headers, data)}
        if average:
            self.average()

    def insert(self, header, data, position=-1):

        def swap(a, b):
            a, b = b, a

        self.info[header] = data
        if header in self.headers:
            index = self.headers.index(header)
            swap(self.headers[index], self.headers[position])
        else:
            self.headers.insert(position, header)

    def sorted_by(self, keyname=None, descending=True):
        if keyname is None:
            return self
        assert keyname in self.info
        sort_index = np.argsort(self.info[keyname], axis=0).reshape(-1)
        if descending:
            sort_index = sort_index[::-1]
        for header in self.headers:
            self.info[header] = self.info[header][sort_index]

        return self

    def filter(self, filter_list=None):
        self.headers = [header for header in self.headers if header not in filter_list]

    def average(self):
        hits = self.info["hits"]
        for i in range(len(self.headers)):
            header = self.headers[i]
            if header.endswith("time"):
                self.info[header + "_avg"] = self.info[header] / hits
                self.headers[i] += "_avg"
                del self.info[header]

    def __repr__(self):
        formatter = np.vectorize(tprofiler.format_time)
        data = np.concatenate(
            [formatter(self.info[k]) if "time" in k else self.info[k] for k in self.headers],
            axis=1,
        )
        table = tabulate(data, headers=self.headers, tablefmt="fancy_grid")
        return table


def genertate_info_tree(traces, trace_events, level="module"):
    """
    """
    assert level in ["module", "operator", "mixed"]
    tree = OrderedDict()

    for trace in traces:
        path, module = trace
        # unwrap all of the events, in case model is called multiple times
        events = [te for tevents in trace_events[path] for te in tevents]
        if level == "module":
            tree[path] = ModuleInfo(
                repr(module),
                sum([e.self_cpu_time_total for e in events]),
                sum([e.cpu_time_total for e in events]),
                sum([e.cuda_time_total for e in events]),
                len(trace_events[path])
            )
        elif level == "operator" or level == "mixed":
            for op in set(event.name for event in events):
                op_events = [e for e in events if e.name == op]
                measure = Measure(
                    sum([e.self_cpu_time_total for e in op_events]),
                    sum([e.cpu_time_total for e in op_events]),
                    sum([e.cuda_time_total for e in op_events]),
                    len(op_events),
                )
                if level == "mixed":
                    tree[path + "." + op] = measure
                else:
                    # operator mode
                    if op not in tree:
                        tree[op] = measure
                    else:
                        tree[op] = Measure(*(a + b for a, b in zip(tree[op], measure)))

    return tree


def traces_to_display(
    traces, trace_events, with_cuda, paths=None, sorted_by="self_cpu_time", average=True
):
    """Construct human readable output of the profiler traces and events.
    """
    headers = ["name", "self_cpu_time", "cpu_time", "cuda_time", "hits"]
    data_type = ["float32", "float32", "float32", "int32"]
    # tree = genertate_infotable(traces, trace_events, "operator")
    tree = genertate_info_tree(traces, trace_events, "module")
    # tree = genertate_infotable(traces, trace_events, "mixed")
    format_lines = [
        (
            name,
            # info.type,
            info.self_cpu_total,
            info.cpu_total,
            info.cuda_total,
            info.hits,
        ) for name, info in tree.items()
    ]
    data = np.array(format_lines)
    data = np.hsplit(data, len(headers))
    data[1:] = [x.astype(dtype) for x, dtype in zip(data[1:], data_type)]
    table = InfoTable(headers, data)
    # table.average()
    table.sorted_by("cpu_time").filter(["cuda_time"])
    return table
