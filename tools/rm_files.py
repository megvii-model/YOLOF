#!/usr/bin/python3
# -*- coding:utf-8 -*-

import argparse
import os
import re
from colorama import Fore, Style


def remove_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-iter", "-s", type=int, default=0, help="start iter to remove")
    parser.add_argument("--end-iter", "-e", type=int, default=0, help="end iter to remove")
    parser.add_argument("--prefix", "-p", type=str, default="model_",
                        help="prefix of model to remove")
    parser.add_argument("--dir", "-d", type=str, default="/data/Outputs",
                        help="dir to remove pth model")
    parser.add_argument("--real", "-r", action="store_true",
                        help="really delete or just show what you will delete")
    return parser


def remove_files(args):
    start = args.start_iter
    end = args.end_iter
    prefix = args.prefix
    for folder, _, files in os.walk(args.dir):
        # l = [x for x in f if x.endswith(".pth")]
        models = [f for f in files if re.search(prefix + r"[0123456789]*\.pth", f)]
        delete = [os.path.join(folder, model) for model in models
                  if start <= int(model[len(prefix):-len(".pth")]) <= end]
        if delete:
            for f in delete:
                if args.real:
                    print(f"remove {f}")
                    os.remove(f)
                else:
                    print(f"you may remove {f}")
    if not args.real:
        print(Fore.RED + "use --real parameter to really delete models" + Style.RESET_ALL)


def main():
    args = remove_parser().parse_args()
    remove_files(args)


if __name__ == "__main__":
    main()
