#!/usr/bin/env python3
# Copyright (c) Hiroyuki Deguchi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import fileinput
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument("--input", type=str, default="-", metavar="FILE")
    parser.add_argument("--split-fwspace", action="store_true",
                        help="Split by full-width space for Japanese corpora.")
    # fmt: on
    args = parser.parse_args()
    return args


def main(args):
    split_fwspace = args.split_fwspace

    with fileinput.input(files=[args.input]) as f:
        heads = []
        shift = 0
        for line in f:
            line = line.strip().split(" ")

            # End of sentence
            if len(line) <= 1:
                print(" ".join(str(h) for h in heads))
                heads = []
                shift = 0
                continue

            # CoNLL-U format:
            # ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
            assert len(line) >= 7
            head = int(line[6])

            if not split_fwspace:
                heads.append(head)
                continue

            num_spaces = line[1].count("\u3000")  # full-width space
            if num_spaces > 0:
                id = int(line[0])
                for i in range(num_spaces):
                    heads.append(id + shift + (i + 1))  # for Japanese
                shift += num_spaces
            heads.append(head + shift if head > 0 else 0)


if __name__ == "__main__":
    args = parse_args()
    main(args)
