#!/usr/bin/env python3
# Copyright (c) Hiroyuki Deguchi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import fileinput
from argparse import ArgumentParser


def get_word_start_positions(bpe_tokens):
    # Example:
    #    `Thi@@ s is an ex@@ am@@ ple .'
    # => [0 (Thi@@), 2 (is), 3 (an), 4 (ex@@), 7 (.)]
    return [0] + [i + 1 for i, w in enumerate(bpe_tokens[:-1]) if not w.endswith("@@")]


def incremental_bpe_dependency(word_heads, bpe_tokens):
    word_start_positions = get_word_start_positions(bpe_tokens)
    wi = 0
    bpe_heads = []
    for i, token in enumerate(bpe_tokens):
        if token.endswith("@@"):
            bpe_heads.append(i + 1)
        else:
            word_head = word_heads[wi]
            bpe_heads.append(
                word_start_positions[word_head - 1] if word_head > 0 else i
            )
            wi += 1
    return bpe_heads


def main(args):
    with fileinput.input(files=[args.heads_file]) as head_lines:
        with open(args.bpe_file) as bpe_lines:
            for head_line, bpe_line in zip(head_lines, bpe_lines):
                word_heads = [int(h) for h in head_line.strip().split()]
                bpe_tokens = bpe_line.strip().split()
                bpe_heads = incremental_bpe_dependency(word_heads, bpe_tokens)
                print(" ".join(str(h) for h in bpe_heads))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bpe-file", type=str, required=True)
    parser.add_argument("--heads-file", type=str, default="-")
    parser.add_argument("--bpe-symbol", type=str, default="subword_nmt")
    args = parser.parse_args()
    main(args)
