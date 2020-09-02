#!/usr/bin/env python3
# Copyright (c) Hiroyuki Deguchi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
import fileinput


def incremental_bpe_dependency(orig_gov, bpe_symbol_mask, word_head_positions):
    wi = 0
    bpe_govs = list(range(2, len(bpe_symbol_mask) + 2))
    for i, is_bpe_symbol in enumerate(bpe_symbol_mask):
        if not is_bpe_symbol:
            gov = word_head_positions[orig_gov[wi]]
            bpe_govs[i] = i + 1 if gov == 0 else gov
            wi += 1
    return bpe_govs


def main(args):

    is_bpe_symbol = lambda x: str.endswith(x, args.bpe_symbol)

    with \
      fileinput.input(files=[args.orig_govs]) as orig_govs_lines, \
      open(args.bpe_tokens) as bpe_tokens_lines:
        for orig_govs, bpe_tokens in zip(orig_govs_lines, bpe_tokens_lines):

            orig_govs = [int(g) for g in orig_govs.strip().split()]
            bpe_tokens = bpe_tokens.strip().split()

            # Example: ``Thi@@ s is an ex@@ am@@ ple .''
            # => [ROOT, Thi@@, is, an, ex@@, .]
            word_head_positions = [0, 1] + [i + 1 for i, w in enumerate(bpe_tokens, start=1) if not is_bpe_symbol(w)][:-1]
            bpe_symbol_masks = [is_bpe_symbol(w) for w in bpe_tokens]
            bpe_govs = incremental_bpe_dependency(orig_govs, bpe_symbol_masks, word_head_positions)

            out_line = ' '.join(str(g) for g in bpe_govs)
            print(out_line)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--bpe-tokens', '-t', type=str, required=True)
    parser.add_argument('--orig-govs', '-g', type=str, default='-')
    parser.add_argument('--bpe-symbol', type=str, default='@@')
    args = parser.parse_args()
    main(args)
