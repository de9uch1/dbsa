#!/usr/bin/env python3
# Copyright (c) Hiroyuki Deguchi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import fileinput
import os
import sys
from argparse import ArgumentParser

import stanza
import torch

MWT_MODELS = {
    "ar",
    "de",
    "en",
    "es",
    "fr",
    "ca",
    "cop",
    "cs",
    "el",
    "fa",
    "fi",
    "gl",
    "he",
    "hy",
    "it",
    "kk",
    "mr",
    "pl",
    "pt",
    "ta",
    "tr",
    "uk",
    "wo",
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, default="-", metavar="FILE")
    parser.add_argument("--lang", "-l", default="en")
    parser.add_argument(
        "--model-dir",
        type=str,
        metavar="DIR",
        default=os.path.join(os.environ["HOME"], "stanza_resources"),
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--depparse", action="store_true")
    parser.add_argument("--tokenize", action="store_true")
    args = parser.parse_args()
    return args


def get_stanza_kwargs(args):
    kwargs = {
        "lang": args.lang,
        "dir": args.model_dir,
        "use_gpu": not args.cpu and torch.cuda.is_available(),
    }

    if args.depparse:
        if args.lang in MWT_MODELS:
            kwargs["processors"] = "tokenize,mwt,pos,lemma,depparse"
        else:
            kwargs["processors"] = "tokenize,pos,lemma,depparse"
        kwargs["tokenize_pretokenized"] = not args.tokenize
    elif args.tokenize:
        kwargs["processors"] = "tokenize"
        kwargs["tokenize_no_ssplit"] = True

    return kwargs


def main(args):
    print(args, file=sys.stderr, flush=True)

    stanza.download(args.lang, model_dir=args.model_dir)
    kwargs = get_stanza_kwargs(args)
    pipeline = stanza.Pipeline(**kwargs)

    def run_stanza(lines):
        if args.depparse:
            doc = pipeline([l.split() for l in lines])
            for sent in doc.sentences:
                print(" ".join(str(word.head) for word in sent.words))
        elif args.tokenize:
            doc = pipeline(lines)
            for sent in doc.sentences:
                print(" ".join(str(word.text) for word in sent.words))

    with fileinput.input(files=[args.input]) as f:
        print("| ", end="", file=sys.stderr, flush=True)
        batch = []
        for i, line in enumerate(f, start=1):
            batch.append(line.strip())
            if i % args.batch_size == 0:
                run_stanza(batch)
                print("{}...".format(i), end="", file=sys.stderr, flush=True)
                batch = []
        if len(batch) > 0:
            print(i, file=sys.stderr)
            run_stanza(batch)

    print("| processed sentences: {}".format(i), file=sys.stderr)


if __name__ == "__main__":
    args = parse_args()
    main(args)
