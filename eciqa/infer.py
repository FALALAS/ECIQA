"""Score an output image conditioned on its original input image."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True, help='Original input/context image')
    parser.add_argument('--candidate', type=Path, required=True, help='Output image to evaluate')
    parser.add_argument('--device', default='cpu', help='cpu or cuda:0')
    parser.add_argument('--precision', choices=('fp32', 'fp16', 'bf16'), default='fp32')
    parser.add_argument('--view-chunk', type=int, default=None)
    parser.add_argument('--cross-query-chunk', type=int, default=None)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    for flag in ('view_chunk', 'cross_query_chunk'):
        if getattr(args, flag) is not None and getattr(args, flag) <= 0:
            parser.error(flag.replace('_', '-') + ' must be positive')
    from . import load_model, predict_pair
    model = load_model(args.weights, device=args.device)
    for flag in ('view_chunk', 'cross_query_chunk'):
        if getattr(args, flag) is not None:
            model.config[flag] = getattr(args, flag)
    if args.cross_query_chunk is not None:
        model.cross_query_chunk = args.cross_query_chunk
    digest = hashlib.sha256()
    with args.weights.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    result = {'model_id': model.config['experiment_id'], 'weights_sha256': digest.hexdigest(),
              'source': str(args.source), 'candidate': str(args.candidate),
              **predict_pair(model, args.source, args.candidate, precision=args.precision)}
    rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered, end='')


if __name__ == '__main__':
    main()
