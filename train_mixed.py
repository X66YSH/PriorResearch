"""Pretrains one nanoTabPFN on a mixture of the generic and finance priors with finance share r.

Tables come from the pre-generated pools in prior_dumps/ (make_prior_dump.py) unless --on_the_fly is given.

Example: python train_mixed.py --r 0.3 --seed 0 --steps 3000 --batch_size 8
"""

import argparse
import os
import time

import torch

from model import NanoTabPFNClassifier, NanoTabPFNModel
from priors.batching import DumpMixedDataLoader, MixedPriorDataLoader
from train import eval, get_default_device, set_randomness_seed, train

parser = argparse.ArgumentParser()
parser.add_argument("--r", type=float, required=True, help="share of tables drawn from the finance prior")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--generic", default="graph_scm", help="TabICL prior type for the generic side: graph_scm (v2) or mix_scm (v1)")
parser.add_argument("--generic_dump", default="prior_dumps/generic_graph_scm_100k_256x24_seed0.h5")
parser.add_argument("--finance_dump", default="prior_dumps/finance_100k_256x24_seed0.h5")
parser.add_argument("--on_the_fly", action="store_true", help="generate tables during training instead of reading the dumps")
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=4e-3)
parser.add_argument("--embedding_size", type=int, default=96)
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--steps_per_eval", type=int, default=50)
parser.add_argument("--out_dir", default="checkpoints")
args = parser.parse_args()

device = get_default_device()
set_randomness_seed(args.seed)
model = NanoTabPFNModel(
    embedding_size=args.embedding_size,
    num_attention_heads=4,
    mlp_hidden_size=2 * args.embedding_size,
    num_layers=args.num_layers,
    num_outputs=2,
)
if args.on_the_fly:
    prior = MixedPriorDataLoader(r=args.r, num_steps=args.steps, batch_size=args.batch_size, seed=args.seed,
                                 generic_type=args.generic)
else:
    assert args.generic in args.generic_dump, f"--generic {args.generic} does not match {args.generic_dump}"
    prior = DumpMixedDataLoader(r=args.r, num_steps=args.steps, batch_size=args.batch_size, seed=args.seed,
                                generic_path=args.generic_dump, finance_path=args.finance_dump)

start = time.time()
model, history = train(model, prior, lr=args.lr, device=device, steps_per_eval=args.steps_per_eval, eval_func=eval)
print(f"wall time {time.time() - start:.0f}s on {device}")
print(f"finance share: {prior.n_finance}/{prior.n_tables} = {prior.n_finance / prior.n_tables:.3f} (target r = {args.r})")
print("breast cancer:", eval(NanoTabPFNClassifier(model, device)))

os.makedirs(args.out_dir, exist_ok=True)
path = os.path.join(args.out_dir, f"r{args.r:.2f}_seed{args.seed}.pth")
torch.save({"model": model.state_dict(), "args": vars(args), "history": history}, path)
print("saved", path)
