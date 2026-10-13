"""Training loop of the original nanoTabPFN (train.py), with the schedule-free eval switch fixed and checkpointing.

Schedule-free keeps different weights for training (y) and evaluation (x). Evaluation and checkpoints must use x
("The optimizer should also be placed in eval mode when saving checkpoints"). Switching to eval mode and back
(optimizer.eval() then optimizer.train()) is not exact in floating point, so it would nudge the training
trajectory. eval_weights therefore snapshots the training weights, switches to eval mode, and afterwards switches
back and restores the snapshot exactly: a run with checkpoints follows the same trajectory as one without.
"""

import time
from contextlib import contextmanager

import schedulefree
import torch
from torch import nn

from tfmprior.utils.device import release_cached_memory


@contextmanager
def eval_weights(model: nn.Module, optimizer):
    """Inside the block the model holds the schedule-free evaluation weights; afterwards training resumes unchanged."""
    snapshot = [p.detach().clone() for p in model.parameters()]
    model.eval()
    optimizer.eval()
    try:
        yield model
    finally:
        optimizer.train()
        model.train()
        with torch.no_grad():
            for p, saved in zip(model.parameters(), snapshot):
                p.copy_(saved)


def train(model: nn.Module, loader, lr: float, device: str, clip_grad_norm: float = 1.0,
          on_step=None, save_at_steps=(), on_save=None):
    """Trains model on loader. Returns the model holding the evaluation weights.

    on_step(step, loss, batch, seconds) is called after every optimizer step (step counts from 1).
    on_save(step, model) is called at each step in save_at_steps, with the evaluation weights in place.
    """
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    save_at_steps = set(save_at_steps)

    model.train()
    optimizer.train()
    for step, full_data in enumerate(loader, start=1):
        step_start_time = time.time()
        train_test_split_index = full_data["train_test_split_index"]
        data = (full_data["x"].to(device),
                full_data["y"][:, :train_test_split_index].to(device))
        targets = full_data["y"].to(device)

        output = model(data, train_test_split_index=train_test_split_index)
        targets = targets[:, train_test_split_index:]

        targets = targets.reshape((-1,)).to(torch.long)
        output = output.view(-1, output.shape[-1])

        loss = criterion(output, targets).mean()
        loss.backward()
        total_loss = loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        release_cached_memory(device)
        if on_step is not None:
            on_step(step, total_loss, full_data, time.time() - step_start_time)
        if step in save_at_steps and on_save is not None:
            with eval_weights(model, optimizer):
                on_save(step, model)

    model.eval()
    optimizer.eval()
    return model
