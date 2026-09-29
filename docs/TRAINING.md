# Training and checkpoint provenance

The training entry point preserves the unfolded WMMSE updates and unsupervised
negative-WSR loss. It provides two explicit modes:

```bash
python -m scripts.train --config train_scratch.json --output-dir results/scratch
python -m scripts.train --config train_warmstart.json --output-dir results/warmstart
```

Scratch mode starts with two layers, Kaiming initialization, and zero output-layer
weights. Warm-start mode initializes the first six of seven layers from the
six-layer checkpoint. It initializes a new optimizer; it is **weight initialization,
not an exact optimizer/scheduler resume**. A mismatched or missing checkpoint
produces an explicit error. New checkpoints go to the selected results directory,
not the supplied inference checkpoint directory.

## Available settings versus historical training

The original supplied training entry used seven layers, six-layer initialization,
one epoch, batch size 50, learning rate 2e-4, 20,000 instances, and a 0.25%
validation split. Those settings are preserved in `train_warmstart.json`.
The scratch example uses the same optimizer settings without a prerequisite
checkpoint; it is not a claim about how the reference two-layer model was trained.

The manuscript describes learning-rate decay from 1e-3 to 1e-6. Fig01's separate
initialization experiment starts at 1e-3; the supplied general training entry
starts at 2e-4. The full stage/epoch history of the existing L=2,...,7 checkpoints
has not been recovered. Their `epoch=0` metadata is insufficient to infer that
history. The supplied weights are the reference evaluation artifacts; a new
training run is not guaranteed to reproduce their parameters or published curves.

## Reproducible new runs

The public training entry explicitly seeds Python, NumPy, and PyTorch. Instance
sampling, validation splitting, and loader shuffling use explicit generators.
The selected train/validation indices are saved in `split.json`, and configuration
and software versions in `run.json`. The validation subset comes from training
cities; it is separate from the Miami evaluation set.

The solver's existing constructor seed behavior is retained. The numerical update
equations, graph architecture, feature normalization, and power projection have
not been changed. Explicit seeding can change a new run relative to an earlier
interactive session whose random state was not recorded.

Training checkpoints include optimizer and scheduler states, configuration, and
PyTorch/loader RNG states. Python/NumPy RNG states and an exact-resume command are
not provided. The code always writes a checkpoint at the end of each epoch,
including runs shorter than the validation interval. Existing output checkpoints
are protected against accidental replacement unless `overwrite` is enabled.

For a short forward/backward and validation check:

```bash
python -m scripts.train --smoke --output-dir results/training-smoke
```

This uses four generated instances and one optimizer batch. It does not establish
convergence or full-training reproducibility. See `VALIDATION.md` for measured checks.
