## Getting started with Training

## Prerequisites
Complete all of the setup as mentioned in [the setup doc](setup.md).

### Launching a small-scale baseline run
```
opt-baselines \
  -n 2 -g 8 \
  -p test_v0 \
  --model-size 125m \
  --azure \
  --checkpoints-dir "INSERT_YOUR_CHECKPOINT_DIR" \
  --no-save-dir  # Remove this if you want to print out full save-dir path
```
#### Bring up tensorboard for the run
```
tensorboard serve --logdir="INSERT_TENSORBOARD_LOGDIR"  --bind_all --port=6018
```
