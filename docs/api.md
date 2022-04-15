# API

The API is a web application to host generation. It runs as a standalone server
and is built on top of Flask.

Currently, the API supports two endpoints:
- `/generate` - which generates for a fixed number of tokens with hardcoded
  generation parameters.
- `/completions` - which allows for setting sampling/beam parameters.

## Launching the API

**Prerequisites**

Complete all of the setup as mentioned in [the Setup doc](setup.md).

**Run locally on a worker**

```bash
metaseq-api-local
```

Note that when you ctrl-C, you may need to also run "killall python" or otherwise
manually kill the processes. This is due to the batching thread, which is not
properly ended on an interrupt.

**Launching a worker from SLURM**

```bash
srun --ntasks-per-node 1 --gpus-per-node 8 --nodes 1 --cpus-per-task 8 --mem 400gb \
    --quit-on-interrupt --job-name genwork \
    python3 -m metaseq_cli.interactive_hosted
```

## FAQ

**How fast is generation?**

Slow. As of 2022-04-7, QPS is about 40 with 3 workers.

**Where can I run this?**

Right now only on Azure, as it requires the 80GB A100s. To enable it on other
locations, we need to either try CPU offloading, or we need to use MP 16. FSDP
should *not* be used because some workers will only be used for parameter
hosting, and will not actually perform computations.
