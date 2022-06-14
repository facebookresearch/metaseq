from metaseq.logging.progress_bar.base_progress_bar import (
    BaseProgressBar,
    logger,
)


try:
    import functools

    from aim import Repo as AimRepo

    @functools.lru_cache()
    def get_aim_run(repo, run_hash):
        from aim import Run

        return Run(run_hash=run_hash, repo=repo)

except ImportError:
    get_aim_run = None
    AimRepo = None


class AimProgressBarWrapper(BaseProgressBar):
    """Log to Aim."""

    def __init__(self, wrapped_bar, aim_repo, aim_run_hash, aim_param_checkpoint_dir):
        self.wrapped_bar = wrapped_bar

        if get_aim_run is None:
            self.run = None
            logger.warning("Aim not found, please install with: pip install aim")
        else:
            logger.info(f"Storing logs at Aim repo: {aim_repo}")
            assert AimRepo is not None

            if not aim_run_hash:
                # Find run based on save_dir parameter
                query = f"run.checkpoint.save_dir == '{aim_param_checkpoint_dir}'"
                try:
                    runs_generator = AimRepo(aim_repo).query_runs(query)
                    run = next(runs_generator.iter_runs())
                    aim_run_hash = run.run.hash
                except Exception:
                    pass

            if aim_run_hash:
                logger.info(f"Appending to run: {aim_run_hash}")

            self.run = get_aim_run(aim_repo, aim_run_hash)

    def __iter__(self):
        return iter(self.wrapped_bar)

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats to Aim."""
        self._log_to_aim(stats, tag, step)
        self.wrapped_bar.log(stats, tag=tag, step=step)

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        self._log_to_aim(stats, tag, step)
        self.wrapped_bar.print(stats, tag=tag, step=step)

    def update_config(self, config):
        """Log latest configuration."""
        if self.run is not None:
            for key in config:
                self.run.set(key, config[key], strict=False)
        self.wrapped_bar.update_config(config)

    def _log_to_aim(self, stats, tag=None, step=None):
        if self.run is None:
            return

        if step is None:
            step = stats["num_updates"]

        if "train" in tag:
            context = {"tag": tag, "subset": "train"}
        elif "val" in tag:
            context = {"tag": tag, "subset": "val"}
        else:
            context = {"tag": tag}

        for key in stats.keys() - {"num_updates"}:
            self.run.track(stats[key], name=key, step=step, context=context)
