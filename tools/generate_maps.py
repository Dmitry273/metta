import os
import signal  # Aggressively exit on ctrl+c

import hydra
from metta.sim.vecenv import make_vecenv
from metta.util.config import config_from_path
from metta.util.runtime_configuration import setup_mettagrid_environment

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg):
    setup_mettagrid_environment(cfg)
    cfg.eval.env = config_from_path(cfg.eval.env, cfg.eval.env_overrides)
    make_vecenv(cfg.eval.env, cfg.vectorization, num_envs=1, render_mode="human")

if __name__ == "__main__":
    main()
