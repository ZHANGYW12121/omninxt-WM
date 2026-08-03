import atexit
import pathlib
import sys
import warnings

import hydra
import torch

import tools
from buffer import Buffer, FactorizedBuffer
from dreamer import Dreamer
from factorized_agent import FactorizedDreamerAgent
from envs import make_envs
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    logger = tools.Logger(logdir)
    # save config
    logger.log_hydra_config(config)

    factorized = bool(config.model.get("factorized", {}).get("enabled", False))
    replay_buffer = FactorizedBuffer(config.buffer) if factorized else Buffer(config.buffer)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    print("Simulate agent.")
    agent_cls = FactorizedDreamerAgent if factorized else Dreamer
    agent = agent_cls(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)
    if config.get("resume"):
        checkpoint = torch.load(pathlib.Path(config.resume).expanduser(), map_location=config.device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        if factorized and "factorized_training_state" in checkpoint:
            agent.load_training_state_dict(checkpoint["factorized_training_state"])
        print("Resumed", config.resume)

    policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs)
    policy_trainer.begin(agent)

    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    if hasattr(agent, "training_state_dict"):
        items_to_save["factorized_training_state"] = agent.training_state_dict()
    torch.save(items_to_save, logdir / "latest.pt")


if __name__ == "__main__":
    main()
