from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import RolloutConfig


def test_rollout_config_accepts_async_skd_agent_fields_from_hydra_overrides():
    cfg = OmegaConf.create(
        {
            "_target_": "verl.workers.config.RolloutConfig",
            "name": "sglang",
            "agent": {
                "_target_": "verl.workers.config.AgentLoopConfig",
                "agent_loop_manager_class": "verl.experimental.async_skd.manager.AsyncSkdAgentLoopManager",
                "async_skd_mode": "lookahead",
                "async_skd_prefetch_limit": 8,
                "async_skd_max_old_gen_chunks": 64,
            },
        }
    )

    rollout_cfg = omega_conf_to_dataclass(cfg, RolloutConfig)

    assert rollout_cfg.agent.agent_loop_manager_class == "verl.experimental.async_skd.manager.AsyncSkdAgentLoopManager"
    assert rollout_cfg.agent.async_skd_mode == "lookahead"
    assert rollout_cfg.agent.async_skd_prefetch_limit == 8
    assert rollout_cfg.agent.async_skd_max_old_gen_chunks == 64
