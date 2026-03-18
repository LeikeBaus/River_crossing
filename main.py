from core.config_loader import load_all_configs


if __name__ == "__main__":
    env_cfg, algo_cfg, exp_cfg = load_all_configs("configs")
    print("Configs loaded successfully")
    print(f"Grid: {env_cfg['grid']['nx']}x{env_cfg['grid']['ny']}")
    print(f"Algorithms: {len(exp_cfg['algorithms'])}")
