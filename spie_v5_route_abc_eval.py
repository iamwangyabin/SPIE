import argparse
import json
import logging
from pathlib import Path
from typing import Dict

from expert_response_analysis import iter_config_seeds, load_json
from spie_v2_route_abc_eval import run_one_seed, setup_parser as setup_base_parser


SUPPORTED_MODEL_NAMES = {"spie_v5", "spiev5"}


def setup_parser() -> argparse.ArgumentParser:
    parser = setup_base_parser()
    parser.description = (
        "Evaluate SPiE v5 route/classification decomposition: "
        "A=P(t_hat=t*), B=P(y_hat=y|t_hat=t*), C=P(y_hat=y|t_hat!=t*)."
    )
    return parser


def validate_config(config: Dict) -> None:
    model_name = str(config.get("model_name", "")).lower()
    if model_name not in SUPPORTED_MODEL_NAMES:
        raise ValueError(
            "spie_v5_route_abc_eval.py expects a SPiE v5 config with "
            f"model_name in {sorted(SUPPORTED_MODEL_NAMES)}, got {config.get('model_name')!r}."
        )


def main() -> None:
    cli_args = setup_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [spie_v5_route_abc_eval] %(message)s")
    config = load_json(cli_args.config)
    validate_config(config)

    seeds = iter_config_seeds(config, cli_args.seed)
    if cli_args.checkpoint and len(seeds) > 1 and cli_args.seed is None:
        logging.warning("--checkpoint was set with multiple config seeds; evaluating the first seed only.")
        seeds = seeds[:1]

    results = [run_one_seed(config, cli_args, seed=seed) for seed in seeds]
    output = results[0] if len(results) == 1 else results

    print(json.dumps(output, indent=2, allow_nan=True))
    if cli_args.output:
        output_path = Path(cli_args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(output, f, indent=2, allow_nan=True)
        logging.info("Saved summary to %s", output_path)


if __name__ == "__main__":
    main()
