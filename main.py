import logging

import hydra
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore

from src.step1_convert import main as convert_main
from src.step2_answer import main as answer_main
from src.step3_judge import main as judge_main
from src.utils import RunConfig

# Load environment variables from .env
load_dotenv()

# Configure logging with Hydra
# https://hydra.cc/docs/tutorials/basic/running_your_app/logging/
log = logging.getLogger(__name__)


cs = ConfigStore.instance()
cs.store(name="config_schema", node=RunConfig)


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def run_experiment(cfg: RunConfig):
    """Run the complete experiment pipeline"""

    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Starting experiment pipeline in {run_dir}")

    if cfg.steps.convert:
        log.info(
            f"Step 1: Converting PPTX/PDF to markdown using {cfg.convert.lib} and the vision model {cfg.convert.model}"
        )
        convert_main(cfg)

    if cfg.steps.answer:
        log.info(f"Step 2: Answering questions using {cfg.answer.model}")
        answer_main(cfg)

    if cfg.steps.judge:
        log.info(f"Step 3: Judging answers using {cfg.judge.model}")
        judge_main(cfg)

    log.info("Experiment pipeline completed")


if __name__ == "__main__":
    run_experiment()
