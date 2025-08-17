import concurrent.futures
import logging
import os
from pathlib import Path
from typing import Any

import litellm
from litellm.caching.caching import Cache
from botocore.config import Config as BotocoreConfig
from litellm.utils import ModelResponse
from pydantic import validate_call
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm
import yaml


logger = logging.getLogger(__name__)

litellm.cache = Cache(
    type="s3",
    s3_bucket_name=os.environ["LITELLM_CACHE_BUCKET"],
    s3_region_name=os.environ["LITELLM_CACHE_AWS_REGION"],
    s3_config=BotocoreConfig(
        max_pool_connections=int(os.environ.get("LITELLM_MAX_S3_CONNECTIONS"), 10)
    ),
)


def resolve_litellm_model_name(model: str) -> str:
    """
    Look up the short model name in the litellm config file and return the full
    path of the model.
    """
    litellm_config_path = Path(__file__).parent.parent / "config/litellm_config.yaml"
    with open(litellm_config_path, "r") as f:
        litellm_config = yaml.safe_load(f)

    model_list = litellm_config.get("model_list", [])
    for model_entry in model_list:
        if model_entry.get("model_name") == model:
            if litellm_params := model_entry.get("litellm_params"):
                if resolved_model := litellm_params.get("model"):
                    return resolved_model

    raise ValueError(f"model_name {model} not found in {litellm_config_path}")


@validate_call(validate_return=True, config=dict(arbitrary_types_allowed=True))
def batch_completion(
    chats: list[list[dict[str, Any]]],
    model: str,
    temperature: float | None = None,
    timeout: float = 300.0,
    workers: int = int(os.environ.get("LITELLM_WORKERS", 4)),
    retries: int = int(os.environ.get("LITELLM_RETRIES", 5)),
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
) -> list[ModelResponse]:
    """
    Run a batch of chat completions using litellm. Requests are cached.
    A progress bar is displayed and logs written.

    Args:
        chats: The chats to complete. Can be a single Chat or a list of Chats.
        model: The model to use for completion.
        temperature: The temperature to use for completions. Set to None
            to not pass this argument to the API. Not all models support setting
            the temperature.
        timeout: The timeout in seconds for an individual completion.
        workers: The number of workers to use. If 1, the completions are run
            sequentially. If > 1, the completions are run in parallel using
            concurrent.futures.ThreadPoolExecutor. This doesn't consider rate
            limits, so don't set it too high.
        retries: The number of times to retry a completion if it fails.
        tools: The tools to use for completions.
        tool_choice: The tool choice to use for completions.

    Returns:
        List of completions.
    """
    if len(chats) == 0:
        return []

    model = resolve_litellm_model_name(model)

    logger.info(f"Running {len(chats)} completion(s) using {model}")

    assert retries >= 0, "Retries can't be negative"
    assert workers > 0, "Workers must be greater than 0"

    # Prepare arguments for completions
    arg_dicts = []
    for chat in chats:
        arg_dict = {"messages": chat, "model": model, "timeout": timeout}
        if temperature is not None:
            arg_dict["temperature"] = temperature
        if tools is not None:
            arg_dict["tools"] = tools
        if tool_choice is not None:
            arg_dict["tool_choice"] = tool_choice
        arg_dicts.append(arg_dict)

    # LiteLLM's max_retries parameter doesn't work reliably and lacks
    # customization, so we use tenacity to retry completions
    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(retries + 1),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type((KeyboardInterrupt, litellm.BadRequestError)),
    )
    def generate_completion(arg_dict: dict) -> litellm.utils.ModelResponse:
        return litellm.completion(**arg_dict)

    # Execute completions
    if len(arg_dicts) < workers:
        # It doesn't make sense to use more workers than completions
        workers = len(arg_dicts)
        logger.debug(f"More workers than completions. Reducing workers to {workers}")

    if workers == 1:
        logger.info("Running completions sequentially")

        responses = [
            generate_completion(arg_dict)
            for arg_dict in tqdm(arg_dicts, desc="Completions")
        ]

    else:
        logger.info(f"Running completions in parallel using {workers} workers")

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            responses = list(
                tqdm(
                    executor.map(generate_completion, arg_dicts),
                    total=len(arg_dicts),
                    desc="Completions",
                )
            )

    return responses
