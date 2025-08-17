import logging
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import modal

log = logging.getLogger(__name__)


# Use an S3 bucket to supply documents to functions running on Modal
# https://modal.com/docs/guide/cloud-bucket-mounts
# The IAM user must be configured to have access to the bucket for document
# storage and the bucket for litellm caching.
s3_bucket_name = "meet-your-new-client"
aws_secret = modal.Secret.from_name(
    name="aws-s3-access",
    required_keys=[
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "LITELLM_CACHE_BUCKET",
        "LITELLM_CACHE_AWS_REGION",
    ],
)

bucket_mount = modal.CloudBucketMount(s3_bucket_name, secret=aws_secret, read_only=True)

app = modal.App(
    "meet-your-new-client",
    secrets=[
        modal.Secret.from_name("openai-secret"),
        modal.Secret.from_name("fireworks-secret"),
        aws_secret,
    ],
)

volume = modal.Volume.from_name("meet-your-new-client", create_if_missing=True)

litellm_config_path = Path(__file__).parent.parent / "config/litellm_config.yaml"


def start_litellm_proxy() -> subprocess.Popen:
    proc = subprocess.Popen(["litellm", "--config", "/root/litellm_config.json"])
    return proc


def check_proxy_health() -> bool:
    """
    Check the health readiness endpoint of a locally running litellm proxy.
    """
    import requests

    try:
        response = requests.get("http://localhost:4000/health/readiness", timeout=1)
    except requests.exceptions.ConnectionError:
        return False

    return response.status_code == 200


def wait_for_proxy_start(timeout: int = 30):
    """
    Wait until a locally running litellm proxy is ready.
    """
    # Check if the proxy is ready to accept connections
    proxy_ready = False
    for _ in range(timeout):
        if check_proxy_health():
            proxy_ready = True
            print("LiteLLM proxy is ready")
            break

        time.sleep(1)

    if not proxy_ready:
        raise RuntimeError(
            f"LiteLLM proxy did not become ready within {timeout} seconds"
        )


@contextmanager
def timing_context():
    start = time.time()
    yield
    end = time.time()
    print(f"Elapsed time: {(end - start):.2f}s")


# MARKITDOWN
# Model and litellm proxy is optional
# Needs ffmpeg
# Doesn't benefit from GPU
markitdown_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(["requests", "markitdown[all]==0.1.2", "openai", "litellm[proxy]"])
    .add_local_file(litellm_config_path, "/root/litellm_config.json")
    .add_local_python_source("src")
)


@app.function(image=markitdown_image, volumes={"/bucket": bucket_mount}, timeout=1800)
def run_markitdown_modal(file: str, model: str | None = None) -> str:
    """
    Convert a document that is located on a mounted bucket to Markdown using MarkItDown.
    """
    from src.convert_markitdown import run_markitdown

    proxy = None
    if model:
        proxy = start_litellm_proxy()
        wait_for_proxy_start()

    with timing_context():
        md = run_markitdown(file="/bucket/" + file, model=model)

    if proxy:
        proxy.terminate()

    return md


# MARKER
# Needs to connect to litellm proxy
# Has custom image captioning code
# Benefits from GPU
# Models need to be pre-fetched for performance reasons at runtime,
# but there is no easy command to do this
# https://github.com/datalab-to/marker/issues/48
# Easiest way is to convert a single example PDF which is created using reportlab
marker_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["libglib2.0-0", "libpango-1.0-0", "libpangoft2-1.0-0"])
    .pip_install(
        ["requests", "marker-pdf[full]>=1.7.5", "openai", "litellm[proxy]", "reportlab"]
    )
    .run_commands(
        [
            "python -c \"from reportlab.pdfgen import canvas; c = canvas.Canvas('/tmp/test.pdf'); c.drawString(100, 750, 'Test'); c.save()\"",
            "marker_single /tmp/test.pdf",
        ]
    )
    .run_commands(["marker_single /tmp/test.pdf"])
    .add_local_file(litellm_config_path, "/root/litellm_config.json")
    .add_local_python_source("src")
)


@app.function(
    image=marker_image,
    volumes={"/bucket": bucket_mount},
    gpu="T4",
    max_containers=10,
    timeout=3600,
)
def run_marker_modal(file: str, model: str) -> str:
    from src.convert_marker import run_marker

    proxy = start_litellm_proxy()
    wait_for_proxy_start()

    with timing_context():
        md = run_marker(file_path="/bucket/" + file, model=model)

    proxy.terminate()

    return md


# ZEROX
# Needs to connect to litellm proxy
# Needs poppler
# Doesn't benefit from GPU
zerox_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("poppler-utils")
    .pip_install(["requests", "py-zerox==0.0.7", "openai", "litellm[proxy]"])
    .add_local_file(litellm_config_path, "/root/litellm_config.json")
    .add_local_python_source("src")
)


@app.function(image=zerox_image, volumes={"/bucket": bucket_mount}, timeout=900)
def run_zerox_modal(file: str, model: str) -> str:
    """
    Convert a document that is located on a mounted bucket to Markdown using Zerox.
    """
    from src.convert_zerox import run_zerox

    proxy = start_litellm_proxy()
    wait_for_proxy_start()

    with timing_context():
        md = run_zerox(file_path="/bucket/" + file, model=model)

    proxy.terminate()

    return md


# DOCLING
# Pre-fetch the models and bake them into the image for faster start times
# https://docling-project.github.io/docling/usage/#model-prefetching-and-offline-usage
# Add litellm proxy to route and cache LLM requests
docling_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(["requests", "docling==2.38.0", "litellm[proxy]"])
    .run_commands("docling-tools models download")
    .add_local_file(litellm_config_path, "/root/litellm_config.json")
    .add_local_python_source("src")
)


@app.function(
    image=docling_image,
    volumes={"/bucket": bucket_mount},
    gpu="A10G",
    max_containers=10,
    timeout=1800,
)
def run_docling_modal(file: str, model: str, img_prompt: str) -> str:
    """
    Convert a document that is located on a mounted bucket to Markdown using Docling.
    """
    from src.convert_docling import run_docling

    proxy = start_litellm_proxy()
    wait_for_proxy_start()

    with timing_context():
        md = run_docling(
            input_doc_path="/bucket/" + file, model=model, img_prompt=img_prompt
        )

    proxy.terminate()

    return md


def main(files: list[str], lib: str, model: str, img_prompt: str):
    if len(files) == 0:
        return []

    log.info(
        f"Converting {len(files)} document(s) to markdown using {lib} with {model}"
    )

    # Define arguments for parallel remote conversion function invocations
    args = []
    for path in files:
        # Only docling needs an img_prompt
        if lib == "docling":
            args.append((path, model, img_prompt))
        else:
            args.append((path, model))

    # Execute conversions in parallel
    # https://modal.com/docs/guide/scale
    if lib == "docling":
        md_strings = list(run_docling_modal.starmap(args, return_exceptions=True))
    elif lib == "markitdown":
        md_strings = list(run_markitdown_modal.starmap(args, return_exceptions=True))
    elif lib == "zerox":
        md_strings = list(run_zerox_modal.starmap(args, return_exceptions=True))
    elif lib == "marker":
        md_strings = list(run_marker_modal.starmap(args, return_exceptions=True))
    else:
        raise ValueError(f"Unknown library {lib}")

    assert len(md_strings) == len(files)

    error_count = 0
    for i, md in enumerate(md_strings):
        if not isinstance(md, str):
            error_msg = f"Error converting {files[i]} to markdown: {md}"
            log.error(error_msg)
            md_strings[i] = error_msg
            error_count += 1

    if (error_count) > 0:
        log.error(f"{error_count} errors occurred during conversion.")

    return md_strings
