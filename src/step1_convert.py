import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Literal
from functools import partial

import polars as pl
from pydantic import validate_call

from src.convert_modal import main as batch_convert
from src.convert_modal import app
from src.utils import RunConfig

log = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of converted markdown files."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        log.debug(f"Using cache directory: {self.cache_dir}")

    def _get_cache_key(
        self,
        file_path: str,
        lib: Literal["markitdown", "docling", "zerox", "marker"],
        model: str | None,
    ) -> str:
        """
        Generate a unique hash for a converted markdown result.
        The hash is unique by file content, library and model used.
        """
        model_str = model or "no_model"

        file_stat = os.stat(file_path)
        # Use basename in key to avoid path-dependent keys
        file_basename = os.path.basename(file_path)

        cache_key_data = f"{file_basename}_{file_stat.st_mtime}_{lib}_{model_str}"
        return hashlib.md5(cache_key_data.encode()).hexdigest()

    @validate_call
    def get(
        self,
        file_path: str | Path,
        lib: Literal["markitdown", "docling", "zerox", "marker"],
        model: str | None,
        retry_cached_failures: bool = True,
    ) -> str | None:
        """
        Check for a cached result and return it if found.
        Args:
            file_path: Path to the file to check for a cached result.
            lib: Conversion library used ("markitdown", "docling", "zerox", or "marker").
            model: Model name used for conversion, or None.
            retry_cached_failures: If True, return None for lookups where the
                cached result indicates a conversion error.
        """
        cache_key = self._get_cache_key(file_path, lib, model)
        cache_file = self.cache_dir / f"{cache_key}.md"

        if cache_file.exists():
            text = cache_file.read_text()
            if retry_cached_failures and text.startswith("Error converting"):
                log.info(
                    f"Skipping cached result for {os.path.basename(file_path)} (cached file: {cache_file}) because it has a conversion error"
                )
                return None
            else:
                log.info(
                    f"Using cached result for {os.path.basename(file_path)} (cached file: {cache_file})"
                )
                return text
        else:
            log.debug(f"No cached result found for {os.path.basename(file_path)}")
            return None

    @validate_call
    def put(
        self,
        md_string: str,
        file_path: str | Path,
        lib: Literal["markitdown", "docling", "zerox", "marker"],
        model: str | None,
    ):
        """Cache a new markdown result."""
        cache_key = self._get_cache_key(file_path, lib, model)
        cache_file = self.cache_dir / f"{cache_key}.md"
        cache_file.write_text(md_string)
        log.debug(
            f"Cached result for {os.path.basename(file_path)} with key {cache_key}"
        )


def get_report_files(report_dir: str | Path, suffix: str | None = None) -> list[str]:
    """
    Scan directory for report files. Returns PDF and PPTX files by default,
    or files with specified suffix. Files are sorted alphabetically.
    """
    log.info(f"Scanning for report files in {report_dir}")
    report_dir = Path(report_dir)
    if suffix is None:
        patterns = ["**/*.pdf", "**/*.pptx"]
        report_files = []
        for pattern in patterns:
            report_files.extend(str(p) for p in report_dir.glob(pattern))
    else:
        report_files = [str(p) for p in report_dir.glob(f"**/*{suffix}")]

    report_files.sort()
    return report_files


def generate_md_path(file, markdown_dir):
    # Get the base filename without extension and create markdown path
    basename_split = os.path.splitext(os.path.basename(file))
    filename = basename_split[0]
    file_extension = basename_split[1].replace(".", "")

    md_path = os.path.join(markdown_dir, f"{filename}_from_{file_extension}.md")

    return md_path


def main(cfg: RunConfig):
    """Convert reports to markdown format"""

    log.info("Starting report conversion")

    # Find files
    files = get_report_files(cfg.paths.reports_dir, suffix=cfg.convert.suffix)

    gen_md_path = partial(generate_md_path, markdown_dir=cfg.paths.markdown_dir)

    # Use a data frame to organize file metadata
    files_df = (
        pl.DataFrame({"file": files})
        .with_columns(
            pl.col("file")
            .map_elements(os.path.basename, return_dtype=pl.Utf8)
            .alias("file_basename")
        )
        .with_columns(
            pl.col("file_basename")
            .map_elements(gen_md_path, return_dtype=pl.Utf8)
            .alias("md_path")
        )
    )

    # Filter files according to conversion config
    if files_df.height == 0:
        log.warning("No files found in the reports directory")
        return
    elif cfg.convert.target_files:
        files_df = files_df.filter(
            pl.col("file_basename").is_in(cfg.convert.target_files)
        )
        log.info(f"Using {files_df.height} files specified in target_files")
    elif cfg.convert.sample_first_n is not None:
        files_df = files_df.head(cfg.convert.sample_first_n)
        log.info(f"Using first {files_df.height} files")
    else:
        log.info(f"Using all {files_df.height} files")

    # Find cached markdown files
    if cfg.convert.read_cache:
        cache = CacheManager(cfg.paths.cache_dir)
        cache_getter = partial(
            cache.get,
            lib=cfg.convert.lib.value,
            model=cfg.convert.model,
            retry_cached_failures=cfg.convert.retry_cached_failures,
        )

        files_df = files_df.with_columns(
            pl.col("file")
            .map_elements(cache_getter, return_dtype=pl.Utf8)
            .alias("md_string")
        ).with_columns(pl.col("md_string").is_not_null().alias("read_from_cache"))

        n_cached = files_df["read_from_cache"].sum()
        log.info(f"Using cached markdown files for {n_cached} documents")
    else:
        files_df = files_df.with_columns(
            pl.Series(name="md_string", values=[None] * files_df.height, dtype=pl.Utf8),
            pl.Series(
                name="read_from_cache",
                values=[False] * files_df.height,
                dtype=pl.Boolean,
            ),
        )
        log.info("Skipping cached markdown files lookup")

    # Identify the files that need to be converted
    files_to_convert = files_df.filter(~pl.col("read_from_cache"))

    if files_to_convert.height > 0:
        start_time = time.time()

        with app.run():
            md_strings = batch_convert(
                files=files_to_convert.get_column("file_basename").to_list(),
                lib=cfg.convert.lib.value,
                model=cfg.convert.model,
                img_prompt=cfg.convert.img_prompt,
            )
        end_time = time.time()
        conversion_time = end_time - start_time
        seconds_per_file = conversion_time / len(files_to_convert)
        log.info(
            f"Converted {len(files_to_convert)} documents to markdown in {conversion_time:.2f}s ({seconds_per_file:.2f}s/document)"
        )

        files_to_convert = files_to_convert.with_columns(
            pl.Series(name="md_string", values=md_strings, dtype=pl.Utf8)
        )

        # Put the result strings back into the files_df
        files_df = files_df.join(
            files_to_convert.select("file", "md_string"), on="file", how="left"
        ).with_columns(pl.coalesce("md_string_right", "md_string").alias("md_string"))
    else:
        log.info(f"All {files_df.height} files were found in cache")

    assert files_df["md_string"].null_count() == 0

    # Write all results to output dir
    os.makedirs(cfg.paths.markdown_dir, exist_ok=True)
    for row in files_df.to_dicts():
        with open(row["md_path"], "w") as f:
            f.write(row["md_string"])

    # Copy new results to cache
    if cfg.convert.write_cache:
        cache = CacheManager(cfg.paths.cache_dir)
        for row in files_df.filter(~pl.col("read_from_cache")).to_dicts():
            cache.put(
                md_string=row["md_string"],
                file_path=row["file"],
                lib=cfg.convert.lib.value,
                model=cfg.convert.model,
            )
    else:
        log.debug("Skipping writing to cache")
