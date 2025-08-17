import asyncio
import tempfile

from pyzerox import zerox


def run_zerox(file_path: str, model: str) -> str:
    """Synchronous wrapper for the async zerox function.

    The zerox library saves output files to a directory. This function uses a
    temporary directory to store these files, which is deleted after the
    function completes.
    """

    async def run_zerox_async():
        with tempfile.TemporaryDirectory() as temp_dir:
            md_content = await zerox(
                file_path=file_path,
                model=model,
                output_dir=temp_dir,
            )
            return md_content

    out = asyncio.run(run_zerox_async())
    return str(out)
