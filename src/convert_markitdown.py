import logging

from markitdown import MarkItDown
from openai import OpenAI

# Mute pdfminer logs (many irrelevant warnings)
logging.getLogger("pdfminer").setLevel(logging.ERROR)


def run_markitdown(file: str, model: str | None = None):
    """Convert file to markdown using MarkItDown with optional LLM integration."""
    if model is None:
        md = MarkItDown()
    else:
        # Connect to litellm proxy
        client = OpenAI(base_url="http://0.0.0.0:4000", api_key="local")
        md = MarkItDown(llm_client=client, llm_model=model)

    result = md.convert(file).text_content
    return result
