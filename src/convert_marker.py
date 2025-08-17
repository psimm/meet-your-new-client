import logging

from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

log = logging.getLogger(__name__)


def run_marker(file_path: str, model: str | None) -> str:
    """Convert PDF to markdown using Marker with optional LLM-based image captioning."""
    config = {"output_format": "markdown"}

    if model is not None:
        # connect to locally running litellm proxy
        config.update(
            {
                "use_llm": True,
                "llm_service": "marker.services.openai.OpenAIService",
                "openai_model": model,
                "openai_api_key": "local",
                "openai_base_url": "http://0.0.0.0:4000",
                "disable_image_extraction": True,
            }
        )

    config_parser = ConfigParser(config)

    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )

    rendered = converter(file_path)
    result, _, _ = text_from_rendered(rendered)

    assert len(result) > 0, f"Marker result for {file_path} is empty."
    return result
