from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    PowerpointFormatOption,
)


def setup_docling_converter(
    model: str, img_prompt: str, artifacts_path: str | None = None
) -> DocumentConverter:
    """
    Set up DocumentConverter with LLM-based image description capabilities.

    Args:
        model: Name of the model to use for image captioning. Must be registered
            on a running litellm proxy.
        img_prompt: Prompt to use for the image captioning task.
        artifacts_path: Optional path to pre-fetched Docling models. If not
            supplied, models are found or downloaded at runtime.
    """

    # Enable picture descriptions via remote LLM APIs
    # https://ds4sd.github.io/docling/examples/pictures_description_api/
    picture_description_options = PictureDescriptionApiOptions(
        # Connect to local litellm proxy
        url="http://0.0.0.0:4000/chat/completions",
        headers={"Authorization": "Bearer " + "local"},
        params=dict(model=model),
        prompt=img_prompt,
        timeout=600,
    )

    args = {
        "enable_remote_services": True,
        "do_picture_description": True,
        "picture_description_options": picture_description_options,
    }

    if artifacts_path is not None:
        args["artifacts_path"] = artifacts_path

    # Enable picture descriptions for PDF conversio
    # It's not supported for PPTX, see https://github.com/docling-project/docling/issues/1001
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(**args),
            ),
            InputFormat.PPTX: PowerpointFormatOption(),
        }
    )

    return doc_converter


def run_docling(input_doc_path: str, model: str, img_prompt: str) -> str:
    """Convert document to markdown using Docling with LLM-generated image descriptions."""
    doc_converter = setup_docling_converter(model=model, img_prompt=img_prompt)
    result = doc_converter.convert(input_doc_path)
    markdown_string = result.document.export_to_markdown()
    return markdown_string
