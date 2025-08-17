import json
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic.dataclasses import dataclass


@dataclass
class PathsConfig:
    reports_dir: Path
    questions_file: Path
    markdown_dir: Path
    answers_file: Path
    evaluated_answers_file: Path
    cache_dir: Path


@dataclass
class StepsConfig:
    convert: bool
    answer: bool
    judge: bool


class ConversionLib(str, Enum):
    docling = "docling"
    markitdown = "markitdown"
    zerox = "zerox"
    marker = "marker"


@dataclass
class ConversionConfig:
    model: str
    lib: ConversionLib
    img_prompt: str
    temperature: float
    suffix: str | None = None
    sample_first_n: int | None = None
    target_files: list[str] | None = None
    read_cache: bool = False
    write_cache: bool = False
    retry_cached_failures: bool = False

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v < 0 or v > 2:
            raise ValueError(f"Invalid temperature: {v}")
        return v

    @field_validator("img_prompt")
    @classmethod
    def validate_img_prompt(cls, v: str) -> str:
        assert len(v) > 0, "img_prompt can't be empty"
        return v


@dataclass
class AnswerConfig:
    model: str
    temperature: float
    prompt: str

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v < 0 or v > 2:
            raise ValueError(f"Invalid temperature: {v}")
        return v

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if "{report_content}" not in v or "{question}" not in v:
            raise ValueError("Prompt must contain {report_content} and {question}")
        return v


@dataclass
class JudgeConfig:
    model: str
    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if "{answer}" not in v or "{ground_truth}" not in v:
            raise ValueError("Prompt must contain {answer} and {ground_truth}")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v < 0 or v > 2:
            raise ValueError(f"Invalid temperature: {v}")
        return v


@dataclass
class RunConfig:
    """Configuration for the experiment"""

    paths: PathsConfig
    steps: StepsConfig
    convert: ConversionConfig
    answer: AnswerConfig
    judge: JudgeConfig

    @classmethod
    def from_yaml(cls, yaml_path: str | Path):
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        return cls(**cfg)


class Question(BaseModel):
    """Pydantic model for a question"""

    report_name: str
    question_id: str
    question: str
    ground_truth: str
    slide_number: int
    layout_element: str

    @field_validator("report_name")
    @classmethod
    def validate_report_name(cls, v: str) -> str:
        if not v:
            raise ValueError("report_name cannot be empty")
        return v


class Evaluation(BaseModel):
    """Pydantic model for evaluation results"""

    reasoning: str = Field(description="Step-by-step reasoning about the judgement")
    correct: bool = Field(description="Whether the answer matches the ground truth")


class ReportAnswer(BaseModel):
    """Pydantic model for an answer from a specific report"""

    report_filename: str
    answer: str
    model: str
    conversion_error: bool = False
    evaluation: Evaluation | None = None


class QuestionAnswer(Question):
    """Pydantic model for storing a question with its answers from multiple reports"""

    report_answers: list[ReportAnswer] = Field(default_factory=list)


def write_json(data: Any, file_path: str | Path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
