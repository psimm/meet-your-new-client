import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.utils import (
    Question,
    QuestionAnswer,
    ReportAnswer,
    RunConfig,
    write_json,
)
from src.llm import batch_completion

log = logging.getLogger(__name__)

load_dotenv()


def load_questions(questions_file: str | Path) -> list[Question]:
    """Load questions from a JSON file and validate using Pydantic."""
    log.info(f"Loading questions from {questions_file}")
    with open(questions_file, "r") as f:
        questions_data = json.load(f)

    questions = [Question(**q_data) for q_data in questions_data]
    log.info(f"Loaded and validated {len(questions)} questions")
    return questions


def find_matching_reports(reports_dir: str | Path, report_name: str) -> list[str]:
    """Find all .md report files starting with the given report_name."""
    matching_reports = []
    for filename in os.listdir(reports_dir):
        if filename.startswith(report_name + "_from") and filename.endswith(".md"):
            matching_reports.append(os.path.join(reports_dir, filename))

    if not matching_reports:
        log.debug(f"No matching reports found for {report_name}")
    else:
        log.debug(f"Found {len(matching_reports)} matching reports for {report_name}")

    return matching_reports


def read_report_content(report_path: str | Path) -> str:
    """Read the content of a report file."""
    log.debug(f"Reading report: {report_path}")
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content


def process_questions(
    questions: list[Question],
    reports_dir: str | Path,
    model: str,
    prompt: str,
    temperature: float,
) -> list[QuestionAnswer]:
    """Process all questions against matching reports."""
    question_answers = []
    chats_to_process = []
    chat_metadata = []

    # Find matching reports and prepare chats
    for question in questions:
        question_answer = QuestionAnswer(**question.model_dump())
        question_answers.append(question_answer)

        log.debug(
            f"Processing question_id: {question.question_id}, report_name: {question.report_name}"
        )

        matching_reports = find_matching_reports(reports_dir, question.report_name)
        if not matching_reports:
            log.debug(
                f"No matching reports found for {question.report_name} (question_id: {question.question_id})"
            )
            continue

        # Add a chat for each converted report
        for report_path in matching_reports:
            content = read_report_content(report_path)
            report_filename = os.path.basename(report_path)

            if content.startswith("Error converting"):
                # This means the report was not converted successfully
                # There is no point in using it for answering questions
                log.info(
                    f"Skipping report: {report_filename} because it was not converted successfully"
                )
                report_answer = ReportAnswer(
                    report_filename=report_filename,
                    model=model,
                    conversion_error=True,
                    answer="CONVERSION_ERROR",
                )
                question_answer.report_answers.append(report_answer)
                continue

            # Create chat
            chat = [
                {
                    "role": "user",
                    "content": prompt.format(
                        question=question.question, report_content=content
                    ),
                }
            ]
            chats_to_process.append(chat)
            chat_metadata.append(
                {"question_answer": question_answer, "report_filename": report_filename}
            )

    if not chats_to_process:
        log.info("No valid reports found to generate answers from.")
        return question_answers

    log.info(f"Processing {len(chats_to_process)} chats with {model}")
    responses = batch_completion(
        chats=chats_to_process, model=model, temperature=temperature
    )

    # Store model answers in the report_answer objects
    for response, metadata in zip(responses, chat_metadata):
        answer_content = response.choices[0].message.content
        report_answer = ReportAnswer(
            report_filename=metadata["report_filename"],
            answer=answer_content,
            model=model,
        )
        metadata["question_answer"].report_answers.append(report_answer)

    return question_answers


def main(cfg: RunConfig):
    """
    Process questions by finding matching reports and querying an LLM for answers.
    Each matching report is processed separately to get individual answers.
    """
    log.info("Starting to answer questions")

    questions = load_questions(questions_file=cfg.paths.questions_file)

    results = process_questions(
        questions=questions,
        reports_dir=cfg.paths.markdown_dir,
        model=cfg.answer.model,
        prompt=cfg.answer.prompt,
        temperature=cfg.answer.temperature,
    )

    result_dicts = [result.model_dump() for result in results]
    write_json(result_dicts, cfg.paths.answers_file)

    log.info(f"Answers generated successfully and saved to {cfg.paths.answers_file}")
