import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.llm import batch_completion
from src.utils import (
    Evaluation,
    QuestionAnswer,
    RunConfig,
    write_json,
)

log = logging.getLogger(__name__)

load_dotenv()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "Provide a reasoned judgment about the evaluation",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Step-by-step reasoning",
                    },
                    "correct": {
                        "type": "boolean",
                        "description": "Final judgment (true/false) based on the reasoning",
                    },
                },
                "required": ["reasoning", "correct"],
            },
        },
    }
]


def create_eval_chat(
    question: str, answer: str, ground_truth: str, prompt: str
) -> list[dict[str, str]]:
    """Create a chat for evaluation."""
    return [
        {
            "content": prompt.format(
                question=question, answer=answer, ground_truth=ground_truth
            ),
            "role": "user",
        }
    ]


def load_answers(answers_file: str | Path) -> list[QuestionAnswer]:
    """Load answers from a JSON file and validate using Pydantic."""
    log.info(f"Loading answers from {answers_file}")
    with open(answers_file, "r") as f:
        data = json.load(f)

    questions = []
    for q_data in data:
        question = QuestionAnswer(**q_data)
        questions.append(question)

    log.info(f"Loaded and validated {len(questions)} questions with answers")
    return questions


def parse_eval_response(response: dict[str, Any]) -> dict[str, Any]:
    tool_call = response.choices[0].message.tool_calls[0]
    args_str = tool_call.function.arguments

    # Fix a common error where the JSON string is not properly terminated
    if args_str[-1] != "}":
        args_str += "}"

    return json.loads(args_str)


def process_evaluations(
    questions: list[QuestionAnswer], model: str, prompt: str
) -> list[QuestionAnswer]:
    """Process all answers and evaluate them against ground truth."""
    # Prepare evaluation requests
    eval_chats = []
    eval_metadata = []

    # Build evaluation requests
    for question in questions:
        for answer in question.report_answers:
            if answer.conversion_error:
                log.debug(
                    f"Skipping evaluation for question {question.question_id} because it has a conversion error"
                )
                answer.evaluation = Evaluation(
                    reasoning="CONVERSION_ERROR", correct=False
                )
                continue

            eval_chats.append(
                create_eval_chat(
                    question=question.question,
                    answer=answer.answer,
                    ground_truth=question.ground_truth,
                    prompt=prompt,
                )
            )
            eval_metadata.append(
                {"answer": answer, "question_id": question.question_id}
            )

    log.info(f"Processing {len(eval_chats)} evaluations with {model}")

    # Get evaluations in batch
    responses = batch_completion(
        chats=eval_chats,
        model=model,
        tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "answer"}},
    )

    # Process responses
    for metadata, response in zip(eval_metadata, responses):
        answer = metadata["answer"]
        try:
            result = parse_eval_response(response)
            answer.evaluation = Evaluation(
                reasoning=result["reasoning"], correct=result["correct"]
            )

        except Exception as e:
            log.error(
                f"Error processing evaluation for question {metadata['question_id']}: {e}"
            )
            answer.evaluation = Evaluation(
                reasoning=f"EVALUATION_ERROR: {str(e)}", correct=False
            )

    return questions


def main(cfg: RunConfig):
    """
    Evaluate answers against ground truth using an LLM as judge.
    Processes all answers in the input file and adds evaluation results.
    """
    log.info("Starting evaluation process")

    questions = load_answers(cfg.paths.answers_file)
    evaluated_questions = process_evaluations(
        questions=questions,
        model=cfg.judge.model,
        prompt=cfg.judge.prompt,
    )

    result_dicts = [question.model_dump() for question in evaluated_questions]
    write_json(result_dicts, cfg.paths.evaluated_answers_file)

    log.info(
        f"Evaluation complete. Results saved to {cfg.paths.evaluated_answers_file}"
    )
