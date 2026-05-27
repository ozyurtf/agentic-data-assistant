import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CHATBOT_DIR = Path(__file__).parent.parent / "chatbot"
sys.path.insert(0, str(CHATBOT_DIR))

from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

from cli import run_question

client = Client()

DATASET_OUTPUT_INSTRUCTIONS = {
    "Data Extraction": (
        "Return only a Python-dict-like mapping of the message type(s) and field name(s) "
        "needed to answer the question. No prose, no values. "
        'Example: {"VFR_HUD": ["airspeed"]}'
    ),
    "External Knowledge Awareness": (
        "Return a concise factual answer in one short sentence. No prose padding."
    ),
    "External Knowledge Usage": (
        "Return only the final numeric value. No units, no prose, no formatting. "
        "Example: 0.11241193860769272"
    ),
    "Extractive": (
        "Return only the final value as a single number. No prose, no units, no labels, "
        "no formatting, no explanation. Example: 1533737338.3569999 or 0"
    ),
    "Multi Step Reasoning": (
        "Return only the final numeric value. No units, no prose, no formatting. "
        "Example: 1.4897782750350912"
    ),
    "Multi Task": (
        "Return a JSON list with one entry per sub-question, each in the form "
        '{"sub_question": "...", "answer": <value>}. No prose outside the JSON.'
    ),
    "Not Found": (
        "If the requested data cannot be found in the available log, respond with "
        "exactly the string: Not found"
    ),
    "Out of Scope": (
        "Return only the final answer as a short value with a trailing period. "
        "No prose, no explanation. Example: 391. or 12."
    ),
}


def make_target(dataset_name: str):
    instruction = DATASET_OUTPUT_INSTRUCTIONS.get(dataset_name, "")

    def target(inputs: dict) -> dict:
        question = inputs["question"]
        if instruction:
            question = f"{question}\n\nOUTPUT FORMAT: {instruction}"
        answer = asyncio.run(run_question(question))
        return {"answer": answer}

    return target


def llm_as_judge(inputs: dict, outputs: dict, reference_outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="anthropic:claude-haiku-4-5",
        feedback_key="correctness",
    )
    return evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    )


DATASET_EVALUATORS = {
    # "Data Extraction": [llm_as_judge],
    # "External Knowledge Awareness": [llm_as_judge],
    # "External Knowledge Usage": [llm_as_judge],
    # "Extractive": [llm_as_judge],
    # "Multi Step Reasoning": [llm_as_judge],
    # "Multi Task": [llm_as_judge],
    "Not Found": [llm_as_judge],
    # "Out of Scope": [llm_as_judge],
}


if __name__ == "__main__":
    for dataset_name, evaluators in DATASET_EVALUATORS.items():
        client.evaluate(
            make_target(dataset_name),
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=f"Experiment - {dataset_name}",
            max_concurrency=3,
        )
