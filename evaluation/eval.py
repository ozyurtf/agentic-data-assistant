# pip install -U langsmith openevals openai

# export LANGSMITH_TRACING=true
# export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
# export LANGSMITH_API_KEY=<your-langsmith-api-key>
# export OPENAI_API_KEY=<your-openai-api-key>

from langsmith import Client, wrappers
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from openai import OpenAI

client = Client()

# Wrap the OpenAI client for LangSmith tracing
openai_client = wrappers.wrap_openai(OpenAI())

# Define what you're evaluating
# Dataset inputs are automatically sent to this target function.
def target(inputs: dict) -> dict:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer the following question accurately"},
            {"role": "user", "content": inputs["question"]},
        ],
    )
    return {"answer": response.choices[0].message.content}


# Define an LLM-as-a-judge evaluator to evaluate correctness of the output
def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="openai:o3-mini",
        feedback_key="correctness",
    )
    eval_result = evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    )
    return eval_result

# Run experiment
experiment_results = client.evaluate(
    target,
    data="ds-essential-conference-73",
    evaluators=[correctness_evaluator],
    experiment_prefix="experiment-quickstart-frosty-publisher-6",
    max_concurrency=2,
)