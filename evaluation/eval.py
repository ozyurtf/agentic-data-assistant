from langsmith import evaluate, Client
from dotenv import load_dotenv
from firecrawl import Firecrawl

load_dotenv()

# 1. Create and/or select your dataset
client = Client()
dataset_name = "MAVLink Questions"

# 2. Define an evaluator
def llm_as_judge(outputs: dict, reference_outputs: dict) -> bool:
    return outputs == reference_outputs

def generate_answer(inputs: dict) -> str: 
    question = inputs["question"]
    return f"The answer is not known for this question: {question}"

# 3. Run an evaluation
# For more info on evaluators, see: https://docs.langchain.com/langsmith/evaluation-concepts
evaluate(
    generate_answer,
    data=dataset_name,
    evaluators=[exact_match],
    experiment_prefix="MAVLink Questions experiment"
)