import os
from dotenv import load_dotenv
import lm_eval
from lm_eval.models.openai_completions import OpenAIChatCompletionRegistry
from lm_eval.utils import setup_logging

load_dotenv()

def run_test():
    setup_logging()
    
    # Run evaluation
    results = lm_eval.simple_evaluate(
        model="openai-chat-completions",
        model_args="model=gpt-4o",
        tasks=["copa_es"],
        limit=2
    )
    
    # Print results table
    from lm_eval.utils import make_table
    print(make_table(results))
    
    # Save results to file
    import json
    with open("results.json", "w") as f:
        json.dump(results["results"], f, indent=2)

if __name__ == "__main__":
    run_test()
