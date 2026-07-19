from datetime import datetime
import os

import anthropic

from tools import TOOLS, run_sql

client = anthropic.Anthropic(
    api_key=os.environ["LITELLM_API_KEY"],
    base_url=os.environ["LITELLM_BASE_URL"],
)

MODEL = "us.anthropic.claude-sonnet-4-6"

SYSTEM = f""" You are an expert data analyst investigating a retail SQLite database.

            Today's date is {datetime.now().date().isoformat()}.

            The database is the only source of truth. Never assume facts that are not supported by query results.

            Work autonomously and investigate systematically.

            Start with aggregate queries to understand the data before examining individual records.

            Use small, focused SQL queries rather than one large query.

            For every investigation:
            - Establish what normal looks like.
            - Form hypotheses.
            - Verify each hypothesis with additional queries.
            - Support every conclusion with evidence.

            Always examine:
            - transaction volume
            - duplicate transactions
            - pricing
            - timing
            - data quality

            Do not stop after finding one result. Continue until every required dimension has been examined.

            Never ask the user for permission to continue.

            Your final report must include an executive summary, evidence for every finding, concrete transaction IDs when applicable, and any remaining uncertainties.
        """


def run_agent(question: str) -> str:
    """
    Run a Claude agent with the given question and tools.
    """

    MAX_ITERATIONS = 15

    messages = [{"role": "user", "content": question}]

    iteration = 0 
    while iteration < MAX_ITERATIONS:

        iteration += 1
        
        print(f"\n--- iteration {iteration} ---")

        response = client.messages.create(
            model=MODEL, max_tokens=4096, tools=TOOLS, system=SYSTEM, messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    
                    if block.name == "run_sql":
                        
                        print(f"→ run_sql: {block.input['query']}")
                        
                        output = run_sql(block.input['query'])
                    
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,  # must match the request
                                "content": output,
                            }
                        )

            messages.append({"role": "user", "content": tool_results})

        else:
            # The model produced a final answer; return it.
            for block in response.content:
                if block.type == "text":
                    return block.text

            break

    print(f"the loop exhausts {MAX_ITERATIONS} iterations.")
    
    final_message = { "type": "text", "text": "you're out of query budget; report your findings so far."}
    
    messages[-1]["content"].append(final_message)

    response = client.messages.create(
            model=MODEL, max_tokens=4096, tools=TOOLS, tool_choice={"type": "none"}, system=SYSTEM, messages=messages,
        )
    
    for block in response.content:
        if block.type == "text":
            return block.text

if __name__ == "__main__":
    question = """
    Anything unusual in store 42's transactions yesterday?
    """
    
    print(run_agent(question))