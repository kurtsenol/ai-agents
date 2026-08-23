"""
Step 11 — LangChain, read-only.

Nothing to fill in. Run it and read:  uv run step11_langchain.py

The point: you have been using LangChain since step 9 without naming it.
This shows what its core abstraction actually is, and why it is not the right
tool for an agent loop — which is exactly why LangGraph exists.
"""

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import tool


def model():
    return ChatAnthropic(
        model="us.anthropic.claude-sonnet-4-6",
        api_key=os.environ["LITELLM_API_KEY"],
        base_url=os.environ["LITELLM_BASE_URL"],
        max_tokens=128,
    )


def section(t: str) -> None:
    print(f"\n{'=' * 68}\n{t}\n{'=' * 68}")


# ---------------------------------------------------------------------------
section("1. Everything is a Runnable")
# ---------------------------------------------------------------------------
# LangChain's core idea is one interface. A Runnable has invoke / stream /
# batch / ainvoke, and anything with that interface composes with anything
# else. That is the whole abstraction.

@tool(parse_docstring=True)
def run_sql(query: str) -> str:
    """Run a query.

    Args:
        query: The SQL.
    """
    return "fake"


prompt = ChatPromptTemplate.from_messages(
    [("system", "Answer in one word."), ("human", "{question}")]
)

for name, obj in [
    ("ChatAnthropic", model()),
    ("ChatPromptTemplate", prompt),
    ("@tool run_sql", run_sql),
    ("StrOutputParser", StrOutputParser()),
]:
    print(f"  {name:<22} Runnable? {isinstance(obj, Runnable)}")

print("\n  -> The model, the prompt, the tool and the parser are all the same")
print("     KIND of thing. That is why .invoke() works on all of them.")


# ---------------------------------------------------------------------------
section("2. LCEL: composing with |")
# ---------------------------------------------------------------------------
# Because they share an interface, `|` chains them. Output of the left
# becomes input of the right. This is LangChain Expression Language.

chain = prompt | model() | StrOutputParser()

print(f"  chain type: {type(chain).__name__}")
print(f"  is Runnable: {isinstance(chain, Runnable)}")

answer = chain.invoke({"question": "Türkiye'nin başkenti?"})
print(f"\n  chain.invoke(...) -> {answer!r}")

print("\n  -> A chain is itself a Runnable, so chains nest. Composition is")
print("     the payoff: one interface, arbitrary pipelines.")


# ---------------------------------------------------------------------------
section("3. batch and stream come free")
# ---------------------------------------------------------------------------

results = chain.batch([
    {"question": "2+2?"},
    {"question": "Fransa'nın başkenti?"},
])
print(f"  chain.batch(2 inputs) -> {results}")

print("\n  chain.stream(...): ", end="")
for piece in chain.stream({"question": "İtalya'nın başkenti?"}):
    print(piece, end="", flush=True)
print()

print("\n  -> You implement invoke; the interface gives you batch and stream.")


# ---------------------------------------------------------------------------
section("4. Why a chain cannot be an agent")
# ---------------------------------------------------------------------------
# A chain is a DAG: data flows one way, left to right. That is a feature for
# a pipeline and a dead end for an agent, because an agent loop needs to go
# BACK to the model after running a tool, an unknown number of times.

def fake_tool_step(msg: AIMessage) -> list:
    return [msg, ToolMessage(content="fake result", tool_call_id="x")]


pipeline = RunnableLambda(lambda q: [HumanMessage(content=q)]) | model()

print("  A chain like:  input | model | tools | ???")
print()
print("  There is no `|` that means 'go back to the model until it stops")
print("  asking for tools'. `|` is one arrow forward. To loop you need:")
print("    - somewhere to keep the growing message list   -> STATE")
print("    - something that decides whether to go again   -> CONDITIONAL EDGE")
print()
print("  Those two words are LangGraph. It exists because LCEL is a DAG and")
print("  an agent is a cyclic graph.")

print("\n  Historical note: LangChain's own answer used to be AgentExecutor —")
print("  a prebuilt loop you configured but could not see into. Same problem")
print("  you hit with PydanticAI in steps 3-6: the loop was someone else's.")
print("  LangGraph replaced it by making the loop data.")


# ---------------------------------------------------------------------------
section("5. What we are NOT using, and why")
# ---------------------------------------------------------------------------

print("""
  Retrievers / vector stores  -> you built this yourself in phase 3, and your
                                 proxy has no embeddings endpoint anyway
  Output parsers              -> Pydantic + bind_tools is stronger: the model
                                 is constrained, not corrected after the fact
  AgentExecutor               -> superseded by LangGraph
  Chains for RAG              -> a DAG suits RAG fine; you just do not need
                                 the abstraction to write four function calls

  What IS worth taking from LangChain: the message types, BaseChatModel as a
  provider-neutral seam, and @tool. You are already using exactly those three.
""")


if __name__ == "__main__":
    pass
