"""
Lab | Your First Agent  --  Google ADK

A single agent that owns two tools (lookup_order, calculate) and solves
a multi-step goal on its own: reason -> act (tool call) -> observe -> repeat.

Setup:
    pip install -r requirements.txt
    export GOOGLE_API_KEY="your-free-gemini-key"

Run:
    python agent.py
"""

import asyncio
import json
import os
from pathlib import Path

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

ORDERS_PATH = Path(__file__).parent / "orders.json"

APP_NAME = "orders_assistant"
USER_ID = "user_1"
MODEL = "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def lookup_order(order_id: str) -> dict:
    """Looks up an order by its ID and returns its details.

    Args:
        order_id: The order ID to look up, e.g. "A1001".

    Returns:
        A dict with keys "found" (bool) and, if found, "item", "price",
        "purchased" (purchase date), and "warranty_months". If not found,
        includes an "error" message instead.
    """
    with open(ORDERS_PATH, "r") as f:
        orders = json.load(f)

    order = orders.get(order_id)
    if order is None:
        return {"found": False, "error": f"No order found with ID '{order_id}'."}

    return {
        "found": True,
        "item": order["item"],
        "price": order["price"],
        "purchased": order["purchased"],
        "warranty_months": order["warranty_months"],
    }


def calculate(expression: str) -> dict:
    """Evaluates a simple arithmetic expression, e.g. "1200 * 2" or "30 + 4*80".

    Args:
        expression: An arithmetic expression using only numbers and
            + - * / ( ) and whitespace.

    Returns:
        A dict with "result" (float) on success, or "error" (str) on failure.
    """
    allowed_chars = set("0123456789.+-*/() ")
    if not set(expression) <= allowed_chars:
        return {"error": f"Expression '{expression}' contains disallowed characters."}
    try:
        # Safe-ish eval: no names/builtins available, only arithmetic chars allowed above.
        result = eval(expression, {"__builtins__": {}}, {})
        return {"result": result}
    except Exception as e:
        return {"error": f"Could not evaluate '{expression}': {e}"}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

orders_agent = Agent(
    name="orders_assistant",
    model=MODEL,
    instruction=(
        "You are a helpful orders assistant for an online store. "
        "You help customers with questions about their past orders: price, "
        "purchase date, warranty status, and any math involving those numbers "
        "(totals, multiples, discounts, etc.).\n\n"
        "Tools available to you:\n"
        "- lookup_order(order_id): look up an order's item, price, purchase "
        "date, and warranty length. ALWAYS use this instead of guessing or "
        "remembering order details.\n"
        "- calculate(expression): evaluate arithmetic. ALWAYS use this "
        "instead of doing math yourself, so the numbers are exact.\n\n"
        "When reasoning about warranty status, today's date is 2026-07-03. "
        "Compare the purchase date plus warranty_months against today to "
        "decide if an order is still covered.\n\n"
        "If lookup_order reports that an order was not found, say so clearly "
        "and honestly to the user. Never invent order details, prices, or "
        "warranty information that a tool did not return."
    ),
    tools=[lookup_order, calculate],
)


# ---------------------------------------------------------------------------
# Runner / trace capture
# ---------------------------------------------------------------------------

async def run_goal(goal: str, session_id: str, session_service: InMemorySessionService):
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    runner = Runner(
        app_name=APP_NAME, agent=orders_agent, session_service=session_service
    )

    print("=" * 80)
    print(f"GOAL: {goal}")
    print("=" * 80)

    user_message = types.Content(role="user", parts=[types.Part(text=goal)])

    final_answer = None
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=user_message
    ):
        if not event.content or not event.content.parts:
            continue
        for part in event.content.parts:
            if getattr(part, "function_call", None):
                fc = part.function_call
                print(f"\n[TOOL CALL] {fc.name}({dict(fc.args)})")
            elif getattr(part, "function_response", None):
                fr = part.function_response
                print(f"[TOOL RESULT] {fr.name} -> {fr.response}")
            elif getattr(part, "text", None):
                print(f"\n[MODEL] {part.text}")
                if event.is_final_response():
                    final_answer = part.text

    print("\n" + "-" * 80)
    print(f"FINAL ANSWER: {final_answer}")
    print("-" * 80 + "\n")
    return final_answer


async def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit(
            "Set GOOGLE_API_KEY before running, e.g.:\n"
            '  export GOOGLE_API_KEY="your-free-gemini-key"'
        )

    session_service = InMemorySessionService()

    # Multi-step goal: requires lookup_order + calculate + reasoning about warranty.
    await run_goal(
        "I'm thinking of buying two more of order A1001. What would those "
        "two cost, and is the original still under warranty?",
        session_id="session_multi_step",
        session_service=session_service,
    )

    # Stretch goal: an order that doesn't exist. Agent should report honestly.
    await run_goal(
        "Can you tell me the price and warranty status of order A9999?",
        session_id="session_not_found",
        session_service=session_service,
    )


if __name__ == "__main__":
    asyncio.run(main())
