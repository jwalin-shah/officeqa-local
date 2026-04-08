#!/usr/bin/env python3
"""
OfficeQA Solver - Minimal orchestrator
Let the model do the thinking. We just provide grep + python compute tools.
"""

import os
import subprocess
import json
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

# Configuration
CORPUS_DIR = Path("corpus")

# Initialize Anthropic client
client = Anthropic()

SYSTEM_PROMPT = """You are an expert at extracting numerical data from U.S. Treasury Bulletin documents.

TASK FLOW:
1. Analyze the question carefully (fiscal vs calendar year, units, etc.)
2. Use grep_files tool to search the Treasury corpus for relevant tables
3. Use python_compute tool for any calculations (sums, percent change, etc.)
4. Return your final numeric answer

TOOLS AVAILABLE:
- grep_files(pattern, limit=10): Search Treasury TXT files. Returns matching lines.
- python_compute(code): Execute Python for calculations. Set 'result' variable.

CRITICAL RULES:
- Fiscal year FY pre-1977: Jul 1 (Y-1) to Jun 30 (Y)
- Fiscal year FY post-1976: Oct 1 (Y-1) to Sep 30 (Y)
- Calendar year CY: Jan 1 to Dec 31
- Tables show "(in millions)" or "(in thousands)" — match units
- Write your best answer immediately after finding data, then refine

Always return a single numeric value at the end."""


class OfficeQASolver:
    def __init__(self):
        self.conversation_history = []
        self.tools = [
            {
                "name": "grep_files",
                "description": "Search Treasury corpus for text pattern",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["pattern"],
                },
            },
            {
                "name": "python_compute",
                "description": "Execute Python code for calculations",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code. Set 'result' variable with answer.",
                        }
                    },
                    "required": ["code"],
                },
            },
        ]

    def grep_files(self, pattern: str, limit: int = 10) -> str:
        """Search corpus for pattern."""
        try:
            result = subprocess.run(
                ["grep", "-i", "-n", "-r", pattern, str(CORPUS_DIR)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            lines = result.stdout.split("\n")[:limit]
            return "\n".join(lines) if lines else "No matches found"
        except Exception as e:
            return f"Error: {e}"

    def python_compute(self, code: str) -> str:
        """Execute Python code safely."""
        try:
            local_vars = {}
            exec(code, {"__builtins__": {}}, local_vars)
            result = local_vars.get("result", "No result set")
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    def solve(self, question: str) -> str:
        """Solve a question using the model."""
        self.conversation_history = [{"role": "user", "content": question}]

        # Agentic loop
        for iteration in range(15):  # Max 15 turns
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                tools=self.tools,
                messages=self.conversation_history,
            )

            # Check if done
            if response.stop_reason == "end_turn":
                # Extract final answer
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return "No answer generated"

            # Process tool calls
            assistant_message = {"role": "assistant", "content": response.content}
            self.conversation_history.append(assistant_message)

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    if block.name == "grep_files":
                        result = self.grep_files(
                            block.input.get("pattern"),
                            block.input.get("limit", 10),
                        )
                    elif block.name == "python_compute":
                        result = self.python_compute(block.input.get("code"))
                    else:
                        result = "Unknown tool"

                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": result}
                    )

            if not tool_results:
                # No tools called, extract answer
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return "No answer generated"

            self.conversation_history.append({"role": "user", "content": tool_results})

        return "Max iterations reached"


if __name__ == "__main__":
    import sys

    solver = OfficeQASolver()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"\n❓ {question}\n")
        answer = solver.solve(question)
        print(f"✅ {answer}\n")
    else:
        print("Usage: python solve.py <question>")
