#!/usr/bin/env python3
"""
OfficeQA Local Agent
Orchestrator for answering U.S. Treasury Bulletin questions using Claude.
"""

import os
import subprocess
import re
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

# Configuration
CORPUS_DIR = Path("corpus")
CPI_SCRIPT = Path("cpi.py")

# System prompt with domain knowledge
SYSTEM_PROMPT = """You are an expert orchestrator for answering U.S. Treasury Bulletin questions.

DOMAIN KNOWLEDGE:
- Treasury data files in /app/resources/. Page files (*_page_*.txt) have answers.
- Fiscal year vs Calendar year:
  * FY pre-1977: Jul 1 (Y-1) to Jun 30 (Y). FY1976 = Jul 1975–Jun 1976.
  * FY post-1976: Oct 1 (Y-1) to Sep 30 (Y). FY1981 = Oct 1980–Sep 1981.
  * CY: Jan 1 to Dec 31. CY1981 = Jan 1981–Dec 1981.
- Tables label rows as "1955" — check if table is fiscal or calendar from headers.
- Tables show "(in millions)" or "(in thousands)" in headers — match question units.

TOOLS AVAILABLE:
- grep_files(pattern, files): Search for text in Treasury files
- python_compute(code): Execute Python for math (cagr, stdev, linreg, etc.)

YOUR PROCESS:
1. Parse the question carefully (fiscal vs calendar year, units, etc.)
2. Create a search plan to find relevant data
3. Use grep_files to locate tables/rows
4. Use python_compute for calculations
5. Return final answer

Always write your best guess immediately, then refine. Wrong answers get partial credit."""


class TreasuryAgent:
    def __init__(self):
        self.client = Anthropic()
        self.conversation_history = []
        self.corpus_dir = CORPUS_DIR

    def grep_files(self, pattern: str, files: list = None) -> str:
        """Search Treasury files for a pattern."""
        if files is None:
            files = list(self.corpus_dir.glob("*.txt"))
        else:
            files = [self.corpus_dir / f if not Path(f).exists() else Path(f) for f in files]

        results = []
        for file_path in files:
            if not file_path.exists():
                continue
            try:
                output = subprocess.run(
                    ["grep", "-i", "-n", pattern, str(file_path)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if output.stdout:
                    results.append(f"=== {file_path.name} ===\n{output.stdout}")
            except Exception as e:
                continue

        return "\n".join(results) if results else "No matches found."

    def python_compute(self, code: str) -> str:
        """Execute Python code for calculations."""
        try:
            # Safe execution with restricted builtins
            local_vars = {}
            exec(code, {"__builtins__": {}}, local_vars)
            result = local_vars.get("result", "Computation complete")
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    def process_tool_call(self, tool_name: str, tool_input: dict) -> str:
        """Process tool calls from Claude."""
        if tool_name == "grep_files":
            return self.grep_files(tool_input.get("pattern"), tool_input.get("files"))
        elif tool_name == "python_compute":
            return self.python_compute(tool_input.get("code"))
        else:
            return f"Unknown tool: {tool_name}"

    def answer_question(self, question: str) -> str:
        """Answer a Treasury question using Claude."""
        self.conversation_history = []

        # Initial user message
        self.conversation_history.append({"role": "user", "content": question})

        # Agentic loop
        for _ in range(10):  # Max 10 iterations
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=self.conversation_history,
            )

            # Check if we have tool use or stop
            if response.stop_reason == "end_turn":
                # Extract final answer
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return "No answer generated"

            # Process tool calls if present
            assistant_message = {"role": "assistant", "content": response.content}
            self.conversation_history.append(assistant_message)

            has_tool_calls = False
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    has_tool_calls = True
                    tool_result = self.process_tool_call(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": tool_result,
                        }
                    )

            if not has_tool_calls:
                # No more tool calls, extract final answer
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return "No answer generated"

            # Add tool results to conversation
            self.conversation_history.append({"role": "user", "content": tool_results})

        return "Max iterations reached"


def main():
    import sys

    agent = TreasuryAgent()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer = agent.answer_question(question)
        print(f"\n📊 Answer:\n{answer}")
    else:
        print("Usage: python agent.py <question>")


if __name__ == "__main__":
    main()
