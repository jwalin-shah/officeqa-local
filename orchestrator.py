#!/usr/bin/env python3
"""
OfficeQA Orchestrator - 4 Phase Pipeline
Based on 75% Arena solution: grep beats DB, simple beats complex

Phase 1: Question → Decomposition (LLM)
Phase 2: Search corpus for facts (grep)
Phase 3: Extraction & Compute (LLM + Python)
Phase 4: Synthesize final answer (LLM)
"""

import os
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configuration
CORPUS_DIR = Path("corpus")
DEDALUS_KEY = os.getenv("DEDALUS_API_KEY")
DEDALUS_BASE = os.getenv("DEDALUS_API_BASE", "https://api.dedaluslabs.ai/v1")
MODEL = "deepseek-chat"

# Use requests directly (avoid OpenAI SDK version issues)
import requests

class Orchestrator:
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {DEDALUS_KEY}",
            "Content-Type": "application/json"
        }
        self.conversation = []

    def _call_llm(self, prompt: str, system: str = None) -> str:
        """Call DeepSeek via Dedalus API."""
        messages = [{"role": "user", "content": prompt}]
        if self.conversation:
            messages = self.conversation + messages

        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000,
        }

        try:
            response = requests.post(
                f"{DEDALUS_BASE}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {e}"

    def grep_search(self, pattern: str, limit: int = 20) -> str:
        """
        Search corpus with grep.
        Best patterns: specific table names, year numbers, metric keywords
        """
        try:
            result = subprocess.run(
                ["grep", "-i", "-n", "-r", pattern, str(CORPUS_DIR)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            lines = result.stdout.split("\n")[:limit]
            return "\n".join([l for l in lines if l])
        except Exception as e:
            return f"Search error: {e}"

    def phase1_decompose(self, question: str) -> dict:
        """
        Phase 1: Analyze question and create search plan
        Returns: {question_type, search_keywords, expected_computation}
        """
        print("📋 Phase 1: Decompose question...")

        prompt = f"""Analyze this Treasury question and output a JSON search plan.

QUESTION: {question}

Output ONLY valid JSON (no markdown, no extra text):
{{
  "question_type": "lookup|summation|percent_change|average|regression|other",
  "search_keywords": ["keyword1", "keyword2", ...],
  "expected_period": "calendar|fiscal|both",
  "expected_computation": "direct_lookup|sum|percent_change|average|linear_regression|other",
  "critical_rules": ["rule1", "rule2", ...]
}}"""

        response = self._call_llm(prompt)
        try:
            plan = json.loads(response)
        except:
            plan = {
                "question_type": "lookup",
                "search_keywords": [question.split()[0]],
                "expected_period": "calendar",
                "expected_computation": "direct_lookup",
                "critical_rules": []
            }

        self.conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(plan)}
        ]

        print(f"  Type: {plan.get('question_type')}")
        print(f"  Keywords: {plan.get('search_keywords')}")
        return plan

    def phase2_search(self, plan: dict) -> str:
        """
        Phase 2: Search corpus for relevant data
        Returns: raw grep results with table lines
        """
        print("🔍 Phase 2: Search corpus...")

        keywords = plan.get("search_keywords", [])
        results = []

        for keyword in keywords[:5]:  # Search top 5 keywords
            print(f"  → Searching: {keyword}")
            result = self.grep_search(keyword, limit=15)
            if result and "error" not in result.lower():
                results.append(f"=== Results for '{keyword}' ===\n{result}\n")

        all_results = "\n".join(results)
        self.conversation.append({"role": "user", "content": f"Search results:\n{all_results}"})
        return all_results

    def phase3_extract(self, question: str, search_results: str, plan: dict) -> str:
        """
        Phase 3: Given search results, extract values and compute
        Returns: extracted values + computation code
        """
        print("🔎 Phase 3: Extract and compute...")

        prompt = f"""Given the Treasury search results, extract values and provide Python code to compute the answer.

ORIGINAL QUESTION: {question}

SEARCH RESULTS (raw grep output):
{search_results[:2000]}

Extract all numeric values found in the search results and write Python code to compute the final answer.
Focus on: {plan.get('expected_computation', 'direct lookup')}

Respond with ONLY valid JSON (no markdown):
{{
  "extracted_values": {{"label": <number>, ...}},
  "unit": "millions",
  "python_code": "result = <your_code>",
  "reasoning": "brief explanation"
}}"""

        response = self._call_llm(prompt)
        self.conversation.append({"role": "user", "content": prompt})
        self.conversation.append({"role": "assistant", "content": response})

        return response

    def phase4_synthesize(self, extraction: str, question: str) -> str:
        """
        Phase 4: Execute computation and return final answer
        """
        print("✨ Phase 4: Synthesize answer...")

        try:
            # Try to parse extraction as JSON
            try:
                extraction_json = json.loads(extraction)
                code = extraction_json.get("python_code", "result = 0")
            except json.JSONDecodeError:
                # If not JSON, ask LLM to extract the code
                prompt = f"""Extract the Python code from this response:
{extraction}

Respond with ONLY valid JSON:
{{"python_code": "...", "extracted_values": {{}}}}"""
                response = self._call_llm(prompt)
                extraction_json = json.loads(response)
                code = extraction_json.get("python_code", "result = 0")

            # Execute computation safely
            local_vars = {}
            exec(code, {"__builtins__": {}, "sum": sum, "len": len, "float": float, "int": int}, local_vars)
            answer = local_vars.get("result", "unknown")

            return str(answer).strip()

        except Exception as e:
            print(f"  Error in computation: {e}")
            # Last resort: ask LLM to extract answer from results
            prompt = f"""Extract the final numeric answer from this Treasury analysis:
{extraction}

Respond with ONLY a number."""
            return self._call_llm(prompt).strip()

    def solve(self, question: str) -> dict:
        """Full 4-phase orchestration."""
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}\n")

        # Phase 1: Decompose
        plan = self.phase1_decompose(question)

        # Phase 2: Search
        search_results = self.phase2_search(plan)

        # Phase 3: Extract
        extraction = self.phase3_extract(question, search_results, plan)

        # Phase 4: Synthesize
        answer = self.phase4_synthesize(extraction, question)

        print(f"\n✅ FINAL ANSWER: {answer}\n")

        return {
            "question": question,
            "plan": plan,
            "search_results": search_results[:500],
            "extraction": extraction,
            "final_answer": answer
        }


if __name__ == "__main__":
    import sys

    orchestrator = Orchestrator()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = orchestrator.solve(question)
    else:
        print("Usage: python orchestrator.py <question>")
        print("\nExample:")
        print('  python orchestrator.py "What were total expenditures for national defense in 1940?"')
