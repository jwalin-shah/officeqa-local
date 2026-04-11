"""Consensus voting component for Treasury Bulletin data extraction pipeline.

Compares results from 3 parallel subsearch strategies (Strict, Fuzzy, Contextual)
and picks the best one based on agreement level and confidence weighting.

Three strategies:
  - Strict: Exact matching (highest confidence when it works)
  - Fuzzy: Keyword matching (moderate confidence, broader coverage)
  - Contextual: Structure-based matching (moderate confidence, semantic understanding)

Voting logic:
  - All 3 agree → confidence boost to 0.95+
  - 2 agree → moderate confidence 0.70-0.85
  - 1 result → trust only that strategy's confidence
  - None return None → agreement depends on valid results only
"""

from typing import Any


class ConsensusVoter:
    """Vote on 3 parallel subsearch strategy results and pick the best value."""

    # Tolerance for numeric comparison
    # For values >= 100: relative tolerance (0.01 = 1%)
    # For values < 100: absolute tolerance (0.5)
    DEFAULT_TOLERANCE = 0.01
    DEFAULT_ABS_TOLERANCE = 0.5
    SMALL_VALUE_THRESHOLD = 100

    def vote(
        self,
        strict_result: dict[str, Any],
        fuzzy_result: dict[str, Any],
        contextual_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare 3 results from parallel subsearch strategies and return winning consensus.

        Args:
            strict_result: {"value": float, "confidence": float, "method": "strict", ...}
            fuzzy_result: {"value": float, "confidence": float, "method": "fuzzy", ...}
            contextual_result: {"value": float, "confidence": float, "method": "contextual", ...}

        Returns:
            {
                'winning_value': float or None,
                'winning_confidence': float,
                'winning_method': str,
                'agreement_level': 'full' | 'partial' | 'none',
                'num_agreeing': int,
                'voting_details': {
                    'strict': {...},
                    'fuzzy': {...},
                    'contextual': {...}
                },
                'reasoning': str,
                'risk_flags': [str]
            }
        """
        # Collect all results with their methods
        results = {
            "strict": strict_result,
            "fuzzy": fuzzy_result,
            "contextual": contextual_result,
        }

        # Extract valid values (non-None) and their sources
        valid_values = {}
        for method, result in results.items():
            if result is not None and result.get("value") is not None:
                valid_values[method] = result

        # Handle all None case
        if not valid_values:
            return {
                "winning_value": None,
                "winning_confidence": 0.0,
                "winning_method": None,
                "agreement_level": "none",
                "num_agreeing": 0,
                "voting_details": {
                    "strict": strict_result or {},
                    "fuzzy": fuzzy_result or {},
                    "contextual": contextual_result or {},
                },
                "reasoning": "All three strategies returned None or no value",
                "risk_flags": ["all_results_none"],
            }

        # Check agreement among valid values
        agreement_groups = self._group_by_agreement(valid_values)

        # Find the largest agreement group (highest vote count)
        best_group = max(agreement_groups, key=lambda g: len(g["methods"]))

        # Determine agreement level
        total_valid = len(valid_values)
        num_agreeing = len(best_group["methods"])
        agreement_level = self._classify_agreement(num_agreeing, total_valid)

        # Override: when partial (2 vs 1), check if the dissenter has much
        # higher confidence than the agreeing pair.  If so, prefer the dissenter.
        if agreement_level == "partial" and num_agreeing == 2 and len(agreement_groups) == 2:
            dissent_group = [g for g in agreement_groups if g is not best_group][0]
            if len(dissent_group["methods"]) == 1:
                dissent_conf = dissent_group["results"][0].get("confidence", 0.0)
                agree_confs = [
                    valid_values[m].get("confidence", 0.5) for m in best_group["methods"]
                ]
                agree_avg = sum(agree_confs) / len(agree_confs)
                if dissent_conf > 0.85 and (dissent_conf - agree_avg) >= 0.3:
                    # High-confidence dissenter wins
                    best_group = dissent_group
                    num_agreeing = 1
                    agreement_level = "none"

        # Get the base confidence from the best method in the winning group
        winning_method = best_group["methods"][0]  # Use first method in group
        winning_result = valid_values[winning_method]
        base_confidence = winning_result.get("confidence", 0.5)

        # Boost confidence based on agreement
        boosted_confidence = self.boost_confidence_for_agreement(
            agreement_level,
            [valid_values[m].get("confidence", 0.5) for m in best_group["methods"]],
        )

        # Build reasoning
        reasoning = self._build_reasoning(
            best_group["value"],
            num_agreeing,
            total_valid,
            best_group["methods"],
            base_confidence,
            boosted_confidence,
            agreement_groups,
        )

        # Identify risk flags
        risk_flags = self._identify_risk_flags(
            agreement_groups, agreement_level, boosted_confidence
        )

        return {
            "winning_value": best_group["value"],
            "winning_confidence": boosted_confidence,
            "winning_method": winning_method,
            "agreement_level": agreement_level,
            "num_agreeing": num_agreeing,
            "voting_details": {
                "strict": (
                    {
                        "value": strict_result.get("value"),
                        "confidence": strict_result.get("confidence"),
                        "method": strict_result.get("method", "strict"),
                    }
                    if strict_result
                    else None
                ),
                "fuzzy": (
                    {
                        "value": fuzzy_result.get("value"),
                        "confidence": fuzzy_result.get("confidence"),
                        "method": fuzzy_result.get("method", "fuzzy"),
                    }
                    if fuzzy_result
                    else None
                ),
                "contextual": (
                    {
                        "value": contextual_result.get("value"),
                        "confidence": contextual_result.get("confidence"),
                        "method": contextual_result.get("method", "contextual"),
                    }
                    if contextual_result
                    else None
                ),
            },
            "reasoning": reasoning,
            "risk_flags": risk_flags,
        }

    def check_agreement(
        self, value1: float, value2: float, tolerance: float = DEFAULT_TOLERANCE
    ) -> str:
        """Compare if two numeric values agree.

        For small values (< 100), uses absolute tolerance (0.5) so that
        close values like 4.14 vs 4.13 are not rejected.
        For larger values (>= 100), uses relative (percentage) tolerance.

        Args:
            value1: First value
            value2: Second value
            tolerance: Percentage tolerance for large values (0.01 = 1%)

        Returns:
            'exact' | 'close' | 'disagree'
        """
        if value1 is None or value2 is None:
            return "disagree"

        if value1 == value2:
            return "exact"

        abs_diff = abs(value1 - value2)

        # For small values, use absolute tolerance
        if max(abs(value1), abs(value2)) < self.SMALL_VALUE_THRESHOLD:
            # Tighter tolerance for very small values (both < 1.0)
            if max(abs(value1), abs(value2)) < 1.0:
                effective_abs_tol = 0.1
            else:
                effective_abs_tol = self.DEFAULT_ABS_TOLERANCE
            if abs_diff <= effective_abs_tol:
                return "close"
            else:
                return "disagree"

        # For larger values, use relative (percentage) tolerance
        max_val = max(abs(value1), abs(value2), 1)
        percent_diff = abs_diff / max_val

        if percent_diff <= tolerance:
            return "close"
        else:
            return "disagree"

    def boost_confidence_for_agreement(
        self, agreement_level: str, individual_confidences: list[float]
    ) -> float:
        """Boost confidence score based on agreement level.

        Args:
            agreement_level: 'full', 'partial', or 'none'
            individual_confidences: List of confidence scores from agreeing methods

        Returns:
            Boosted confidence score (capped at 0.99)
        """
        if not individual_confidences:
            return 0.0

        # Use the mean confidence as base (pessimistic: don't let one high score inflate)
        base_confidence = sum(individual_confidences) / len(individual_confidences)

        if agreement_level == "full":
            # All 3 agree: boost by 15% but cap at 0.99
            boosted = min(base_confidence * 1.15, 0.99)
        elif agreement_level == "partial":
            # 2 out of 3 agree: boost by 5%
            boosted = min(base_confidence * 1.05, 0.99)
        else:  # agreement_level == "none"
            # Slight penalty: reduce by 5%
            boosted = base_confidence * 0.95

        return round(boosted, 4)

    def rank_results(self, results: list[dict]) -> list[tuple[dict, float]]:
        """Rank results by confidence and agreement.

        Args:
            results: List of result dictionaries (each with 'value' and 'confidence')

        Returns:
            List of (result, score) tuples sorted by score descending
        """
        ranked = []
        for result in results:
            if result is None or result.get("value") is None:
                continue
            confidence = result.get("confidence", 0.0)
            # Could add more sophisticated scoring here (agreement count, etc.)
            ranked.append((result, confidence))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    # --- Private helper methods ---

    def _group_by_agreement(self, valid_values: dict[str, dict]) -> list[dict]:
        """Group results by agreement on value.

        Returns list of groups:
          [{
              'value': 1350576.0,
              'methods': ['strict', 'fuzzy'],
              'results': [strict_result, fuzzy_result]
           }, ...]
        """
        groups = []

        for method, result in valid_values.items():
            value = result.get("value")
            matched = False

            # Try to match with existing group (must agree with ALL members)
            for group in groups:
                agrees_with_all = all(
                    self.check_agreement(value, r.get("value", 0.0)) != "disagree"
                    for r in group["results"]
                )
                if agrees_with_all:
                    group["methods"].append(method)
                    group["results"].append(result)
                    matched = True
                    break

            # No match: create new group
            if not matched:
                groups.append(
                    {
                        "value": value,
                        "methods": [method],
                        "results": [result],
                    }
                )

        return groups

    def _classify_agreement(self, num_agreeing: int, total_valid: int) -> str:
        """Classify agreement level based on count.

        Args:
            num_agreeing: Number of strategies agreeing
            total_valid: Total number of valid (non-None) results

        Returns:
            'full' | 'partial' | 'none'
        """
        if num_agreeing == total_valid and total_valid >= 2:
            return "full"
        elif num_agreeing >= 2:
            return "partial"
        else:
            return "none"

    def _build_reasoning(
        self,
        winning_value: float,
        num_agreeing: int,
        total_valid: int,
        agreeing_methods: list[str],
        base_confidence: float,
        boosted_confidence: float,
        all_groups: list[dict],
    ) -> str:
        """Build human-readable reasoning string."""
        if num_agreeing == total_valid and total_valid >= 2:
            return (
                f"All {total_valid} strategies agreed on value {winning_value}, "
                f"boosting confidence from {base_confidence:.2f} to {boosted_confidence:.2f}"
            )
        elif num_agreeing >= 2:
            disagreers = []
            for group in all_groups:
                if group["value"] != winning_value:
                    disagreers.extend(group["methods"])

            return (
                f"{num_agreeing} out of {total_valid} strategies agreed on {winning_value} "
                f"({', '.join(agreeing_methods)}). "
                f"Disagreers: {', '.join(disagreers)}. "
                f"Confidence: {base_confidence:.2f} -> {boosted_confidence:.2f}"
            )
        else:
            return (
                f"Only {agreeing_methods[0]} returned a valid value ({winning_value}). "
                f"Using its confidence: {boosted_confidence:.2f}"
            )

    def _identify_risk_flags(
        self, all_groups: list[dict], agreement_level: str, confidence: float
    ) -> list[str]:
        """Identify potential risk flags in the voting result."""
        flags = []

        # Flag if no agreement
        if agreement_level == "none":
            flags.append("no_agreement_all_differ")

        # Flag if confidence is very low
        if confidence < 0.5:
            flags.append("low_confidence")

        # Flag if there are multiple distinct values
        if len(all_groups) > 1:
            values_str = ", ".join(
                f"{g['value']} (from {', '.join(g['methods'])})" for g in all_groups
            )
            flags.append(f"multiple_values_found: {values_str}")

        return flags


# --- Standalone utility functions ---


def vote_three_results(
    strict_result: dict[str, Any],
    fuzzy_result: dict[str, Any],
    contextual_result: dict[str, Any],
) -> dict[str, Any]:
    """Convenience function: vote on 3 results without instantiating voter.

    Args:
        strict_result: Result from strict strategy
        fuzzy_result: Result from fuzzy strategy
        contextual_result: Result from contextual strategy

    Returns:
        Voting result dictionary (see ConsensusVoter.vote)
    """
    voter = ConsensusVoter()
    return voter.vote(strict_result, fuzzy_result, contextual_result)
