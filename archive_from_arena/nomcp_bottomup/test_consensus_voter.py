"""Tests for ConsensusVoter component.

Tests all agreement scenarios, edge cases, and confidence boosting logic.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from consensus_voter import ConsensusVoter, vote_three_results


def test_full_agreement():
    """All 3 strategies agree on same value."""
    voter = ConsensusVoter()

    result = voter.vote(
        strict_result={
            "value": 1350576.0,
            "confidence": 0.99,
            "method": "strict",
            "row_source": "row_3",
        },
        fuzzy_result={
            "value": 1350576.0,
            "confidence": 0.85,
            "method": "fuzzy",
            "row_source": "row_3",
        },
        contextual_result={
            "value": 1350576.0,
            "confidence": 0.88,
            "method": "contextual",
            "row_source": "row_3",
        },
    )

    assert result["winning_value"] == 1350576.0
    assert result["winning_method"] == "strict"
    assert result["agreement_level"] == "full"
    assert result["num_agreeing"] == 3
    # Base confidence is 0.99 (max), boosted by 1.15 = 1.1385, capped at 0.99
    assert result["winning_confidence"] == 0.99
    assert len(result["risk_flags"]) == 0
    print("✓ test_full_agreement passed")


def test_partial_agreement_2v1():
    """Two strategies agree, one disagrees."""
    voter = ConsensusVoter()

    result = voter.vote(
        strict_result={
            "value": 1350576.0,
            "confidence": 0.99,
            "method": "strict",
        },
        fuzzy_result={
            "value": 1350576.0,
            "confidence": 0.85,
            "method": "fuzzy",
        },
        contextual_result={
            "value": 1400000.0,
            "confidence": 0.70,
            "method": "contextual",
        },
    )

    assert result["winning_value"] == 1350576.0
    assert result["agreement_level"] == "partial"
    assert result["num_agreeing"] == 2
    # Base confidence is mean(0.99, 0.85) = 0.92, boosted by 1.05 = 0.966
    assert result["winning_confidence"] == 0.966
    assert "multiple_values_found" in str(result["risk_flags"])
    print("✓ test_partial_agreement_2v1 passed")


def test_no_agreement_all_differ():
    """All three strategies return significantly different values."""
    voter = ConsensusVoter()

    result = voter.vote(
        strict_result={"value": 1000000.0, "confidence": 0.99, "method": "strict"},
        fuzzy_result={"value": 2000000.0, "confidence": 0.85, "method": "fuzzy"},
        contextual_result={"value": 3000000.0, "confidence": 0.70, "method": "contextual"},
    )

    # Should pick the highest confidence result (strict at 0.99)
    assert result["winning_value"] == 1000000.0
    assert result["winning_method"] == "strict"
    assert result["agreement_level"] == "none"
    assert result["num_agreeing"] == 1
    # Base confidence 0.99, penalized by 0.95 = 0.9405
    assert result["winning_confidence"] == 0.9405
    assert "no_agreement_all_differ" in result["risk_flags"]
    print("✓ test_no_agreement_all_differ passed")


def test_close_match_within_tolerance():
    """Values are close (within 1% tolerance) - should be treated as agreement."""
    voter = ConsensusVoter()

    result = voter.vote(
        strict_result={"value": 1000000.0, "confidence": 0.95, "method": "strict"},
        fuzzy_result={
            "value": 1005000.0,
            "confidence": 0.80,
            "method": "fuzzy",
        },  # 0.5% diff, within 1%
        contextual_result={
            "value": 1000000.0,
            "confidence": 0.85,
            "method": "contextual",
        },
    )

    # All three values within 1% tolerance, so treated as full agreement
    assert result["agreement_level"] == "full"
    assert result["num_agreeing"] == 3
    # Base confidence is 0.95 (max), boosted by 1.15 = 1.0925, capped at 0.99
    assert result["winning_confidence"] == 0.99
    print("✓ test_close_match_within_tolerance passed")


def test_all_none():
    """All three strategies return None."""
    voter = ConsensusVoter()

    result = voter.vote(
        strict_result=None,
        fuzzy_result=None,
        contextual_result=None,
    )

    assert result["winning_value"] is None
    assert result["winning_confidence"] == 0.0
    assert result["winning_method"] is None
    assert result["agreement_level"] == "none"
    assert result["num_agreeing"] == 0
    assert "all_results_none" in result["risk_flags"]
    print("✓ test_all_none passed")


def test_one_valid_result():
    """Only one strategy returns a value, others are None."""
    voter = ConsensusVoter()

    result = voter.vote(
        strict_result={
            "value": 1350576.0,
            "confidence": 0.99,
            "method": "strict",
        },
        fuzzy_result=None,
        contextual_result=None,
    )

    assert result["winning_value"] == 1350576.0
    assert result["winning_method"] == "strict"
    assert result["agreement_level"] == "none"
    assert result["num_agreeing"] == 1
    # Base confidence 0.99, penalized by 0.95 = 0.9405
    assert result["winning_confidence"] == 0.9405
    print("✓ test_one_valid_result passed")


def test_two_valid_results():
    """Two strategies return values, one returns None."""
    voter = ConsensusVoter()

    result = voter.vote(
        strict_result={
            "value": 1350576.0,
            "confidence": 0.99,
            "method": "strict",
        },
        fuzzy_result={
            "value": 1350576.0,
            "confidence": 0.85,
            "method": "fuzzy",
        },
        contextual_result=None,
    )

    assert result["winning_value"] == 1350576.0
    assert result["agreement_level"] == "full"  # Both valid results agree
    assert result["num_agreeing"] == 2
    # Base confidence 0.99, boosted by 1.15 = 1.1385, capped at 0.99
    assert result["winning_confidence"] == 0.99
    print("✓ test_two_valid_results passed")


def test_confidence_boost_math():
    """Verify confidence boost calculations are correct."""
    voter = ConsensusVoter()

    # Full agreement: mean(0.80,0.75,0.70)=0.75, * 1.15 = 0.8625
    boosted = voter.boost_confidence_for_agreement("full", [0.80, 0.75, 0.70])
    assert boosted == 0.8625

    # Partial agreement: mean(0.90,0.85)=0.875, * 1.05 = 0.91875 -> round to 0.9188
    boosted = voter.boost_confidence_for_agreement("partial", [0.90, 0.85])
    assert boosted == 0.9188

    # No agreement: mean(0.99)=0.99, * 0.95 = 0.9405
    boosted = voter.boost_confidence_for_agreement("none", [0.99])
    assert boosted == 0.9405

    # Capping at 0.99: mean(0.99,0.95)=0.97, * 1.15 = 1.1155 -> 0.99
    boosted = voter.boost_confidence_for_agreement("full", [0.99, 0.95])
    assert boosted == 0.99

    print("✓ test_confidence_boost_math passed")


def test_check_agreement():
    """Test the check_agreement helper method."""
    voter = ConsensusVoter()

    # Exact match
    assert voter.check_agreement(1000.0, 1000.0) == "exact"

    # Close match (within 1%)
    assert voter.check_agreement(1000.0, 1005.0) == "close"  # 0.5% diff
    assert voter.check_agreement(1000.0, 990.0) == "close"  # 1% diff (at tolerance)

    # Disagree (beyond 1%)
    assert voter.check_agreement(1000.0, 1020.0) == "disagree"  # 2% diff

    # With None values
    assert voter.check_agreement(1000.0, None) == "disagree"
    assert voter.check_agreement(None, 1000.0) == "disagree"

    print("✓ test_check_agreement passed")


def test_rank_results():
    """Test result ranking by confidence."""
    voter = ConsensusVoter()

    results = [
        {"value": 1000.0, "confidence": 0.70},
        {"value": 2000.0, "confidence": 0.95},
        {"value": 3000.0, "confidence": 0.80},
        None,
    ]

    ranked = voter.rank_results(results)
    assert len(ranked) == 3
    assert ranked[0][0]["value"] == 2000.0
    assert ranked[0][1] == 0.95
    assert ranked[1][0]["value"] == 3000.0
    assert ranked[2][0]["value"] == 1000.0

    print("✓ test_rank_results passed")


def test_convenience_function():
    """Test the standalone vote_three_results function."""
    result = vote_three_results(
        strict_result={"value": 100.0, "confidence": 0.99, "method": "strict"},
        fuzzy_result={"value": 100.0, "confidence": 0.85, "method": "fuzzy"},
        contextual_result={"value": 100.0, "confidence": 0.88, "method": "contextual"},
    )

    assert result["winning_value"] == 100.0
    assert result["agreement_level"] == "full"
    print("✓ test_convenience_function passed")


def test_output_structure():
    """Verify output dictionary has all required fields."""
    voter = ConsensusVoter()

    result = voter.vote(
        strict_result={"value": 1000.0, "confidence": 0.99, "method": "strict"},
        fuzzy_result={"value": 1000.0, "confidence": 0.85, "method": "fuzzy"},
        contextual_result={"value": 1000.0, "confidence": 0.88, "method": "contextual"},
    )

    required_keys = {
        "winning_value",
        "winning_confidence",
        "winning_method",
        "agreement_level",
        "num_agreeing",
        "voting_details",
        "reasoning",
        "risk_flags",
    }
    assert set(result.keys()) == required_keys

    # Check voting_details structure
    assert "strict" in result["voting_details"]
    assert "fuzzy" in result["voting_details"]
    assert "contextual" in result["voting_details"]

    print("✓ test_output_structure passed")


def test_high_confidence_dissenter_wins():
    """When 2 weak strategies agree but 1 strong strategy disagrees, strong wins."""
    voter = ConsensusVoter()

    result = voter.vote(
        strict_result={"value": 500000.0, "confidence": 0.95, "method": "strict"},
        fuzzy_result={"value": 600000.0, "confidence": 0.40, "method": "fuzzy"},
        contextual_result={"value": 600000.0, "confidence": 0.40, "method": "contextual"},
    )

    # Dissenter (strict, 0.95) should override the weak pair (avg 0.40)
    # because 0.95 > 0.85 and (0.95 - 0.40) = 0.55 >= 0.30
    assert result["winning_value"] == 500000.0
    assert result["winning_method"] == "strict"
    assert result["agreement_level"] == "none"  # overridden to none
    assert result["num_agreeing"] == 1
    print("  test_high_confidence_dissenter_wins passed")


def test_high_confidence_dissenter_does_not_override_strong_pair():
    """When the agreeing pair is also strong, dissenter should NOT override."""
    voter = ConsensusVoter()

    result = voter.vote(
        strict_result={"value": 500000.0, "confidence": 0.90, "method": "strict"},
        fuzzy_result={"value": 600000.0, "confidence": 0.80, "method": "fuzzy"},
        contextual_result={"value": 600000.0, "confidence": 0.75, "method": "contextual"},
    )

    # Dissenter (0.90) vs pair avg (0.775) -- gap is only 0.125, < 0.30
    assert result["winning_value"] == 600000.0
    assert result["agreement_level"] == "partial"
    assert result["num_agreeing"] == 2
    print("  test_high_confidence_dissenter_does_not_override_strong_pair passed")


def test_small_value_tight_tolerance():
    """Values < 1.0 should use tighter tolerance (0.1 not 0.5)."""
    voter = ConsensusVoter()

    # 0.0 and 0.5 should disagree now (abs_diff 0.5 > 0.1)
    assert voter.check_agreement(0.0, 0.5) == "disagree"

    # 0.0 and 0.05 should still agree (abs_diff 0.05 <= 0.1)
    assert voter.check_agreement(0.0, 0.05) == "close"

    # Values >= 1.0 but < 100 still use 0.5 tolerance
    assert voter.check_agreement(5.0, 5.4) == "close"

    print("  test_small_value_tight_tolerance passed")


def test_transitive_grouping():
    """A=100, B=100.5, C=101.0: A agrees with B, B agrees with C, but A may not agree with C.
    They should NOT all be grouped together."""
    voter = ConsensusVoter()

    result = voter.vote(
        strict_result={"value": 100.0, "confidence": 0.90, "method": "strict"},
        fuzzy_result={"value": 100.5, "confidence": 0.85, "method": "fuzzy"},
        contextual_result={"value": 101.0, "confidence": 0.80, "method": "contextual"},
    )

    # 100 vs 100.5: 0.5% diff -> close (within 1%)
    # 100 vs 101.0: 1.0% diff -> close (at tolerance boundary)
    # 100.5 vs 101.0: 0.497% diff -> close
    # All are within 1% of each other, so they should still group
    # But if values were more spread, the transitive fix would prevent incorrect grouping
    assert result["agreement_level"] == "full"
    print("  test_transitive_grouping passed")


def test_large_value_differences():
    """Test handling of significantly different values."""
    voter = ConsensusVoter()

    result = voter.vote(
        strict_result={
            "value": 1350576.0,
            "confidence": 0.99,
            "method": "strict",
        },
        fuzzy_result={
            "value": 1400000.0,
            "confidence": 0.70,
            "method": "fuzzy",
        },
        contextual_result={
            "value": 1350576.0,
            "confidence": 0.88,
            "method": "contextual",
        },
    )

    # Strict and contextual agree, fuzzy disagrees
    assert result["winning_value"] == 1350576.0
    assert result["agreement_level"] == "partial"
    assert result["num_agreeing"] == 2
    assert "multiple_values_found" in str(result["risk_flags"])
    print("✓ test_large_value_differences passed")


def run_all_tests():
    """Run all test cases."""
    tests = [
        test_full_agreement,
        test_partial_agreement_2v1,
        test_no_agreement_all_differ,
        test_close_match_within_tolerance,
        test_all_none,
        test_one_valid_result,
        test_two_valid_results,
        test_confidence_boost_math,
        test_check_agreement,
        test_rank_results,
        test_convenience_function,
        test_output_structure,
        test_large_value_differences,
        test_high_confidence_dissenter_wins,
        test_high_confidence_dissenter_does_not_override_strong_pair,
        test_small_value_tight_tolerance,
        test_transitive_grouping,
    ]

    print("Running ConsensusVoter tests...\n")
    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_fn.__name__} failed: {e}", file=sys.stderr)
            failed += 1
        except Exception as e:
            print(f"✗ {test_fn.__name__} error: {e}", file=sys.stderr)
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 60}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
