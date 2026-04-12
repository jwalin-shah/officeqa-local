from eval_attribution import _classify_retrieval


def test_classify_retrieval_empty() -> None:
    stage, detail = _classify_retrieval(
        {
            "total_entries": 0,
            "first_file_hit_rank": None,
            "first_table_hit_rank": None,
            "first_row_hit_rank": None,
        }
    )
    assert stage == "retrieve_empty"
    assert "no retrieved entries" in detail


def test_classify_retrieval_not_in_pool() -> None:
    stage, _detail = _classify_retrieval(
        {
            "total_entries": 10,
            "first_file_hit_rank": None,
            "first_table_hit_rank": None,
            "first_row_hit_rank": None,
        }
    )
    assert stage == "retrieve_not_in_pool"


def test_classify_retrieval_row_miss() -> None:
    stage, _detail = _classify_retrieval(
        {
            "total_entries": 10,
            "first_file_hit_rank": 3,
            "first_table_hit_rank": None,
            "first_row_hit_rank": None,
        }
    )
    assert stage == "retrieve_row_miss"


def test_classify_retrieval_table_selection() -> None:
    stage, _detail = _classify_retrieval(
        {
            "total_entries": 10,
            "first_file_hit_rank": 3,
            "first_table_hit_rank": None,
            "first_row_hit_rank": 2,
        }
    )
    assert stage == "retrieve_table_selection"


def test_classify_retrieval_ok() -> None:
    stage, _detail = _classify_retrieval(
        {
            "total_entries": 10,
            "first_file_hit_rank": 3,
            "first_table_hit_rank": 4,
            "first_row_hit_rank": 2,
        }
    )
    assert stage == "retrieve_ok"
