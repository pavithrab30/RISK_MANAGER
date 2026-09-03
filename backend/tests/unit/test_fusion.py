from app.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_favors_items_ranked_high_in_multiple_lists():
    dense = ["a", "b", "c", "d"]
    keyword = ["b", "a", "e", "f"]
    scores = reciprocal_rank_fusion([dense, keyword])

    # "a" and "b" both appear near the top of both lists, so they should
    # outscore anything that only appears in one list.
    assert scores["a"] > scores["c"]
    assert scores["b"] > scores["e"]
    assert scores["a"] > 0
    assert scores["b"] > 0


def test_rrf_item_absent_from_a_list_still_scores_from_the_other():
    scores = reciprocal_rank_fusion([["a", "b"], ["c"]])
    assert set(scores.keys()) == {"a", "b", "c"}
    assert scores["a"] > scores["b"]  # a ranked higher in the list it's in


def test_rrf_empty_rankings_returns_empty():
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[], []]) == {}


def test_rrf_is_score_only_not_rank_position_dependent_on_list_order():
    # fusing [A,B] then [B,A] should give the same combined result as
    # [B,A] then [A,B] - order of the *lists* shouldn't matter, only each
    # item's rank within its own list.
    s1 = reciprocal_rank_fusion([["x", "y"], ["y", "x"]])
    s2 = reciprocal_rank_fusion([["y", "x"], ["x", "y"]])
    assert s1 == s2
