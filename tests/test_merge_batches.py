from src.mining.merge_batches import route_splits


def test_route_splits_prefers_heldout_then_icl():
    labels = ["a", "b"]
    rows = [
        {"gold_subclass": "a"},
        {"gold_subclass": "a"},
        {"gold_subclass": "b"},
        {"gold_subclass": "b"},
    ]
    icl, hd = route_splits(rows, labels, icl_per_label=1, heldout_per_label=1)
    assert len(hd) == 2
    assert {r["gold_subclass"] for r in hd} == {"a", "b"}
    assert len(icl) == 2
    assert {r["gold_subclass"] for r in icl} == {"a", "b"}
