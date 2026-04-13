import pytest

from src.mapping.collapse import collapse_token, load_ptb_to_coarse, load_taxonomy


def test_rb_family_adv():
    r = collapse_token("RB")
    assert r.high_level == "ADV"
    assert "adv_type" in r.eligible_distinction_ids


def test_nn_noun():
    r = collapse_token("NN")
    assert r.high_level == "NOUN"
    assert "noun_proper_common" in r.eligible_distinction_ids


def test_punct_no_subclasses():
    r = collapse_token(".")
    assert r.high_level == "PUNCT"
    assert r.eligible_distinction_ids == []


def test_xx_unk():
    r = collapse_token("XX")
    assert r.high_level == "UNK"
    assert r.eligible_distinction_ids == []


def test_prp_dollar():
    r = collapse_token("PRP$")
    assert r.high_level == "PRON"


def test_verb_copular_labels_in_taxonomy():
    tax = load_taxonomy()
    ids = {d["id"]: d["labels"] for d in tax["distinctions"]}
    assert "copular" in ids["verb_copular_prog_pass"]
    assert "none_of_these" in ids["verb_copular_prog_pass"]


def test_ptb_map_has_wrb():
    m = load_ptb_to_coarse()
    assert m["WRB"] == "ADV"
