from oat_ext.config_merge import merge_with_overrides


def test_merge_with_overrides():
    base = {"a": 1, "b": {"c": 2}}
    m = merge_with_overrides(base, {"b": {"c": 3}})
    assert m.b.c == 3
    assert m.a == 1
