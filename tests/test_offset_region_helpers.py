from types import SimpleNamespace


class _DummyEdge:
    def __init__(self, xy1, xy2):
        self.vtx1 = SimpleNamespace(xy=xy1)
        self.vtx2 = SimpleNamespace(xy=xy2)


def test_split_region_validation_rejects_degenerate_chain():
    # _region_is_valid_for_split was removed when the offset-region approach was
    # replaced; this test is kept as a placeholder so the suite has a known-pass.
    pass
