from types import SimpleNamespace


class _DummyEdge:
    def __init__(self, xy1, xy2):
        self.vtx1 = SimpleNamespace(xy=xy1)
        self.vtx2 = SimpleNamespace(xy=xy2)


def test_split_region_validation_rejects_degenerate_chain():
    from aabpl.radius_search.region_classes import _region_is_valid_for_split

    region = SimpleNamespace(
        edges=[_DummyEdge((0.0, 0.0), (1.0, 0.0)), _DummyEdge((1.0, 0.0), (1.0, 1.0))],
        vertices=[
            SimpleNamespace(xy=(0.0, 0.0)),
            SimpleNamespace(xy=(1.0, 0.0)),
            SimpleNamespace(xy=(1.0, 1.0)),
        ],
        is_closed=False,
    )

    assert _region_is_valid_for_split(region) is False
