from thermopt.layout.geometry import hpwl, outline_violation, overlap_area
from thermopt.layout.objects import Chiplet, FloorplanCase, Layout, Net, Placement


def make_case() -> FloorplanCase:
    return FloorplanCase(
        chiplets=(Chiplet("A", 10, 10, 1), Chiplet("B", 10, 10, 1)),
        nets=(Net("N0", ("A", "B")),),
        outline_width=40,
        outline_height=30,
    )


def test_overlap_area() -> None:
    case = make_case()
    assert overlap_area(case, Placement("A", 0, 0), Placement("B", 5, 5)) == 25
    assert overlap_area(case, Placement("A", 0, 0), Placement("B", 20, 20)) == 0


def test_outline_violation() -> None:
    case = make_case()
    assert outline_violation(case, Placement("A", 0, 0)) == 0
    assert outline_violation(case, Placement("A", 35, 25)) == 50


def test_hpwl() -> None:
    case = make_case()
    layout = Layout((Placement("A", 0, 0), Placement("B", 10, 10)))
    assert hpwl(case, layout) == 20


def test_hpwl_uses_pin_offsets() -> None:
    case = FloorplanCase(
        chiplets=(Chiplet("A", 10, 10, 1), Chiplet("B", 10, 10, 1)),
        nets=(Net("N0", ("A", "B"), ((5, 0), (-5, 0))),),
        outline_width=40,
        outline_height=30,
    )
    layout = Layout((Placement("A", 0, 0), Placement("B", 10, 0)))
    assert hpwl(case, layout) == 0


def test_hpwl_rotates_pin_offsets_for_all_orientations() -> None:
    case = FloorplanCase(
        chiplets=(Chiplet("A", 10, 10, 1), Chiplet("B", 10, 10, 1)),
        nets=(Net("N0", ("A", "B"), ((2, 3), (0, 0))),),
        outline_width=40,
        outline_height=30,
    )

    assert hpwl(case, Layout((Placement("A", 0, 0, 0), Placement("B", 0, 0, 0)))) == 5
    assert hpwl(case, Layout((Placement("A", 0, 0, 90), Placement("B", 0, 0, 0)))) == 5
    assert hpwl(case, Layout((Placement("A", 0, 0, 180), Placement("B", 0, 0, 0)))) == 5
    assert hpwl(case, Layout((Placement("A", 0, 0, 270), Placement("B", 0, 0, 0)))) == 5
