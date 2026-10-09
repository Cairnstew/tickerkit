"""Symbol resolution and normalisation."""

from __future__ import annotations

import pytest

from tickerkit.normalise import (
    book_levels,
    depth_within,
    imbalance_from,
    spread_from,
    strip_html,
)
from tickerkit.resolve import normalise_query, pick_best_candidate


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("BTC", "BTC"),
        ("btc", "BTC"),
        ("  btc  ", "BTC"),
        ("BTC/USDT", "BTC"),
        ("btc-usd", "BTC"),
        ("ETH_USDT", "ETH"),
        ("SOL:USDT", "SOL"),
    ],
)
def test_normalise_query(raw: str, expected: str) -> None:
    assert normalise_query(raw) == expected


def test_pick_best_candidate_prefers_market_cap() -> None:
    winner, others = pick_best_candidate(
        [
            {"id": "small", "symbol": "x", "market_cap": 10.0},
            {"id": "big", "symbol": "x", "market_cap": 1_000_000.0},
        ]
    )
    assert winner is not None
    assert winner["id"] == "big"
    assert [o.coingecko_id for o in others] == ["small"]


def test_pick_best_candidate_does_not_treat_missing_cap_as_zero_wins() -> None:
    """A token with no market cap must sort below one that has a cap."""
    winner, _ = pick_best_candidate(
        [
            {"id": "no-cap", "symbol": "x", "market_cap": None},
            {"id": "has-cap", "symbol": "x", "market_cap": 5.0},
        ]
    )
    assert winner is not None
    assert winner["id"] == "has-cap"


def test_pick_best_candidate_handles_empty_and_malformed() -> None:
    assert pick_best_candidate([]) == (None, [])
    assert pick_best_candidate([{"symbol": "x"}]) == (None, [])  # no id -> unusable


def test_pick_best_candidate_falls_back_when_no_cap_at_all() -> None:
    winner, others = pick_best_candidate(
        [{"id": "a", "symbol": "x"}, {"id": "b", "symbol": "x"}]
    )
    assert winner is not None and winner["id"] in {"a", "b"}
    assert len(others) == 1


class TestBookLevels:
    def test_sorts_and_drops_invalid_entries(self) -> None:
        raw = [
            [100.0, 1.0],
            [102.0, 2.0],
            [101.0, 0.0],
            [float("nan"), 1.0],
            [99.0, None],
            ["bad"],
            [98.0],
        ]
        bids = book_levels(raw, ascending=False)
        assert [lvl.price for lvl in bids] == [102.0, 100.0]
        assert [lvl.amount for lvl in bids] == [2.0, 1.0]

    def test_asks_ascend(self) -> None:
        asks = book_levels([[101.0, 1.0], [100.0, 1.0]], ascending=True)
        assert [lvl.price for lvl in asks] == [100.0, 101.0]

    def test_non_sequence_is_empty_not_an_error(self) -> None:
        assert book_levels(None, ascending=False) == ()
        assert book_levels("nonsense", ascending=True) == ()


class TestSpread:
    def test_normal_book(self) -> None:
        spread, bps = spread_from(100.0, 100.2)
        assert spread == pytest.approx(0.2)
        assert bps == pytest.approx(19.98, rel=1e-3)

    def test_crossed_book_yields_no_numbers(self) -> None:
        """A locked/crossed book would make the derived figures meaningless."""
        assert spread_from(100.2, 100.0) == (None, None)

    @pytest.mark.parametrize(("bid", "ask"), [(None, 1.0), (1.0, None), (0.0, 1.0)])
    def test_missing_sides(self, bid: float | None, ask: float | None) -> None:
        assert spread_from(bid, ask) == (None, None)


class TestDepth:
    def test_bands_sum_only_levels_inside_the_band(self) -> None:
        bids = book_levels([[99.9, 1.0], [99.0, 2.0], [98.0, 4.0]], ascending=False)
        asks = book_levels([[100.1, 3.0], [101.0, 5.0]], ascending=True)
        bands = depth_within(bids, asks, (0.005, 0.02), 100.0)

        near = bands[0]
        assert near.pct == 0.005
        assert near.bid_base == 1.0  # only 99.9 is within 0.5% of 100
        assert near.ask_base == 3.0

        wide = bands[1]
        assert wide.bid_base == 7.0  # 99.9 + 99.0
        assert wide.ask_base == 8.0  # 100.1 + 101.0

    def test_no_mid_means_no_bands(self) -> None:
        assert depth_within([], [], (0.01,), None) == ()


def test_imbalance_edges() -> None:
    assert imbalance_from(10.0, 10.0) == pytest.approx(0.0)
    assert imbalance_from(10.0, 0.0) == pytest.approx(1.0)
    assert imbalance_from(0.0, 10.0) == pytest.approx(-1.0)
    assert imbalance_from(0.0, 0.0) is None
    assert imbalance_from(None, 1.0) is None


class TestStripHtml:
    def test_strips_inline_and_block_tags(self) -> None:
        raw = "<p>Bitcoin is a <b>decentralised</b> currency.</p><p>No <i>central</i> bank.</p>"
        out = strip_html(raw)
        assert out is not None
        assert "<" not in out and ">" not in out
        assert out == "Bitcoin is a decentralised currency. No central bank."

    def test_decodes_entities(self) -> None:
        assert strip_html("<p>A &amp; B &lt;3</p>") == "A & B <3"

    def test_block_tags_do_not_merge_words(self) -> None:
        assert strip_html("<p>one</p><p>two</p>") == "one two"

    @pytest.mark.parametrize("raw", [None, "", "   ", 42])
    def test_non_strings_yield_none(self, raw: object) -> None:
        assert strip_html(raw) is None

    def test_truncates(self) -> None:
        assert strip_html("<p>abcdefghij</p>", limit=4) == "abcd"