"""
Unit tests for fda_client.py — query building, HTTP fetch/retry, result
flattening, label/peer/class discovery, quarter boundaries, sanitization
against Lucene injection / special characters, and PSUR chunked/paginated
retrieval (Phase 2).
"""

import re
import pytest
import requests
from datetime import date

from adverse_score.fda_client import (
    ChunkResult,
    compute_psur_period,
    _compute_psur_chunks,
    _merge_chunks,
)


# ── Query Building ────────────────────────────────────────────────────────


class TestQueryBuilder:
    """Tests for build_query() and _sanitize_for_query()."""

    def test_build_query_basic(self, client):
        """Query string contains the drug name in a Lucene quoted field, a date range, and limit parameter."""
        query = client.build_query("KEYTRUDA")
        assert 'patient.drug.medicinalproduct:"KEYTRUDA"' in query
        assert "receivedate:" in query
        assert "limit=500" in query

    def test_build_query_with_demographics(self, client):
        """When patient_sex and patient_age are provided, the query includes sex code and age cohort bracket."""
        query = client.build_query("KEYTRUDA", patient_age=50, patient_sex="F")
        assert "patient.patientsex:2" in query
        assert "patient.patientonsetage:" in query
        assert "[45+TO+55]" in query

    def test_build_query_sex_code_mapping(self, client):
        """Verifies F maps to sex code '2' and M maps to sex code '1' (the corrected openFDA mapping)."""
        female_query = client.build_query("X", patient_sex="F")
        male_query = client.build_query("X", patient_sex="M")
        assert "patientsex:2" in female_query
        assert "patientsex:1" in male_query

    def test_sanitize_for_query_escapes_quotes(self, client):
        """Double quotes in input are escaped to backslash-quote for Lucene safety."""
        result = client._sanitize_for_query('DRUG"NAME')
        assert result == 'DRUG\\"NAME'

    def test_sanitize_for_query_escapes_backslash(self, client):
        """Backslashes in input are escaped to double-backslash for Lucene safety."""
        result = client._sanitize_for_query("DRUG\\NAME")
        assert result == "DRUG\\\\NAME"

    def test_sanitize_for_query_clean_input(self, client):
        """Normal drug names like 'KEYTRUDA' pass through unchanged."""
        assert client._sanitize_for_query("KEYTRUDA") == "KEYTRUDA"


class TestBuildQueryDateRange:
    """Tests for build_query() with explicit start_date/end_date parameters."""

    def test_explicit_date_range(self, client):
        """When start_date and end_date are provided, they appear in the query instead of days_back."""
        query = client.build_query("ASPIRIN", start_date="20250401", end_date="20250630")
        assert "20250401" in query
        assert "20250630" in query

    def test_days_back_still_works(self, client):
        """Without start_date/end_date, the query still uses days_back for date range."""
        query = client.build_query("ASPIRIN", days_back=30)
        assert "receivedate:" in query


# ── HTTP Fetch + Retry ────────────────────────────────────────────────────


class TestFetchEvents:
    """Tests for fetch_events() — the primary FDA API call."""

    def test_fetch_events_returns_data(self, client, mock_fda_response, monkeypatch):
        """A successful 200 response returns the parsed JSON dict with 'results' key."""
        class MockResponse:
            status_code = 200
            def json(self):
                return mock_fda_response
            def raise_for_status(self):
                pass
        monkeypatch.setattr(client.session, "get", lambda *a, **kw: MockResponse())
        result = client.fetch_events("KEYTRUDA")
        assert result is not None
        assert "results" in result
        assert len(result["results"]) == 4

    def test_fetch_events_404_returns_none(self, client, monkeypatch):
        """A 404 response (zero results) returns None instead of raising an exception."""
        class MockResponse:
            status_code = 404
        monkeypatch.setattr(client.session, "get", lambda *a, **kw: MockResponse())
        result = client.fetch_events("NONEXISTENTDRUG")
        assert result is None

    def test_fetch_events_http_error_returns_none(self, client, monkeypatch):
        """HTTP exceptions return None, not an unhandled exception."""
        import requests
        def raise_error(*a, **kw):
            raise requests.exceptions.ConnectionError("server down")
        monkeypatch.setattr(client.session, "get", raise_error)
        result = client.fetch_events("KEYTRUDA")
        assert result is None

    def test_fetch_events_retries_on_connection_error(self, client, monkeypatch):
        """tenacity retries transient ConnectionError before propagating failure."""
        import requests
        call_count = {"n": 0}
        def flaky_get(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise requests.exceptions.ConnectionError("transient")
            class OKResponse:
                status_code = 200
                def json(self): return {"results": []}
                def raise_for_status(self): pass
            return OKResponse()
        monkeypatch.setattr(client.session, "get", flaky_get)
        result = client.fetch_events("TESTDRUG")
        assert result is not None
        assert call_count["n"] == 3  # 2 failures + 1 success

    def test_fetch_events_exhausts_retries(self, client, monkeypatch):
        """After 3 failed attempts, tenacity stops and the method returns None."""
        import requests
        call_count = {"n": 0}
        def always_fail(*a, **kw):
            call_count["n"] += 1
            raise requests.exceptions.ConnectionError("persistent")
        monkeypatch.setattr(client.session, "get", always_fail)
        result = client.fetch_events("TESTDRUG")
        assert result is None
        assert call_count["n"] == 3


# ── Result Flattening ─────────────────────────────────────────────────────


class TestFlattenResults:
    """Tests for _flatten_results() — FDA JSON → flat dicts."""

    def test_flatten_results_extracts_fields(self, client, mock_fda_response):
        """Each flattened report contains report_id, date, severity, is_death, is_hospitalization, symptoms."""
        flat = client._flatten_results(mock_fda_response)
        assert len(flat) == 4
        required_keys = {"report_id", "date", "severity", "is_death", "is_hospitalization", "symptoms", "company"}
        for report in flat:
            assert required_keys.issubset(report.keys())
        # Verify severity mapping
        assert flat[0]["is_death"] is True
        assert flat[1]["is_hospitalization"] is True
        assert flat[0]["severity"] == "Serious"
        assert flat[3]["severity"] == "Non-Serious"
        # Verify symptoms joined
        assert "CARDIAC ARREST" in flat[0]["symptoms"]
        assert "DEATH" in flat[0]["symptoms"]

    def test_flatten_results_none_input(self, client):
        """None input (from a failed fetch) returns an empty list."""
        assert client._flatten_results(None) == []

    def test_flatten_results_empty_results(self, client):
        """A response with 'results': [] returns an empty list."""
        assert client._flatten_results({"results": []}) == []


# ── Label Text + Class/Peer Discovery ─────────────────────────────────────


class TestLabelAndDiscovery:
    """Tests for fetch_label_text(), _discover_drug_class(), and _discover_peers()."""

    def test_fetch_label_text_success(self, client, mock_label_response, monkeypatch):
        """Returns lowercase joined adverse_reactions text from a valid label response."""
        class MockResponse:
            status_code = 200
            def json(self):
                return mock_label_response
            def raise_for_status(self):
                pass
        monkeypatch.setattr(client.session, "get", lambda *a, **kw: MockResponse())
        result = client.fetch_label_text("KEYTRUDA")
        assert "nausea" in result
        assert "hepatotoxicity" in result
        assert result == result.lower()  # must be lowercase

    def test_fetch_label_text_failure_returns_empty(self, client, monkeypatch):
        """API failure (timeout, 500, etc.) returns empty string as the unlabeled fallback."""
        def raise_error(*a, **kw):
            raise Exception("timeout")
        monkeypatch.setattr(client.session, "get", raise_error)
        assert client.fetch_label_text("KEYTRUDA") == ""

    def test_discover_drug_class_sanitizes_input(self, client, monkeypatch):
        """Drug name is passed through _sanitize_for_query before embedding in the Lucene search string.
        Uses params= dict so the escaped value is in kwargs['params']['search'], not the raw URL."""
        captured_kwargs = []
        class MockResponse:
            status_code = 200
            def json(self):
                return {"results": [{"term": "Test Class [EPC]", "count": 100}]}
            def raise_for_status(self):
                pass
        def capture_get(url, **kw):
            captured_kwargs.append(kw)
            return MockResponse()
        monkeypatch.setattr(client.session, "get", capture_get)
        client._discover_drug_class('DRUG"NAME')
        # The escaped quote must appear in the search param value
        search_param = captured_kwargs[0].get("params", {}).get("search", "")
        assert '\\"' in search_param

    def test_discover_peers_excludes_target_drug(self, client, monkeypatch):
        """The target drug itself is filtered out of the peer list."""
        class MockResponse:
            status_code = 200
            def json(self):
                return {"results": [
                    {"term": "KEYTRUDA", "count": 500},
                    {"term": "OPDIVO", "count": 400},
                    {"term": "YERVOY", "count": 300},
                    {"term": "TECENTRIQ", "count": 200},
                ]}
            def raise_for_status(self):
                pass
        monkeypatch.setattr(client.session, "get", lambda *a, **kw: MockResponse())
        peers = client._discover_peers("Programmed Death Receptor [EPC]", "KEYTRUDA")
        assert "KEYTRUDA" not in peers
        assert len(peers) == 3
        assert "OPDIVO" in peers

    def test_discover_peers_respects_min_name_length(self, client, monkeypatch):
        """Peer names with 3 or fewer characters are excluded as likely abbreviations."""
        class MockResponse:
            status_code = 200
            def json(self):
                return {"results": [
                    {"term": "AB", "count": 500},
                    {"term": "XYZ", "count": 400},
                    {"term": "OPDIVO", "count": 300},
                    {"term": "YERVOY", "count": 200},
                ]}
            def raise_for_status(self):
                pass
        monkeypatch.setattr(client.session, "get", lambda *a, **kw: MockResponse())
        peers = client._discover_peers("Test Class [EPC]", "KEYTRUDA")
        assert "AB" not in peers
        assert "XYZ" not in peers
        assert "OPDIVO" in peers

    def test_fetch_symptom_counts_handles_malformed_items(self, client, monkeypatch):
        """Result items missing 'term' or 'count' keys are skipped instead of raising KeyError."""
        class MockResponse:
            status_code = 200
            def json(self):
                return {"results": [
                    {"term": "NAUSEA", "count": 50},
                    {"count": 30},           # missing term
                    {"term": "HEADACHE"},     # missing count
                    {"term": "FATIGUE", "count": 20},
                ]}
            def raise_for_status(self):
                pass
        monkeypatch.setattr(client.session, "get", lambda *a, **kw: MockResponse())
        counts = client._fetch_symptom_counts(drug_name="KEYTRUDA")
        assert counts == {"NAUSEA": 50, "FATIGUE": 20}
        assert "HEADACHE" not in counts


# ── Quarter Boundaries ────────────────────────────────────────────────────


class TestComputeQuarterBoundaries:
    """Tests for _compute_quarter_boundaries() method."""

    def test_returns_requested_count(self, client):
        result = client._compute_quarter_boundaries(4)
        assert len(result) == 4

    def test_quarter_label_format(self, client):
        result = client._compute_quarter_boundaries(4)
        for label, start, end in result:
            assert re.match(r"\d{4}-Q[1-4]", label), f"Invalid label: {label}"

    def test_date_format(self, client):
        result = client._compute_quarter_boundaries(4)
        for label, start, end in result:
            assert len(start) == 8 and start.isdigit(), f"Invalid start date: {start}"
            assert len(end) == 8 and end.isdigit(), f"Invalid end date: {end}"

    def test_chronological_order(self, client):
        result = client._compute_quarter_boundaries(4)
        for i in range(1, len(result)):
            assert result[i][1] > result[i-1][2], "Quarters not in chronological order"


# ── Special Character / Adversarial Input Handling ────────────────────────
# Only sub-tests exercising surviving fda_client.py surface (build_query,
# _sanitize_for_query) are kept. Sub-tests that instantiated the now-removed
# ClinicalQuerySchema (agent_tools.py) were dropped.


class TestSpecialCharacterHandling:
    """Tests that drug names with special characters are handled safely by build_query."""

    def test_drug_name_with_quotes(self, client):
        """A drug name containing '"' is properly escaped to '\\"' in the Lucene query string."""
        query = client.build_query('DRUG"NAME')
        # The quote must be escaped so the Lucene quoted field isn't broken
        assert 'DRUG\\"NAME' in query

    def test_drug_name_with_backslash(self, client):
        """A drug name containing '\\' is properly escaped to '\\\\' in the Lucene query string."""
        query = client.build_query("DRUG\\NAME")
        assert "DRUG\\\\NAME" in query

    def test_build_query_lucene_injection(self, client):
        """Adversarial input like '" OR *:*' is escaped and does not break the Lucene query structure."""
        query = client.build_query('" OR *:*')
        # The quote should be escaped, preventing injection
        assert '\\"' in query
        # The query should still have exactly one opening and closing quote around the drug name
        search_part = query.split("search=")[1].split("&")[0]
        assert 'medicinalproduct:"' in query


# ── PSUR Period Anchoring (Phase 2) ────────────────────────────────────────


class TestComputePsurPeriod:
    """Tests for compute_psur_period() — IBD-cycle anchoring to market_authorization_date."""

    def test_worked_example_keytruda_1yr(self):
        """Approved 2014-09-04, period=1yr, as_of=2026-07-13 -> most recently completed
        cycle is 2024-09-04 to 2025-09-04 (the 2025-09-04..2026-09-04 cycle hasn't ended yet)."""
        start, end, used_fallback, reason = compute_psur_period(
            date(2014, 9, 4), "1yr", as_of=date(2026, 7, 13))
        assert start == date(2024, 9, 4)
        assert end == date(2025, 9, 4)
        assert used_fallback is False
        assert reason is None

    @pytest.mark.parametrize("period,months", [("6mo", 6), ("1yr", 12), ("2yr", 24), ("3yr", 36)])
    def test_cycle_length_per_period(self, period, months):
        start, end, used_fallback, reason = compute_psur_period(
            date(2014, 9, 4), period, as_of=date(2026, 7, 13))
        assert used_fallback is False
        # end is exactly `months` months after start
        expected_end_year = start.year + (start.month - 1 + months) // 12
        expected_end_month = (start.month - 1 + months) % 12 + 1
        assert end.year == expected_end_year
        assert end.month == expected_end_month
        assert end.day == start.day

    def test_none_anchor_falls_back(self):
        start, end, used_fallback, reason = compute_psur_period(
            None, "1yr", as_of=date(2026, 7, 13))
        assert used_fallback is True
        assert reason == "NO_MARKET_AUTH_DATE"
        assert end == date(2026, 7, 13)
        assert (end - start).days == 365

    def test_anchor_less_than_one_period_ago_falls_back(self):
        """Approved 2 months before as_of, period=1yr -> no completed cycle exists yet."""
        start, end, used_fallback, reason = compute_psur_period(
            date(2026, 5, 13), "1yr", as_of=date(2026, 7, 13))
        assert used_fallback is True
        assert reason == "AUTH_DATE_LESS_THAN_ONE_PERIOD_AGO"
        assert end == date(2026, 7, 13)

    def test_day_of_month_clamping_does_not_raise(self):
        """Anchor on the 31st of a month must not raise when a target month has fewer days."""
        start, end, used_fallback, reason = compute_psur_period(
            date(2020, 1, 31), "1yr", as_of=date(2026, 7, 13))
        assert used_fallback is False
        assert start.day in (28, 29, 30, 31)
        assert end.day in (28, 29, 30, 31)


# ── PSUR Chunk Boundaries (Phase 2) ────────────────────────────────────────


class TestComputePsurChunks:
    """Tests for _compute_psur_chunks() — quarterly sub-period slicing, IBD-anchored."""

    @pytest.mark.parametrize("period,expected_chunks", [
        ("6mo", 2), ("1yr", 4), ("2yr", 8), ("3yr", 12),
    ])
    def test_chunk_count_per_period(self, period, expected_chunks):
        start, end, _, _ = compute_psur_period(date(2014, 9, 4), period, as_of=date(2026, 7, 13))
        chunks = _compute_psur_chunks(start, end)
        assert len(chunks) == expected_chunks

    def test_first_and_last_chunk_match_period_bounds(self):
        start, end, _, _ = compute_psur_period(date(2014, 9, 4), "1yr", as_of=date(2026, 7, 13))
        chunks = _compute_psur_chunks(start, end)
        assert chunks[0][1] == start.strftime("%Y%m%d")
        assert chunks[-1][2] == end.strftime("%Y%m%d")

    def test_chunks_are_chronologically_contiguous(self):
        start, end, _, _ = compute_psur_period(date(2014, 9, 4), "1yr", as_of=date(2026, 7, 13))
        chunks = _compute_psur_chunks(start, end)
        for i in range(1, len(chunks)):
            assert chunks[i][1] == chunks[i - 1][2], "Chunks are not contiguous"

    def test_label_format(self):
        start, end, _, _ = compute_psur_period(date(2014, 9, 4), "1yr", as_of=date(2026, 7, 13))
        chunks = _compute_psur_chunks(start, end)
        for label, s, e in chunks:
            assert re.match(r"chunk-\d+-\d{8}-to-\d{8}", label), f"Invalid label: {label}"


# ── PSUR Chunk Pagination (Phase 2) ────────────────────────────────────────


class MockPageResponse:
    def __init__(self, status_code=200, total=0, results=None):
        self.status_code = status_code
        self._total = total
        self._results = results or []

    def raise_for_status(self):
        pass

    def json(self):
        return {"meta": {"results": {"total": self._total}}, "results": self._results}


def _fake_report(report_id):
    return {"safetyreportid": report_id, "receivedate": "20250101",
            "seriousness": "1", "seriousnessdeath": None,
            "seriousnesshospitalization": None,
            "patient": {"reaction": [{"reactionmeddrapt": "NAUSEA"}]},
            "companynumb": "PHARMA-001"}


class TestFetchChunkPaginated:
    """Tests for _fetch_chunk_paginated() — skip/limit pagination, ceiling, and failure handling."""

    def test_exhausted_before_ceiling(self, client, monkeypatch):
        from adverse_score.config import PSUR_PAGE_SIZE
        page1 = [_fake_report(f"R{i}") for i in range(PSUR_PAGE_SIZE)]
        page2 = [_fake_report(f"R{i}") for i in range(PSUR_PAGE_SIZE, PSUR_PAGE_SIZE + 10)]
        calls = []

        def route(url, **kw):
            calls.append(kw["params"]["skip"])
            if kw["params"]["skip"] == 0:
                return MockPageResponse(200, total=len(page1) + len(page2), results=page1)
            return MockPageResponse(200, total=len(page1) + len(page2), results=page2)
        monkeypatch.setattr(client.session, "get", route)

        result = client.fda._fetch_chunk_paginated(["KEYTRUDA"], "chunk-1", "20250101", "20250401")
        assert len(calls) == 2
        assert result.truncated is False
        assert result.error is None
        assert result.retrieved_count == len(page1) + len(page2)

    def test_ceiling_hit_flags_truncated(self, client, monkeypatch):
        from adverse_score.config import PSUR_PAGE_SIZE, PSUR_SKIP_CEILING
        full_page = [_fake_report(f"R{i}") for i in range(PSUR_PAGE_SIZE)]

        def route(url, **kw):
            # Always return a full page — simulates a dataset larger than the ceiling
            return MockPageResponse(200, total=999999, results=full_page)
        monkeypatch.setattr(client.session, "get", route)

        result = client.fda._fetch_chunk_paginated(["KEYTRUDA"], "chunk-1", "20250101", "20250401")
        assert result.truncated is True
        assert result.error is None
        # Loop must terminate (not run forever) once skip exceeds the ceiling
        max_expected_calls = (PSUR_SKIP_CEILING // PSUR_PAGE_SIZE) + 1
        assert result.retrieved_count <= max_expected_calls * PSUR_PAGE_SIZE

    def test_mid_pagination_failure_aborts_only_this_chunk(self, client, monkeypatch):
        from adverse_score.config import PSUR_PAGE_SIZE
        page1 = [_fake_report(f"R{i}") for i in range(PSUR_PAGE_SIZE)]
        calls = {"n": 0}

        def route(url, **kw):
            calls["n"] += 1
            if kw["params"]["skip"] == 0:
                return MockPageResponse(200, total=PSUR_PAGE_SIZE * 2, results=page1)
            raise requests.exceptions.ConnectionError("simulated mid-pagination failure")
        monkeypatch.setattr(client.session, "get", route)

        result = client.fda._fetch_chunk_paginated(["KEYTRUDA"], "chunk-1", "20250101", "20250401")
        assert result.truncated is True
        assert result.error is not None
        assert result.retrieved_count == len(page1)

    def test_404_first_page_is_empty_not_error(self, client, monkeypatch):
        monkeypatch.setattr(client.session, "get", lambda *a, **kw: MockPageResponse(404))
        result = client.fda._fetch_chunk_paginated(["ZZZNOTADRUG"], "chunk-1", "20250101", "20250401")
        assert result.retrieved_count == 0
        assert result.truncated is False
        assert result.error is None

    def test_retries_per_page_via_tenacity(self, client, monkeypatch):
        """Existing tenacity retry behavior still fires per-page, not once globally."""
        call_count = {"n": 0}

        def flaky_get(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise requests.exceptions.ConnectionError("transient")
            return MockPageResponse(200, total=0, results=[])
        monkeypatch.setattr(client.session, "get", flaky_get)

        result = client.fda._fetch_chunk_paginated(["KEYTRUDA"], "chunk-1", "20250101", "20250401")
        assert call_count["n"] == 3  # 2 failures + 1 success, per tenacity's retry-per-call
        assert result.retrieved_count == 0
        assert result.error is None


# ── PSUR Chunk Merge/Dedup (Phase 2) ───────────────────────────────────────


def _flat_report(report_id):
    """A pre-flattened report dict — the shape ChunkResult.reports actually holds
    (post _flatten_results), distinct from _fake_report's raw FDA JSON shape used
    to mock HTTP page responses in TestFetchChunkPaginated above."""
    return {"report_id": report_id, "date": "20250101", "severity": "Serious",
            "is_death": False, "is_hospitalization": False,
            "symptoms": "NAUSEA", "company": "PHARMA-001"}


class TestMergeChunks:
    """Tests for _merge_chunks() — boundary-collision list hygiene, not Phase 3's dedup."""

    def test_boundary_duplicate_report_id_kept_once(self):
        shared = _flat_report("SHARED-ID")
        chunk1 = ChunkResult(label="chunk-1", start_date="20240101", end_date="20240401",
                             reports=[shared, _flat_report("UNIQUE-1")],
                             retrieved_count=2, estimated_total_count=2, truncated=False)
        chunk2 = ChunkResult(label="chunk-2", start_date="20240401", end_date="20240701",
                             reports=[shared, _flat_report("UNIQUE-2")],
                             retrieved_count=2, estimated_total_count=2, truncated=False)

        merged = _merge_chunks([chunk1, chunk2])
        ids = [r["report_id"] for r in merged]
        assert len(merged) == 3
        assert ids.count("SHARED-ID") == 1
        assert "UNIQUE-1" in ids and "UNIQUE-2" in ids

    def test_empty_chunk_list_returns_empty(self):
        assert _merge_chunks([]) == []

    def test_chunk_order_preserved(self):
        chunk1 = ChunkResult(label="chunk-1", start_date="20240101", end_date="20240401",
                             reports=[_flat_report("A")],
                             retrieved_count=1, estimated_total_count=1, truncated=False)
        chunk2 = ChunkResult(label="chunk-2", start_date="20240401", end_date="20240701",
                             reports=[_flat_report("B")],
                             retrieved_count=1, estimated_total_count=1, truncated=False)
        merged = _merge_chunks([chunk1, chunk2])
        assert [r["report_id"] for r in merged] == ["A", "B"]
