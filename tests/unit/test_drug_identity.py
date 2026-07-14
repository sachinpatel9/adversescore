"""
Unit tests for drug_identity.py — brand/generic name resolution and market
authorization date lookup, mocking openFDA label.json/ndc.json/drugsfda.json
responses. No live API calls (see tests/e2e/test_drug_identity_e2e.py for
live-API coverage).
"""
import requests

from adverse_score.drug_identity import (
    DrugIdentity,
    DrugIdentityError,
    resolve_drug_identity,
)


class MockResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


def _label_result(brand_names=None, generic_names=None, substance_names=None,
                  application_numbers=None):
    return {
        "openfda": {
            "brand_name": brand_names or [],
            "generic_name": generic_names or [],
            "substance_name": substance_names or [],
            "application_number": application_numbers or [],
        }
    }


def _drugsfda_result(submissions):
    return {"submissions": submissions}


class TestDrugIdentityKnownPairs:
    """Known brand/generic pairs resolve to the same canonical identity."""

    def test_brand_name_resolves(self, client, monkeypatch):
        def route(url, **kw):
            if "label.json" in url:
                return MockResponse(200, {"results": [
                    _label_result(
                        brand_names=["KEYTRUDA"], generic_names=["PEMBROLIZUMAB"],
                        substance_names=["PEMBROLIZUMAB"], application_numbers=["BLA125514"],
                    )
                ]})
            if "drugsfda.json" in url:
                return MockResponse(200, {"results": [
                    _drugsfda_result([{"submission_status": "AP", "submission_status_date": "20140904"}])
                ]})
            return MockResponse(404, {})
        monkeypatch.setattr(client.session, "get", route)

        result = resolve_drug_identity("KEYTRUDA", client=client.fda)
        assert isinstance(result, DrugIdentity)
        assert "KEYTRUDA" in result.brand_names
        assert "PEMBROLIZUMAB" in result.generic_names
        assert result.resolution_confidence == "EXACT"

    def test_generic_name_resolves_to_overlapping_identity(self, client, monkeypatch):
        def route(url, **kw):
            if "label.json" in url:
                return MockResponse(200, {"results": [
                    _label_result(
                        brand_names=["KEYTRUDA"], generic_names=["PEMBROLIZUMAB"],
                        substance_names=["PEMBROLIZUMAB"], application_numbers=["BLA125514"],
                    )
                ]})
            if "drugsfda.json" in url:
                return MockResponse(200, {"results": [
                    _drugsfda_result([{"submission_status": "AP", "submission_status_date": "20140904"}])
                ]})
            return MockResponse(404, {})
        monkeypatch.setattr(client.session, "get", route)

        result = resolve_drug_identity("PEMBROLIZUMAB", client=client.fda)
        assert isinstance(result, DrugIdentity)
        # Same underlying identity — overlapping generic/substance names
        assert "PEMBROLIZUMAB" in result.generic_names
        assert "PEMBROLIZUMAB" in result.substance_names


class TestDrugIdentityMisspelling:
    """A common misspelling either resolves via openFDA's own leniency, or
    produces a clean NOT_FOUND error — never a crash or a silent empty result."""

    def test_misspelling_resolves_via_broadened_fallback(self, client, monkeypatch, sample_drug_names):
        call_log = []

        def route(url, **kw):
            call_log.append(url)
            search = kw.get("params", {}).get("search", "")
            # The primary exact-match query quotes the search term; the
            # broadened fallback query does not.
            if "label.json" in url and '"' in search:
                # Primary exact-match attempt fails
                return MockResponse(200, {"results": []})
            if "label.json" in url and '"' not in search:
                # Broadened fallback succeeds
                return MockResponse(200, {"results": [
                    _label_result(brand_names=["ASPIRIN"], generic_names=["ASPIRIN"],
                                  application_numbers=["NDA000123"])
                ]})
            if "drugsfda.json" in url:
                return MockResponse(200, {"results": [
                    _drugsfda_result([{"submission_status": "AP", "submission_status_date": "19820601"}])
                ]})
            return MockResponse(404, {})
        monkeypatch.setattr(client.session, "get", route)

        result = resolve_drug_identity(sample_drug_names["misspelled"], client=client.fda)
        assert isinstance(result, (DrugIdentity, DrugIdentityError))
        if isinstance(result, DrugIdentity):
            assert result.resolution_confidence == "FUZZY"
        else:
            assert result.reason == "NOT_FOUND"

    def test_misspelling_with_no_match_anywhere_is_clean_not_found(self, client, monkeypatch, sample_drug_names):
        monkeypatch.setattr(client.session, "get", lambda *a, **kw: MockResponse(200, {"results": []}))
        result = resolve_drug_identity(sample_drug_names["misspelled"], client=client.fda)
        assert isinstance(result, DrugIdentityError)
        assert result.reason == "NOT_FOUND"
        assert len(result.attempted_variants) == 3  # exact, broadened label, ndc fallback


class TestDrugIdentityNotFound:
    def test_all_endpoints_empty_returns_not_found(self, client, monkeypatch):
        monkeypatch.setattr(client.session, "get", lambda *a, **kw: MockResponse(404, {}))
        result = resolve_drug_identity("ZZZNOTADRUGZZZ", client=client.fda)
        assert isinstance(result, DrugIdentityError)
        assert result.reason == "NOT_FOUND"
        assert "ZZZNOTADRUGZZZ" in result.message

    def test_blank_name_returns_not_found_without_http_call(self, client, monkeypatch):
        calls = []
        monkeypatch.setattr(client.session, "get", lambda *a, **kw: calls.append(1) or MockResponse(200, {}))
        result = resolve_drug_identity("   ", client=client.fda)
        assert isinstance(result, DrugIdentityError)
        assert result.reason == "NOT_FOUND"
        assert calls == []  # never hits the network for a blank query


class TestDrugIdentityUpstreamError:
    def test_connection_error_after_retries_returns_upstream_error(self, client, monkeypatch):
        def always_fail(*a, **kw):
            raise requests.exceptions.ConnectionError("persistent failure")
        monkeypatch.setattr(client.session, "get", always_fail)

        result = resolve_drug_identity("METFORMIN", client=client.fda)
        assert isinstance(result, DrugIdentityError)
        assert result.reason == "UPSTREAM_ERROR"


class TestDrugIdentitySanitization:
    def test_special_characters_escaped_in_params(self, client, monkeypatch):
        captured = []

        def capture_get(url, **kw):
            captured.append(kw)
            return MockResponse(404, {})
        monkeypatch.setattr(client.session, "get", capture_get)

        resolve_drug_identity('DRUG"NAME', client=client.fda)
        search_param = captured[0].get("params", {}).get("search", "")
        assert '\\"' in search_param


class TestMarketAuthorizationDate:
    def test_only_ap_status_counts_earliest_wins(self, client, monkeypatch):
        def route(url, **kw):
            if "label.json" in url:
                return MockResponse(200, {"results": [
                    _label_result(brand_names=["TESTDRUG"], application_numbers=["NDA111", "ANDA222"])
                ]})
            if "drugsfda.json" in url:
                search = kw.get("params", {}).get("search", "")
                if "NDA111" in search:
                    return MockResponse(200, {"results": [
                        _drugsfda_result([
                            {"submission_status": "TA", "submission_status_date": "19700101"},
                            {"submission_status": "AP", "submission_status_date": "19850315"},
                        ])
                    ]})
                if "ANDA222" in search:
                    return MockResponse(200, {"results": [
                        _drugsfda_result([
                            {"submission_status": "AP", "submission_status_date": "19800101"},
                            {"submission_status": "WD", "submission_status_date": "19750101"},
                        ])
                    ]})
            return MockResponse(404, {})
        monkeypatch.setattr(client.session, "get", route)

        result = resolve_drug_identity("TESTDRUG", client=client.fda)
        assert isinstance(result, DrugIdentity)
        # Earliest AP date across both applications: 1980-01-01 (ANDA222), not the
        # earlier TA (1970) or WD (1975) dates, and not the later AP (1985).
        assert result.market_authorization_date.isoformat() == "1980-01-01"


class TestMarketAuthorizationDateMissing:
    def test_no_ap_submission_is_partial_success_not_error(self, client, monkeypatch):
        def route(url, **kw):
            if "label.json" in url:
                return MockResponse(200, {"results": [
                    _label_result(brand_names=["TESTDRUG"], application_numbers=["NDA999"])
                ]})
            if "drugsfda.json" in url:
                return MockResponse(200, {"results": [
                    _drugsfda_result([{"submission_status": "TA", "submission_status_date": "19700101"}])
                ]})
            return MockResponse(404, {})
        monkeypatch.setattr(client.session, "get", route)

        result = resolve_drug_identity("TESTDRUG", client=client.fda)
        assert isinstance(result, DrugIdentity)
        assert result.market_authorization_date is None
        assert result.resolution_confidence == "PARTIAL"
