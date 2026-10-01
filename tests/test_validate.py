import pytest

from churn.pipeline.validate import DataContractError, validate_raw


def test_clean_sample_passes(raw_sample):
    rep = validate_raw(raw_sample)
    assert rep.ok
    # the blank TotalCharges on the tenure-0 row is expected and only warned about
    assert any("TotalCharges" in w for w in rep.warnings)


def test_missing_column_is_fatal(raw_sample):
    with pytest.raises(DataContractError):
        validate_raw(raw_sample.drop(columns="Contract"))


def test_unknown_category_is_caught(raw_sample):
    bad = raw_sample.copy()
    bad.loc[0, "Contract"] = "Three year"
    rep = validate_raw(bad, strict=False)
    assert not rep.ok
    assert any("Contract" in e for e in rep.errors)


def test_duplicate_ids(raw_sample):
    bad = raw_sample.copy()
    bad.loc[1, "customerID"] = bad.loc[0, "customerID"]
    assert not validate_raw(bad, strict=False).ok


def test_blank_total_charges_with_tenure_is_an_error(raw_sample):
    bad = raw_sample.copy()
    bad.loc[1, "TotalCharges"] = " "
    rep = validate_raw(bad, strict=False)
    assert any("TotalCharges" in e for e in rep.errors)
