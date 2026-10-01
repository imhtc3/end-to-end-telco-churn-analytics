import numpy as np

from churn.pipeline.transform import add_features, clean, to_snake, transform


def test_snake_case():
    assert to_snake("customerID") == "customer_id"
    assert to_snake("MonthlyCharges") == "monthly_charges"
    assert to_snake("SeniorCitizen") == "senior_citizen"
    assert to_snake("tenure") == "tenure"


def test_clean_types_and_imputation(raw_sample):
    df = clean(raw_sample)
    assert df["total_charges"].dtype == float
    assert df.loc[df["tenure"] == 0, "total_charges"].eq(0).all()
    assert set(df["churn"].unique()) <= {0, 1}
    assert df["churn"].sum() == 2


def test_service_counts(raw_sample):
    df = transform(raw_sample)
    everything = df.set_index("customer_id").loc["0005-E"]
    assert everything["n_addons"] == 6
    assert everything["n_services"] == 1 + 1 + 1 + 6
    phone_only = df.set_index("customer_id").loc["0004-D"]
    assert phone_only["n_addons"] == 0
    assert phone_only["has_internet"] == 0


def test_charge_drift_zero_for_new_customers(raw_sample):
    df = transform(raw_sample)
    assert df.loc[df["tenure"] == 0, "charge_drift"].eq(0).all()


def test_tenure_bands_cover_everything(raw_sample):
    df = transform(raw_sample)
    assert df["tenure_band"].notna().all()
    assert df.loc[df["tenure"] == 72, "tenure_band"].iloc[0] == "61m+"


def test_add_features_does_not_mutate_input(raw_sample):
    base = clean(raw_sample)
    before = base.copy()
    add_features(base)
    assert base.equals(before)


def test_auto_pay_flag(raw_sample):
    df = transform(raw_sample)
    assert np.array_equal(
        df["auto_pay"].to_numpy(),
        df["payment_method"].str.contains("automatic").astype(int).to_numpy(),
    )
