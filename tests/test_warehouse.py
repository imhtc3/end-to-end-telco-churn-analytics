from churn.pipeline.load import load, read_query
from churn.pipeline.transform import transform


def test_load_roundtrip(raw_sample, tmp_path):
    df = transform(raw_sample)
    db = tmp_path / "test.db"
    load(df, db_path=db)

    n = read_query("SELECT COUNT(*) AS n FROM vw_customer_360", db_path=db)["n"].iloc[0]
    assert n == len(df)

    mart = read_query("SELECT * FROM vw_feature_mart ORDER BY customer_id", db_path=db)
    assert list(mart["customer_id"]) == sorted(df["customer_id"])
    assert mart["tech_support"].notna().all()

    churned = read_query("SELECT SUM(churn) AS c FROM fact_subscription", db_path=db)["c"].iloc[0]
    assert churned == df["churn"].sum()


def test_bridge_is_six_rows_per_customer(raw_sample, tmp_path):
    df = transform(raw_sample)
    db = tmp_path / "test.db"
    load(df, db_path=db)
    per_customer = read_query(
        "SELECT customer_id, COUNT(*) AS n FROM bridge_customer_service GROUP BY customer_id", db_path=db
    )
    assert per_customer["n"].eq(6).all()
