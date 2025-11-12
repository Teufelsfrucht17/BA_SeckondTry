import pandas as pd

from refinitiv_client import RefinitivClient


def test_to_dataframe_passthrough():
    df = pd.DataFrame({"a": [1, 2]})
    result = RefinitivClient._to_dataframe(df)
    assert result is df


def test_to_dataframe_handles_df_attribute():
    df = pd.DataFrame({"b": [3]})

    class Wrapper:
        def __init__(self, inner_df):
            self.df = inner_df

    wrapper = Wrapper(df)
    result = RefinitivClient._to_dataframe(wrapper)
    assert result is df


def test_to_dataframe_handles_nested_data_df():
    df = pd.DataFrame({"c": [4]})

    class Nested:
        def __init__(self, inner_df):
            self.data = type("Inner", (), {"df": inner_df})()

    nested = Nested(df)
    result = RefinitivClient._to_dataframe(nested)
    assert result is df


def test_to_dataframe_dict_list_coercion():
    payload = {"data": [{"ric": "ABC", "value": 1}]}
    result = RefinitivClient._to_dataframe(payload)
    assert list(result.columns) == ["ric", "value"]
    assert result.iloc[0]["ric"] == "ABC"


def test_standardize_history_columns_aliases():
    raw = pd.DataFrame(
        {
            "Instrument": ["RIC.A"],
            "Date": ["2024-01-01"],
            "OPEN_PRC": [10],
            "HIGH_1": [12],
            "LOW_1": [9],
            "TRDPRC_1": [11],
            "ACVOL_1": [1000],
        }
    )
    normalized = RefinitivClient._standardize_history_columns(raw)
    assert list(normalized.columns) == [
        "ric",
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert normalized.iloc[0]["ric"] == "RIC.A"


def test_standardize_history_columns_flattens_multiindex():
    tuples = [
        ("Instrument", "value"),
        ("Date", "value"),
        ("OPEN_PRC", "value"),
        ("HIGH_1", "value"),
        ("LOW_1", "value"),
        ("TRDPRC_1", "value"),
        ("ACVOL_1", "value"),
    ]
    columns = pd.MultiIndex.from_tuples(tuples)
    raw = pd.DataFrame([["RIC.A", "2024-01-01", 1, 2, 0, 1.5, 500]], columns=columns)
    normalized = RefinitivClient._standardize_history_columns(raw)
    assert list(normalized.columns) == [
        "ric",
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
