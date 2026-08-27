import pandas as pd
import pytest

from visqai.validation import (
    require_dataframe,
    require_nonempty_dataframe,
    require_columns,
    require_path_exists,
    require_positive,
    require_positive_int,
    require_non_negative,
    require_in,
    require_type,
    require_nonempty,
)


# --------------------------------------------------------------------------
# require_dataframe / require_nonempty_dataframe / require_columns
# --------------------------------------------------------------------------

def test_require_dataframe_accepts_a_dataframe():
    require_dataframe(pd.DataFrame({"a": [1]}), "df")  # must not raise


def test_require_dataframe_rejects_non_dataframe():
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        require_dataframe([1, 2, 3], "df")


def test_require_dataframe_error_names_the_argument_and_actual_type():
    with pytest.raises(TypeError, match="my_arg.*dict"):
        require_dataframe({"a": 1}, "my_arg")


def test_require_nonempty_dataframe_accepts_nonempty():
    require_nonempty_dataframe(pd.DataFrame({"a": [1]}), "df")


def test_require_nonempty_dataframe_rejects_empty():
    with pytest.raises(ValueError, match="is empty"):
        require_nonempty_dataframe(pd.DataFrame(), "df")


def test_require_nonempty_dataframe_rejects_non_dataframe_first():
    with pytest.raises(TypeError):
        require_nonempty_dataframe(None, "df")


def test_require_columns_accepts_when_all_present():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    require_columns(df, ["a", "b"], "df")


def test_require_columns_rejects_when_missing():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match=r"missing required column\(s\)"):
        require_columns(df, ["a", "b", "c"], "df")


def test_require_columns_error_lists_only_the_missing_ones():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match=r"\['b', 'c'\]"):
        require_columns(df, ["a", "b", "c"], "df")


def test_require_columns_rejects_non_dataframe():
    with pytest.raises(TypeError):
        require_columns(None, ["a"], "df")


# --------------------------------------------------------------------------
# require_path_exists
# --------------------------------------------------------------------------

def test_require_path_exists_accepts_existing_path_and_returns_path_object(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hi")
    result = require_path_exists(f, "f")
    assert result == f


def test_require_path_exists_rejects_none():
    with pytest.raises(ValueError, match="must be given"):
        require_path_exists(None, "f")


def test_require_path_exists_rejects_empty_string():
    with pytest.raises(ValueError, match="must be given"):
        require_path_exists("", "f")


def test_require_path_exists_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        require_path_exists(tmp_path / "does_not_exist.txt", "f")


def test_require_path_exists_kind_file_accepts_file(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hi")
    require_path_exists(f, "f", kind="file")


def test_require_path_exists_kind_file_rejects_directory(tmp_path):
    with pytest.raises(ValueError, match="is not a file"):
        require_path_exists(tmp_path, "f", kind="file")


def test_require_path_exists_kind_dir_accepts_directory(tmp_path):
    require_path_exists(tmp_path, "d", kind="dir")


def test_require_path_exists_kind_dir_rejects_file(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hi")
    with pytest.raises(ValueError, match="is not a directory"):
        require_path_exists(f, "d", kind="dir")


# --------------------------------------------------------------------------
# require_positive / require_positive_int / require_non_negative
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value", [1, 1.5, 0.0001, 1e10])
def test_require_positive_accepts_positive_numbers(value):
    require_positive(value, "v")


@pytest.mark.parametrize("value", [0, -1, -0.5])
def test_require_positive_rejects_zero_and_negative(value):
    with pytest.raises(ValueError, match="positive"):
        require_positive(value, "v")


def test_require_positive_rejects_non_numeric():
    with pytest.raises(ValueError):
        require_positive("5", "v")


def test_require_positive_rejects_bool():
    """bool is a subclass of int in Python -- True/False must not silently
    pass a numeric check."""
    with pytest.raises(ValueError):
        require_positive(True, "v")


@pytest.mark.parametrize("value", [1, 5, 1000])
def test_require_positive_int_accepts_positive_ints(value):
    require_positive_int(value, "v")


def test_require_positive_int_rejects_float():
    with pytest.raises(ValueError):
        require_positive_int(1.5, "v")


def test_require_positive_int_rejects_zero():
    with pytest.raises(ValueError):
        require_positive_int(0, "v")


def test_require_positive_int_rejects_negative():
    with pytest.raises(ValueError):
        require_positive_int(-3, "v")


def test_require_positive_int_rejects_bool():
    with pytest.raises(ValueError):
        require_positive_int(False, "v")


@pytest.mark.parametrize("value", [0, 0.0, 1, 5.5])
def test_require_non_negative_accepts_zero_and_positive(value):
    require_non_negative(value, "v")


def test_require_non_negative_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        require_non_negative(-0.01, "v")


def test_require_non_negative_rejects_bool():
    with pytest.raises(ValueError):
        require_non_negative(True, "v")


# --------------------------------------------------------------------------
# require_in / require_type / require_nonempty
# --------------------------------------------------------------------------

def test_require_in_accepts_member_of_choices():
    require_in("linear", {"linear", "kernel"}, "corrector_mode")


def test_require_in_rejects_non_member():
    with pytest.raises(ValueError, match="corrector_mode"):
        require_in("quadratic", {"linear", "kernel"}, "corrector_mode")


def test_require_type_accepts_matching_type():
    require_type([1, 2], list, "num_cols")


def test_require_type_rejects_mismatched_type():
    with pytest.raises(TypeError, match="num_cols"):
        require_type("not a list", list, "num_cols")


def test_require_type_accepts_tuple_of_types():
    require_type(5, (int, float), "v")
    require_type(5.0, (int, float), "v")


def test_require_type_rejects_when_not_in_type_tuple():
    with pytest.raises(TypeError):
        require_type("5", (int, float), "v")


def test_require_nonempty_accepts_nonempty_list():
    require_nonempty([1], "samples")


def test_require_nonempty_rejects_empty_list():
    with pytest.raises(ValueError, match="samples"):
        require_nonempty([], "samples")


def test_require_nonempty_rejects_none():
    with pytest.raises(ValueError):
        require_nonempty(None, "samples")
