from io import BytesIO

import pandas as pd
import pytest

from src.data.loader import DatasetLoadError, list_workbook_sheets, load_dataset


def test_load_csv_utf8_sig_preserves_vietnamese_and_columns():
    payload = "\ufefftitle,body_text\nTin tức tiếng Việt,Nội dung đầy đủ\n".encode("utf-8")

    frame = load_dataset(payload, filename="news.csv")

    assert list(frame.columns) == ["title", "body_text"]
    assert frame.loc[0, "title"] == "Tin tức tiếng Việt"
    assert frame.loc[0, "body_text"] == "Nội dung đầy đủ"


def test_load_jsonl_objects_and_reject_malformed_line():
    frame = load_dataset(
        b'{"id":"a","title":"B\\u00e0i 1"}\n{"id":"b","title":"B\\u00e0i 2"}\n',
        filename="news.jsonl",
    )
    assert frame["id"].tolist() == ["a", "b"]
    assert frame["title"].tolist() == ["Bài 1", "Bài 2"]

    with pytest.raises(DatasetLoadError, match="dòng 2"):
        load_dataset(b'{"id": 1}\nnot-json\n', filename="bad.jsonl")


def test_load_xlsx_and_list_multiple_sheets():
    workbook = BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame({"title": ["Một"]}).to_excel(writer, sheet_name="Dữ liệu A", index=False)
        pd.DataFrame({"title": ["Hai"]}).to_excel(writer, sheet_name="Dữ liệu B", index=False)

    raw = workbook.getvalue()
    assert list_workbook_sheets(raw, "news.xlsx") == ["Dữ liệu A", "Dữ liệu B"]
    frame = load_dataset(raw, filename="news.xlsx", sheet_name="Dữ liệu B")
    assert frame.loc[0, "title"] == "Hai"


def test_reject_unsupported_extension():
    with pytest.raises(DatasetLoadError, match="chưa được hỗ trợ"):
        load_dataset(b"x", filename="news.tsv")
