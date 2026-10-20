"""Load supported tabular datasets without changing their source columns."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".jsonl"}


class DatasetLoadError(ValueError):
    """Raised when an input file cannot be read as a supported dataset."""


def _filename(source: Any, filename: str | None) -> str:
    if filename:
        return filename
    if isinstance(source, (str, Path)):
        return Path(source).name
    return str(getattr(source, "name", ""))


def _read_bytes(source: Any) -> bytes | str | Path:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise DatasetLoadError(f"Không tìm thấy tệp: {path}")
        try:
            return path.read_bytes()
        except OSError as exc:
            raise DatasetLoadError(f"Không đọc được tệp: {exc}") from exc

    if isinstance(source, bytes):
        return source
    if isinstance(source, bytearray):
        return bytes(source)
    if hasattr(source, "getvalue"):
        return bytes(source.getvalue())
    if hasattr(source, "read"):
        try:
            content = source.read()
            if hasattr(source, "seek"):
                source.seek(0)
            if isinstance(content, str):
                return content
            return bytes(content)
        except (OSError, TypeError, ValueError) as exc:
            raise DatasetLoadError(f"Không đọc được nội dung tệp: {exc}") from exc
    raise DatasetLoadError("Nguồn dữ liệu phải là đường dẫn, bytes hoặc tệp đã tải lên.")


def list_workbook_sheets(source: Any, filename: str | None = None) -> list[str]:
    """Return sheet names for an XLSX input."""
    name = _filename(source, filename)
    if Path(name).suffix.lower() != ".xlsx":
        raise DatasetLoadError("Chỉ có thể liệt kê sheet của tệp XLSX.")
    content = _read_bytes(source)
    try:
        excel_source = content if isinstance(content, Path) else BytesIO(
            content.encode("utf-8") if isinstance(content, str) else content
        )
        with pd.ExcelFile(excel_source) as workbook:
            return list(workbook.sheet_names)
    except Exception as exc:
        raise DatasetLoadError(f"Không đọc được danh sách sheet Excel: {exc}") from exc


def _read_jsonl(content: bytes | str) -> pd.DataFrame:
    try:
        text = content.decode("utf-8-sig") if isinstance(content, bytes) else content.lstrip("\ufeff")
    except UnicodeDecodeError as exc:
        raise DatasetLoadError("JSONL phải được mã hóa UTF-8 hoặc UTF-8-SIG.") from exc

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetLoadError(
                f"JSONL không hợp lệ ở dòng {line_number}, cột {exc.colno}: {exc.msg}"
            ) from exc
        if not isinstance(row, dict):
            raise DatasetLoadError(f"Mỗi dòng JSONL phải là một object (dòng {line_number}).")
        rows.append(row)
    return pd.DataFrame(rows)


def load_dataset(
    source: Any,
    filename: str | None = None,
    sheet_name: str | int | None = None,
) -> pd.DataFrame:
    """Read a CSV, XLSX sheet or JSONL file into a DataFrame.

    The function accepts a local path, raw bytes, or a binary file-like object
    (including Streamlit's UploadedFile). Input column names are left intact.
    """
    name = _filename(source, filename)
    extension = Path(name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DatasetLoadError(f"Định dạng {extension or '(không rõ)'} chưa được hỗ trợ. Dùng: {allowed}.")

    content = _read_bytes(source)
    try:
        if extension == ".csv":
            if isinstance(content, Path):
                frame = pd.read_csv(content, encoding="utf-8-sig")
            elif isinstance(content, str):
                frame = pd.read_csv(BytesIO(content.encode("utf-8-sig")), encoding="utf-8-sig")
            else:
                frame = pd.read_csv(BytesIO(content), encoding="utf-8-sig")
        elif extension == ".jsonl":
            frame = _read_jsonl(content)
        else:
            selected_sheet = 0 if sheet_name is None else sheet_name
            if isinstance(content, Path):
                frame = pd.read_excel(content, sheet_name=selected_sheet, engine="openpyxl")
            else:
                raw = content.encode("utf-8") if isinstance(content, str) else content
                frame = pd.read_excel(BytesIO(raw), sheet_name=selected_sheet, engine="openpyxl")
    except DatasetLoadError:
        raise
    except UnicodeDecodeError as exc:
        raise DatasetLoadError("CSV phải được mã hóa UTF-8 hoặc UTF-8-SIG.") from exc
    except Exception as exc:
        raise DatasetLoadError(f"Không thể đọc {name}: {exc}") from exc

    if not isinstance(frame, pd.DataFrame):
        raise DatasetLoadError("Không đọc được sheet dữ liệu trong tệp.")
    return frame
