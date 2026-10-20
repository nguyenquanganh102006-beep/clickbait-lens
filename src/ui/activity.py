"""Visible operation status and a small session-only activity history."""
from contextlib import contextmanager
from datetime import datetime
from time import perf_counter
from zoneinfo import ZoneInfo

import streamlit as st

from src.providers.base import sanitize_error_message


@contextmanager
def activity(label: str):
    started = perf_counter()
    item = {"action": label, "time": datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds"), "state": "Đang xử lý"}
    history = list(st.session_state.get("activity_history", []))
    history.append(item)
    st.session_state["activity_history"] = history[-10:]
    with st.status(f"Đang xử lý: {label}", expanded=True) as status:
        try:
            yield status
        except Exception as exc:
            secret = st.session_state.get("provider_api_key", "")
            detail = sanitize_error_message(exc, secret)
            item.update(state="Lỗi", detail=detail, seconds=round(perf_counter() - started, 1))
            status.update(label=f"Lỗi tại: {label}", state="error", expanded=True)
            status.write(detail)
            raise
        else:
            item.update(state="Hoàn tất", seconds=round(perf_counter() - started, 1))
            status.update(label=f"Hoàn tất: {label} · {item['seconds']} giây", state="complete", expanded=False)
        finally:
            st.session_state["activity_history"] = history[-10:]


def render_activity_history():
    history = st.session_state.get("activity_history", [])
    with st.expander("Hoạt động gần đây", expanded=bool(history and history[-1]["state"] == "Lỗi")):
        if not history:
            st.caption("Chưa có thao tác xử lý trong phiên này.")
        for item in reversed(history):
            st.write(f"{item['state']} · {item['action']}")
            st.caption(f"{item['time']} · {item.get('seconds', 0)} giây")
            if item.get("detail"):
                st.error(item["detail"])
