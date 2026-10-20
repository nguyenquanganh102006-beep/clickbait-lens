"""A fixed, non-private article used for real labeling capability checks."""
from src.labeling.engine import LabelingEngine

SAMPLE = {
    "record_id": "connection-check",
    "title": "Thư viện thành phố mở cửa trở lại",
    "content": "Thư viện thành phố mở cửa trở lại sau hai tháng sửa chữa. Bạn đọc có thể mượn sách từ 8 giờ đến 17 giờ mỗi ngày. Khu đọc sách đã được bổ sung bàn ghế, sách mới và hệ thống chiếu sáng.",
}


def probe_labeling(provider, tasks, mode="independent"):
    tasks = list(tasks)
    execution = LabelingEngine(provider).label_record(SAMPLE, tasks, mode=mode)
    if set(execution.outputs) != set(tasks) or not execution.usage_by_task:
        raise ValueError("Chưa gọi API và kiểm tra đủ tác vụ.")
    if mode == "independent" and set(execution.usage_by_task) != set(tasks):
        raise ValueError("Chưa gọi đủ API cho các tác vụ độc lập.")
    failed = [task for task, output in execution.outputs.items() if output.status != "success"]
    if failed:
        raise ValueError("Bài mẫu chưa được gán nhãn thành công: " + ", ".join(failed))
    return execution
