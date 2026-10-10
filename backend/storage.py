"""storage.py - Quản lý projects.json (CRUD, atomic save).
Thread-safe, atomic write để tránh corrupt file khi ghi đồng thời.
"""
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import config

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_cache: Optional[Dict[str, Any]] = None
_last_save_time: float = 0.0
_SAVE_DEBOUNCD: float = 2.0


def _db_path() -> Path:
    return config.DB_PATH


def _load() -> Dict[str, Any]:
    """Load projects từ file JSON. Thread-safe."""
    global _cache
    with _lock:
        if _cache is not None:
            return _cache
        path = _db_path()
        if not path.exists():
            _cache = {}
            return _cache
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _cache = data
            else:
                logger.warning("projects.json không phải dict, reset")
                _cache = {}
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error("projects.json hỏng: %s — tạo mới", e)
            _cache = {}
        return _cache


def _save(force: bool = False) -> None:
    """Ghi projects ra file JSON. Atomic write (ghi tmp rồi rename)."""
    global _last_save_time
    with _lock:
        now = time.time()
        if not force and (now - _last_save_time) < _SAVE_DEBOUNCD:
            return
        if _cache is None:
            return
        path = _db_path()
        tmp_path = path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(_cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(path)
            _last_save_time = time.time()
            logger.debug("DB saved atomically: %d projects", len(_cache))
        except Exception as e:
            logger.error("Lưu projects.json thất bại: %s", e)
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


# ============================================================
# PUBLIC API
# ============================================================

def list_projects() -> List[Dict[str, Any]]:
    """Liệt kê tất cả projects, sắp xếp theo created_ts giảm dần."""
    data = _load()
    projects = list(data.values())
    projects.sort(key=lambda p: p.get("created_ts", 0), reverse=True)
    return projects


def get_project(project_id: str) -> Optional[Dict[str, Any]]:
    """Lấy project theo ID."""
    data = _load()
    return data.get(project_id)


def upsert_project(project: Dict[str, Any]) -> None:
    """Tạo mới hoặc cập nhật toàn bộ project."""
    pid = project.get("id")
    if not pid:
        raise ValueError("Project phải có 'id'")
    with _lock:
        data = _load()
        data[pid] = project
    _save()
    logger.info(
        "Upsert project %s | segments=%d translated=%d pairs=%d",
        pid,
        len(project.get("segments", [])),
        len(project.get("translated_texts", [])),
        len(project.get("translation_pairs", [])),
    )


def update_project(project_id: str, **kwargs) -> None:
    """Cập nhật từng field của project."""
    with _lock:
        data = _load()
        if project_id not in data:
            logger.warning("update_project: project %s không tồn tại", project_id)
            return
        changed_keys = []
        for key, value in kwargs.items():
            old = data[project_id].get(key)
            if old != value:
                data[project_id][key] = value
                changed_keys.append(key)
        if not changed_keys:
            return
    _save()
    logger.info(
        "Update project %s: %s | segments=%d translated=%d pairs=%d",
        project_id, changed_keys,
        len(data[project_id].get("segments", [])),
        len(data[project_id].get("translated_texts", [])),
        len(data[project_id].get("translation_pairs", [])),
    )


def delete_project(project_id: str) -> None:
    """Xóa project khỏi DB."""
    with _lock:
        data = _load()
        if project_id in data:
            del data[project_id]
            logger.info("Deleted project %s", project_id)
        else:
            logger.warning("delete_project: project %s không tồn tại", project_id)
    _save(force=True)


def force_save() -> None:
    """Ép ghi ngay lập tức (bỏ qua debounce)."""
    _save(force=True)


def reload() -> None:
    """Reload cache từ đĩa (dùng khi file bị sửa ngoài)."""
    global _cache
    with _lock:
        _cache = None
    _load()
    logger.info("Storage cache reloaded")


# ============================================================
# ĐỐI TƯỢNG MODULE-LEVEL ĐỂ main.py IMPORT ĐƯỢC
# from storage import storage  →  storage.get_project(...)
# ============================================================
storage = sys.modules[__name__]