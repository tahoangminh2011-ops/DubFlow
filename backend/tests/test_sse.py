from fastapi.testclient import TestClient

import main


def test_ocr_sse_replays_terminal_event_and_closes(monkeypatch):
    pid = "sse_unit_test"
    monkeypatch.setattr(main, "_project_or_404", lambda _pid: {"id": _pid})
    main._clear_ocr_event_bus(pid)
    main._publish_ocr_event(pid, {"type": "progress", "phase": "scan", "overall_percent": 10})
    main._publish_ocr_event(pid, {"type": "complete", "overall_percent": 100, "message": "done"})

    with TestClient(main.app) as client:
        response = client.get(f"/api/projects/{pid}/ocr/events")

    assert response.status_code == 200
    assert response.text.count('"type":"complete"') == 1
    assert ": heartbeat" not in response.text
    assert not main._ocr_event_subscribers.get(pid)
