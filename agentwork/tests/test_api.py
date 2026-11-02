from smartpark.app import create_app


def test_status_and_mode(tmp_path):
    client = create_app(tmp_path / "parking.db").test_client()
    assert client.get("/api/status").status_code == 200
    response = client.post("/api/parking/mode", json={"mode": "exit"})
    assert response.get_json()["mode"] == "exit"
    assert client.post("/api/parking/mode", json={"mode": "invalid"}).status_code == 400
