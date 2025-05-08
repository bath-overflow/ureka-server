import pytest
from fastapi.testclient import TestClient

from server.main import app
from server.services.chat import ChatEvent, ChatInfo


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
        c.close()


def test_connect_websocket(client):
    with client.websocket_connect("/ws/chat") as websocket:
        # Check if the connection is established
        assert websocket is not None

        # Receive the message
        data = websocket.receive_text()
        assert data == f"{ChatEvent.CONNECTED.value}: {ChatInfo.CONNECTED.value}"

        # Close the connection
        websocket.close()
