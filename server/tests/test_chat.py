import pytest
from fastapi.testclient import TestClient

from server.main import app
from server.routers.chat import ChatEvent


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
        assert data == f"{ChatEvent.CONNECTED.value}: A user has connected."

        # Close the connection
        websocket.close()


def test_chat_websocket(client):
    with client.websocket_connect("/ws/chat") as websocket:
        # Check if the connection is established
        assert websocket is not None

        data = websocket.receive_text()

        # Send a message
        test_message = "Hello, World!"
        websocket.send_text(test_message)

        # Receive the echo message
        data = websocket.receive_text()
        assert data == f"{ChatEvent.MESSAGE_RECEIVED.value}: {test_message}"

        # Close the connection
        websocket.close()
