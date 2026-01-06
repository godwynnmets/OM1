import asyncio
import pytest
from actions.base import ActionConfig
from actions.move.connector.ros2 import MoveUnitreeSDKConnector
from actions.move.interface import MoveInput


class DummySportClient:
    def __init__(self):
        self.hello_called = False

    def SetTimeout(self, t):
        pass

    def Init(self):
        pass

    def StopMove(self):
        pass

    def Hello(self):
        self.hello_called = True


class DummyMoveInput(MoveInput):
    def __init__(self, action: str):
        self.action = action


@pytest.mark.asyncio
async def test_shake_paw_invokes_sport_client(monkeypatch):
    cfg = ActionConfig()
    conn = MoveUnitreeSDKConnector(cfg)

    # Inject a dummy sport client
    dummy = DummySportClient()
    conn.sport_client = dummy

    # Call connect with shake paw action
    input_msg = DummyMoveInput("shake paw")
    await conn.connect(input_msg)

    assert dummy.hello_called
