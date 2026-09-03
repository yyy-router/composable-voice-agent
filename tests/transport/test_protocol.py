from composable_voice_agent.transport.websocket.protocol import (
    ProtocolError,
    event_message,
    parse_control_message,
)


def test_parse_control_message() -> None:
    message = parse_control_message({"type": "audio.start", "payload": {"format": "pcm"}})
    assert message.type == "audio.start"
    assert message.payload["format"] == "pcm"


def test_protocol_rejects_invalid_message() -> None:
    try:
        parse_control_message({"type": "audio.start", "payload": []})
    except ProtocolError:
        pass
    else:
        raise AssertionError("invalid payload was accepted")


def test_event_message_has_stable_shape() -> None:
    assert event_message("session.ready") == {
        "version": 1,
        "type": "session.ready",
        "payload": {},
    }
