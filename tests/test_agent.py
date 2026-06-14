from agent import triage_step


def test_triage_completes_with_reviewed_action_plan(monkeypatch):
    monkeypatch.setattr("agent.log_trace", lambda *args, **kwargs: "trace")
    state = {}
    replies = []
    for answer in [
        "An adult is choking",
        "Yes",
        "No",
        "No severe bleeding",
        "They cannot speak or cough",
    ]:
        reply, state = triage_step(answer, state)
        replies.append(reply)
    assert state["complete"] is True
    assert "ACTION PLAN" in replies[-1]
    assert "Source:" in replies[-1]
