"""会话等待发送者绑定的测试（#9377）。

模拟群聊场景：成员 A 通过空 @ 唤醒机器人进入等待后，
其他成员 B 的普通消息不应被会话等待拦截。
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from astrbot.core.utils.session_waiter import (
    FILTERS,
    USER_SESSIONS,
    DefaultSessionFilter,
    SenderSessionFilter,
    SessionController,
    session_waiter,
)


def _make_event(umo: str, sender_id: str) -> MagicMock:
    event = MagicMock()
    event.unified_msg_origin = umo
    event.get_sender_id.return_value = sender_id
    return event


async def _trigger_like_session_control_agent(event) -> bool:
    """模拟 builtin_stars/astrbot/main.py 中 handle_session_control_agent 的分发逻辑。

    返回该事件是否命中了某个正在等待的会话。
    """
    from astrbot.core.utils.session_waiter import SessionWaiter

    triggered = False
    for session_filter in list(FILTERS):
        session_id = session_filter.filter(event)
        if session_id in USER_SESSIONS:
            await SessionWaiter.trigger(session_id, event)
            triggered = True
    return triggered


def test_sender_session_filter_distinguishes_senders():
    group_umo = "aiocqhttp:GroupMessage:123456"
    filter_ = SenderSessionFilter()
    id_a = filter_.filter(_make_event(group_umo, "1001"))
    id_b = filter_.filter(_make_event(group_umo, "1002"))
    assert id_a != id_b

    # 默认过滤器对同群所有成员返回相同标识符（导致 #9377 的根因）
    default = DefaultSessionFilter()
    assert default.filter(_make_event(group_umo, "1001")) == default.filter(
        _make_event(group_umo, "1002")
    )


@pytest.mark.asyncio
async def test_sender_bound_waiter_ignores_other_members():
    group_umo = "aiocqhttp:GroupMessage:123456"
    handled_senders: list[str] = []

    @session_waiter(5)
    async def waiter(controller: SessionController, event) -> None:
        handled_senders.append(event.get_sender_id())
        controller.stop()

    event_a = _make_event(group_umo, "1001")
    wait_task = asyncio.create_task(
        waiter(event_a, session_filter=SenderSessionFilter())
    )
    await asyncio.sleep(0.05)  # 等待会话注册

    try:
        # 成员 B 在等待窗口内发送普通消息：不应被拦截
        event_b = _make_event(group_umo, "1002")
        assert await _trigger_like_session_control_agent(event_b) is False
        assert handled_senders == []

        # 成员 A 的后续消息：应被会话处理
        event_a2 = _make_event(group_umo, "1001")
        assert await _trigger_like_session_control_agent(event_a2) is True
        assert handled_senders == ["1001"]

        await asyncio.wait_for(wait_task, timeout=5)
    finally:
        if not wait_task.done():
            wait_task.cancel()


@pytest.mark.asyncio
async def test_sender_bound_waiters_coexist_for_multiple_members():
    group_umo = "aiocqhttp:GroupMessage:654321"
    handled: list[str] = []

    @session_waiter(5)
    async def waiter(controller: SessionController, event) -> None:
        handled.append(event.get_sender_id())
        controller.stop()

    task_a = asyncio.create_task(
        waiter(_make_event(group_umo, "2001"), session_filter=SenderSessionFilter())
    )
    task_b = asyncio.create_task(
        waiter(_make_event(group_umo, "2002"), session_filter=SenderSessionFilter())
    )
    await asyncio.sleep(0.05)

    try:
        assert await _trigger_like_session_control_agent(
            _make_event(group_umo, "2002")
        )
        assert await _trigger_like_session_control_agent(
            _make_event(group_umo, "2001")
        )
        await asyncio.gather(
            asyncio.wait_for(task_a, 5),
            asyncio.wait_for(task_b, 5),
        )
        assert sorted(handled) == ["2001", "2002"]
    finally:
        for t in (task_a, task_b):
            if not t.done():
                t.cancel()
