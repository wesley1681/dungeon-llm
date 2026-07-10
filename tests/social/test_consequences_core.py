from tests.social.helpers import make_world
from trpg.engine import consequences as cq


def setup_function(_fn):
    # 每個測試用干淨的假模板，不污染全域註冊表
    cq.CONSEQUENCE_REGISTRY.pop("_T_OK", None)
    cq.CONSEQUENCE_REGISTRY.pop("_T_BAD", None)
    cq._register(cq.ConsequenceTemplate(
        "_T_OK",
        validate=lambda p, w, ctx: [] if "x" in p else ["缺 x"],
        execute=lambda p, w, ctx: w.social.flags.add(f"ok:{p['x']}") or f"執行了 {p['x']}"))
    cq._register(cq.ConsequenceTemplate(
        "_T_BAD",
        validate=lambda p, w, ctx: ["永遠不合法"],
        execute=lambda p, w, ctx: "不該被執行"))


def teardown_function(_fn):
    cq.CONSEQUENCE_REGISTRY.pop("_T_OK", None)
    cq.CONSEQUENCE_REGISTRY.pop("_T_BAD", None)


def test_unknown_template_rejected():
    ws = make_world()
    ok, msgs = cq.commit_bundle(ws, [{"template": "NO_SUCH", "params": {}}], actor="party")
    assert not ok and any("未註冊" in m for m in msgs)
    assert len(ws.social.ledger) == 0


def test_bundle_is_atomic_validate_all_before_execute():
    ws = make_world()
    ok, msgs = cq.commit_bundle(
        ws,
        [{"template": "_T_OK", "params": {"x": "a"}},
         {"template": "_T_BAD", "params": {}}],
        actor="party")
    assert not ok
    assert "ok:a" not in ws.social.flags        # 第一個合法實例也不得執行
    assert len(ws.social.ledger) == 0           # 整束不落帳


def test_success_executes_and_ledgers_each_with_ruling():
    ws = make_world()
    ruling = {"kind": "check", "dc": 15, "roll": 18, "success": True, "reason": "唬騙"}
    ok, msgs = cq.commit_bundle(
        ws,
        [{"template": "_T_OK", "params": {"x": "a"}},
         {"template": "_T_OK", "params": {"x": "b"}}],
        actor="party", ruling=ruling)
    assert ok and msgs == ["執行了 a", "執行了 b"]
    commits = ws.social.ledger.query(template_id="_T_OK")
    assert len(commits) == 2
    assert all(c.ruling["dc"] == 15 for c in commits)
    assert all(c.time_minutes == ws.social.time_minutes for c in commits)
