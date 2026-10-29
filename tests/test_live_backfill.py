"""Backfill: the first session start in a repository reads the sessions the agent ran there
before, from its own logs, and folds each as a finished session is folded.

Each test writes past sessions where the agent keeps them (`CLAUDE_CONFIG_DIR`, `CODEX_HOME`,
under the test's directory), drives session starts through `hook.main`, and pins one property.
The job runs inline (`TE_FOLD_SYNC=1`); in the plugin it is a detached `te backfill`.
"""
import json
import os
import time

import pytest

from tests.test_live_core import _git_repo
from tests.test_live_hooks import call, repo, store_for, trap_session, write_session  # noqa: F401
from tokeneconomizer.plugin import backfill, cli, hosts
from tokeneconomizer.plugin.context import LiveContext

DAY = 86400


@pytest.fixture
def env(tmp_path):
    return {"TOKENECONOMIZER_HOME": str(tmp_path / "home"), "TE_FOLD_SYNC": "1",
            "TE_MICROSKILLS": "envfacts", "TE_TOOLPRUNE_QUIET_MINUTES": "0",
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"), "CODEX_HOME": str(tmp_path / "codex"),
            "GIT_CEILING_DIRECTORIES": str(tmp_path), "PATH": os.environ.get("PATH", "")}


def _iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(t))


def past(env, cwd, sid, days_ago, prompts=(), writer=None, sidechain=False, path=None):
    """A Claude Code transcript of a session that started in `cwd` `days_ago` days ago, filed
    where Claude Code files it (or at `path`), its log last written ten minutes later."""
    path = path or os.path.join(env["CLAUDE_CONFIG_DIR"], "projects", cwd.replace("/", "-"),
                                f"{sid}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    (writer or (lambda p: write_session(p, prompts)))(path)
    started = time.time() - days_ago * DAY
    head = {"type": "queue-operation", "sessionId": sid, "cwd": cwd, "timestamp": _iso(started),
            "isSidechain": sidechain}
    with open(path) as fh:
        body = fh.read()
    with open(path, "w") as fh:
        fh.write(json.dumps(head) + "\n" + body)
    os.utime(path, (started + 600, started + 600))
    return path


def edit_and_test(repo, passed=True):
    return [("Edit", {"file_path": f"{repo}/a.py", "old_string": "a", "new_string": "b"},
             "ok", False),
            ("Bash", {"command": "python -m pytest -q"},
             "3 passed in 0.1s" if passed else "Exit code 1\n1 failed, 2 passed in 0.1s",
             not passed)]


def start(env, repo, sid="now"):
    return call(env, {"hook_event_name": "SessionStart", "session_id": sid, "source": "startup",
                      "transcript_path": "", "cwd": repo})


def ledger(env, repo, host="claude"):
    return store_for(env, repo, host).read_lines("ledger.jsonl")


def test_the_first_start_folds_past_sessions_oldest_first_marked_backfill(env, repo):
    for i in range(3):
        past(env, repo, f"old{i}", 3 - i, writer=lambda p, i=i: trap_session(p, repo, i))
    start(env, repo)
    recs = ledger(env, repo)
    assert [r["session_id"] for r in recs] == ["old0", "old1", "old2"]
    assert all(r["source"] == "backfill" and r["kind"] == "session" for r in recs)
    assert all(r["effective_tokens"] for r in recs)                     # priced from the log
    assert store_for(env, repo).read("injection.json")["sessions"] == 3
    assert store_for(env, repo).read("untool.json")["sessions"] == 3


def test_the_next_start_says_once_how_many_it_read(env, repo):
    for i in range(2):
        past(env, repo, f"old{i}", 2 - i, [("go", edit_and_test(repo))])
    assert "earlier" not in json.dumps(start(env, repo, "s1"))   # the job runs after it
    line = start(env, repo, "s2")["systemMessage"]
    assert "read 2 earlier claude session(s) of acme/widget" in line
    assert "earlier" not in json.dumps(start(env, repo, "s3"))


def test_it_counts_a_subdirectory_a_clone_and_a_worktree_not_another_repository(
        env, repo, tmp_path):
    os.makedirs(os.path.join(repo, "pkg"))
    clone = _git_repo(str(tmp_path / "clone"), "git@github.com:acme/widget.git")
    wt_git = os.path.join(repo, ".git", "worktrees", "wt")
    os.makedirs(wt_git)
    with open(os.path.join(wt_git, "commondir"), "w") as fh:
        fh.write("../..")
    wt = str(tmp_path / "wt")
    os.makedirs(wt)
    with open(os.path.join(wt, ".git"), "w") as fh:
        fh.write(f"gitdir: {wt_git}\n")
    other = _git_repo(str(tmp_path / "other"), "https://github.com/acme/gadget.git")
    for sid, cwd in (("sub", os.path.join(repo, "pkg")), ("clone", clone), ("wt", wt),
                     ("other", other), ("gone", str(tmp_path / "removed-worktree"))):
        past(env, cwd, sid, 1, [("go", edit_and_test(repo))])
    start(env, repo)
    assert sorted(r["session_id"] for r in ledger(env, repo)) == ["clone", "sub", "wt"]


def test_it_skips_this_session_read_running_subagent_and_tool_less_ones(env, repo):
    calls = [("go", edit_and_test(repo))]
    past(env, repo, "done", 6, calls)
    store_for(env, repo).write("folded.json", {"done": {"at": 1, "transcript": "x"}})
    past(env, repo, "chat", 5, [("hello", [])])
    past(env, repo, "side", 4, calls, sidechain=True)
    parent = past(env, repo, "parent", 3, calls)
    past(env, repo, "agent-x", 2, calls,
         path=os.path.join(parent[:-len(".jsonl")], "subagents", "agent-x.jsonl"))
    os.utime(past(env, repo, "running", 1, calls))
    past(env, repo, "now", 1, calls)
    start(env, repo, "now")
    assert [r["session_id"] for r in ledger(env, repo)] == ["parent"]


def test_it_reads_the_most_recent_n_and_te_backfill_adds_only_what_is_missing(
        env, repo, monkeypatch, capsys):
    for i in range(4):
        past(env, repo, f"old{i}", 4 - i, [("go", edit_and_test(repo))])
    start(dict(env, TE_BACKFILL_SESSIONS="2"), repo)
    assert [r["session_id"] for r in ledger(env, repo)] == ["old2", "old3"]
    for k, v in dict(env, TE_BACKFILL_SESSIONS="2").items():
        monkeypatch.setenv(k, v)
    assert cli.main(["--cwd", repo, "--host", "claude", "backfill"]) == 0
    assert capsys.readouterr().out.startswith("read 0 earlier claude session(s)")
    monkeypatch.setenv("TE_BACKFILL_SESSIONS", "50")
    assert cli.main(["--cwd", repo, "--host", "claude", "backfill"]) == 0
    assert [r["session_id"] for r in ledger(env, repo)] == ["old2", "old3", "old0", "old1"]


def test_backfill_off_leaves_the_history_alone(env, repo):
    past(env, repo, "old", 1, [("go", edit_and_test(repo))])
    off = dict(env, TE_BACKFILL="off")
    start(off, repo, "s1")
    start(off, repo, "s2")
    assert ledger(env, repo) == []


def test_past_turns_replay_through_the_guard_and_only_green_ones_feed_the_envelope(env, repo):
    def session(path):
        write_session(path, [("green", edit_and_test(repo)), ("red", edit_and_test(repo, False)),
                             ("read", [("Read", {"file_path": f"{repo}/a.py"}, "x", False)] * 3)])
        with open(path, "a") as fh:                      # the person stopped that last turn
            fh.write(json.dumps({"type": "user", "message": {"role": "user", "content": [
                {"type": "text", "text": "[Request interrupted by user]"}]}}) + "\n")
    past(env, repo, "old", 1, writer=session)
    start(env, repo)
    assert store_for(env, repo).read("envelope.json")["resolved_turns"] == [2]


def test_codex_rollouts_are_read_and_subagent_threads_are_not(env, repo):
    def rollout(sid, days_ago, source="vscode"):
        d = os.path.join(env["CODEX_HOME"], "sessions", "2026", "10", "01")
        os.makedirs(d, exist_ok=True)
        when = time.time() - days_ago * DAY
        item = {"type": "event_msg", "payload": {"type": "item_completed"}}
        recs = [{"type": "session_meta", "payload": {"id": sid, "cwd": repo, "source": source,
                                                     "timestamp": _iso(when)}},
                {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "t1"}},
                dict(item, payload=dict(item["payload"], item={
                    "type": "FileChange", "id": "f", "status": "completed",
                    "changes": {f"{repo}/a.py": {"type": "update"}}})),
                dict(item, payload=dict(item["payload"], item={
                    "type": "CommandExecution", "id": "c", "command": ["bash", "-lc", "pytest"],
                    "aggregated_output": "3 passed in 0.1s", "exit_code": 0})),
                {"type": "event_msg", "payload": {"type": "token_count", "info": {
                    "total_token_usage": {"input_tokens": 3000, "cached_input_tokens": 2000,
                                          "output_tokens": 90}}}}]
        path = os.path.join(d, f"rollout-2026-10-01T00-00-00-{sid}.jsonl")
        with open(path, "w") as fh:
            fh.write("\n".join(json.dumps(r) for r in recs) + "\n")
        os.utime(path, (when, when))
    rollout("thread-a", 2)
    rollout("thread-b", 1, source={"subagent": {"thread_spawn": {"parent_thread_id": "a"}}})
    call(env, {"hook_event_name": "SessionStart", "session_id": "now", "source": "startup",
               "transcript_path": "", "cwd": repo}, "codex")
    assert [(r["session_id"], r["source"]) for r in ledger(env, repo, "codex")] == [
        ("thread-a", "backfill")]
    assert store_for(env, repo, "codex").read("envelope.json")["resolved_turns"] == [2]


def test_a_second_job_leaves_while_one_runs(env, repo):
    past(env, repo, "old", 1, [("go", edit_and_test(repo))])
    ctx = LiveContext.build(hosts.HOSTS["claude"](), repo, env)
    with ctx.store.locked("backfill"):
        assert backfill.run(ctx) is None
    assert backfill.run(ctx) == 1
    assert backfill.run(ctx) == 0
