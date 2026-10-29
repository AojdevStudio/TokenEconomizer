"""Learning from the sessions an agent ran in a repository before the plugin saw them.

The fold learns from each session as it ends, so a repository starts blank however much history
the agent keeps. The first session start in a repository leaves a detached `te backfill` that
reads that history from the agent's own logs (`Host.history_logs`: Claude Code's
`~/.claude/projects`, Codex's `~/.codex/sessions`) and gives each past session what a finished
one gets: `fold.fold_session` (micro-skills, tool pruning's evidence, and the ledger, its record
marked `source: "backfill"`), and then the loop guard, each prompt-turn replayed call by call as
its hooks would have received it, so that the turns that ended green feed the envelope as their
Stop hooks would have.

  * Which sessions. One counts when the directory its log names resolves to this repository now
    (`paths.repo_identity`, the key a live session gets): its checkout, a subdirectory, another
    clone, a worktree. A directory that is gone says nothing, and the session is left out. So are
    a subagent's log, a log written to within `fold.STALE_S` (a session still running, this one
    included), and a session that called no tool. Of the rest, the most recent
    `backfill_sessions` are the window, read oldest first.
  * Read once. A session is processed when `folded.json` holds its id or its log, as it does for
    the live fold, or when it has a record under `sessions/` (the hooks', or one backfill wrote).
    `fold_session` folds a session once whoever asks, so a run never double-counts one, and
    running again folds only what is still missing from the window. The record backfill writes
    (`origin: "backfill"`) is what `repos` and the next session start need to fold the session
    again after a merge, and it says whether its turns were replayed yet, so a job killed between
    a session's fold and its replay leaves the replay to the next run.
  * In order. Micro-skills and tool pruning learn in the order sessions ran: a fact later
    sessions retired must stay retired. When the window holds a session older than one already
    folded, the learned state is set aside (`prebackfill-*`, nothing is deleted) and every folded
    session is folded again, with the window's, oldest first, as `repos` does after a merge. That
    needs every folded session's log: when one is gone, the history is folded after them instead,
    out of order, rather than lose what that session taught.
  * One job at a time. `backfill.json` says a job was started and when it finished; concurrent
    session starts may each leave one, and the job takes the `backfill` lock or leaves (or, run
    after the first, reads nothing new). A job that died is started again by a session start
    `RESPAWN_S` later.
"""
import json
import os
import tempfile
import time

from .. import loopguard
from . import fold
from .paths import repo_identity
from .repos import LEARNED

STATE = "backfill.json"
RESPAWN_S = 3600


def _processed(ctx):
    """(session ids, log paths) the live fold has, or will have, folded."""
    folded = ctx.store.read("folded.json") or {}
    ids, logs = set(folded), {v.get("transcript") for v in folded.values()
                             if isinstance(v, dict)}
    try:
        names = os.listdir(ctx.store.path("sessions"))
    except OSError:
        names = []
    for name in names:
        if name.endswith(".json"):
            ids.add(name[:-5])
            logs.add((ctx.store.read(os.path.join("sessions", name)) or {}).get("transcript"))
    logs.discard(None)
    return ids, {os.path.realpath(p) for p in logs}


def window(ctx, now=None):
    """The past sessions of this repository to read: [(started, session id, log path)], oldest
    first, those already processed left out."""
    now = now or time.time()
    keys, found = {}, []
    for path in ctx.host.history_logs(ctx.env):
        try:
            if now - os.path.getmtime(path) < fold.STALE_S:
                continue
            info = ctx.host.history_session(path)
        except OSError:
            continue
        cwd = info.get("cwd")
        if not info.get("session_id") or info.get("subagent") or not cwd \
                or not os.path.isdir(cwd):
            continue
        if cwd not in keys:
            keys[cwd] = repo_identity(cwd, ctx.env).key
        if keys[cwd] == ctx.repo.key:
            found.append((info.get("started") or os.path.getmtime(path), info["session_id"],
                          path))
    ids, logs = _processed(ctx)
    picked = []
    for started, sid, path in sorted(found, reverse=True):
        if len(picked) >= ctx.config.backfill_sessions:
            break
        if any(payloads for payloads, _ in ctx.host.history_turns(path)):
            picked.append((started, sid, path, sid in ids or os.path.realpath(path) in logs))
    return [(started, sid, path) for started, sid, path, done in sorted(picked) if not done]


def replay_turns(ctx, path):
    """Each prompt-turn of a past session through the loop guard, as `controller` runs a live
    one: every call folded from the payload its PostToolUse hook would have received, and a turn
    that ended green (not interrupted, no failing check after its last edit) recorded in the
    envelope as `loopguard.record_turn` records it, in one write for the whole session. The
    session ran unguarded, so nothing is decided here: no envelope, no judge."""
    with ctx.store.locked("envelope"):
        tracker = loopguard.load_envelope(ctx.store, ctx.config.loopguard_aggr,
                                          ctx.config.loopguard_thresholds)
        for payloads, stopped in ctx.host.history_turns(path):
            g = loopguard.TurnGuard()
            for p in payloads:
                g.fold(*ctx.host.tool_call(p))
            if g.cum_turns and stopped:
                tracker.update(g.cum_turns, g.resolved_proxy(), snapshot=g.whole.snapshot())
        loopguard.save_envelope(ctx.store, tracker)


def _records(ctx):
    """{session id: record} under `sessions/`."""
    try:
        names = os.listdir(ctx.store.path("sessions"))
    except OSError:
        return {}
    return {n[:-5]: ctx.store.read(os.path.join("sessions", n)) or {}
            for n in names if n.endswith(".json")}


def _unfold_after(ctx, since):
    """When a folded session started after `since`, and every folded session's log still exists
    to fold it again from: set the learned state aside and unmark them all, their ledger records
    dropped (their folds write them again). Returns [(started, session id, log)] to fold again,
    [] when the order holds already, or when a log is gone: then the history is folded after
    them, since folding again without that session would lose what it taught. The caller holds
    the fold lock."""
    folded = ctx.store.read("folded.json") or {}
    recs = _records(ctx)
    again = []
    for sid, entry in folded.items():
        log = (entry or {}).get("transcript") or recs.get(sid, {}).get("transcript") or ""
        if not os.path.exists(log):
            return []
        again.append((float(recs.get(sid, {}).get("started")
                            or ctx.host.history_session(log).get("started") or 0), sid, log))
    if not any(t > since for t, _, _ in again):
        return []
    ids = set(folded)
    with ctx.store.locked("mechanisms"):
        aside = None
        for name in LEARNED:
            if os.path.lexists(ctx.store.path(name)):
                aside = aside or tempfile.mkdtemp(
                    prefix=time.strftime("prebackfill-%Y%m%d-%H%M%S-"), dir=ctx.store.base)
                os.replace(ctx.store.path(name), os.path.join(aside, name))
        ctx.store.write("folded.json", {})
        keep = [r for r in ctx.store.read_lines("ledger.jsonl")
                if not (r.get("kind") == "session" and r.get("session_id") in ids)]
        tmp = ctx.store.path(f"ledger.jsonl.{os.getpid()}.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, sort_keys=True) + "\n" for r in keep)
        os.replace(tmp, ctx.store.path("ledger.jsonl"))
        for sid in ids & set(recs):
            ctx.store.write(os.path.join("sessions", f"{sid}.json"),
                            dict(recs[sid], folded=False))
    return again


def run(ctx, announce=True):
    """Read the window's sessions, with those a killed run recorded but never folded, folding
    again the folded sessions that started after the oldest of them (`_unfold_after`); then
    replay the turns of every backfilled session not replayed yet. It holds the fold lock
    throughout, so no live fold lands in the middle of that order. Returns how many sessions
    were read, or None while another run, or a fold past its wait, holds a lock. With
    `announce`, the next session start says how many."""
    with ctx.store.locked("backfill", timeout=0) as got:
        if not got:
            return None
        with ctx.store.locked("fold", timeout=fold.FOLD_LOCK_WAIT_S) as held:
            if not held:
                return None
            todo = [(t, sid, path, "new") for t, sid, path in window(ctx)]
            folded = ctx.store.read("folded.json") or {}
            todo += [(float(rec.get("started") or 0), sid, rec.get("transcript"), "stranded")
                     for sid, rec in _records(ctx).items()
                     if rec.get("origin") == "backfill" and sid not in folded]
            todo.sort()
            again = _unfold_after(ctx, todo[0][0]) if todo else []
            n = 0
            for started, sid, path, kind in sorted(todo + [(*a, "again") for a in again]):
                if kind == "new":
                    info = ctx.host.history_session(path)
                    ctx.store.write(os.path.join("sessions", f"{sid}.json"), {
                        "session_id": sid, "host": ctx.host.name, "origin": "backfill",
                        "cwd": info.get("cwd"), "transcript": path, "started": started,
                        "ended": os.path.getmtime(path), "turns": [], "replayed": False})
                rec = fold.fold_session(ctx, sid, path, held=True)
                n += rec is not None and kind != "again"
        if ctx.config.loopguard != "off":
            recs = _records(ctx)
            for sid, rec in sorted(recs.items(), key=lambda kv: float(kv[1].get("started") or 0)):
                if rec.get("origin") == "backfill" and rec.get("folded") \
                        and not rec.get("replayed"):
                    replay_turns(ctx, rec["transcript"])
                    ctx.store.write(os.path.join("sessions", f"{sid}.json"),
                                    dict(rec, replayed=True))
        doc = ctx.store.read(STATE) or {}
        # Added to what a run before it read and no session start has announced yet: a second
        # job left by a concurrent start finds nothing, and must not erase the first one's count.
        ctx.store.write(STATE, dict(doc, finished=time.time(), read=n,
                                    unannounced=int(doc.get("unannounced") or 0)
                                    + (n if announce else 0)))
        return n


def start(ctx):
    """At a session start: leave the job, once per repository and agent (again after one that
    died). Inline under `TE_FOLD_SYNC=1`, for tests."""
    if ctx.config.backfill == "off":
        return False
    doc = ctx.store.read(STATE) or {}
    if doc.get("finished") or time.time() - float(doc.get("started") or 0) < RESPAWN_S:
        return False
    ctx.store.write(STATE, dict(doc, started=time.time()))
    if ctx.env.get("TE_FOLD_SYNC") == "1":
        run(ctx)
        return True
    return ctx.spawn(["backfill", "--quiet"], "backfill.log")


def notice(ctx, command):
    """Once, after a job read anything: a sentence saying how many sessions (else None)."""
    if not (ctx.store.read(STATE) or {}).get("unannounced"):
        return None
    with ctx.store.locked("backfill", timeout=0) as got:
        if not got:
            return None
        doc = ctx.store.read(STATE) or {}
        if not doc.get("unannounced"):
            return None
        ctx.store.write(STATE, dict(doc, unannounced=0))
    return (f"TokenEconomizer read {doc['unannounced']} earlier {ctx.host.name} session(s) of "
            f"{ctx.repo.label} from this agent's own logs (`{command}`).")
