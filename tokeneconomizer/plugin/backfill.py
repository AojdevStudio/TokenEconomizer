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
    the live fold, or when the hooks recorded it (the live fold's to fold). `fold_session` folds
    a session once whoever asks, so a run never double-counts one, and running again folds only
    what is still missing from the window.
  * One job at a time. `backfill.json` says a job was started and when it finished; concurrent
    session starts may each leave one, and the job takes the `backfill` lock or leaves. A job
    that died is started again by a session start `RESPAWN_S` later.
"""
import os
import time

from .. import loopguard
from . import fold
from .paths import repo_identity

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
    """The past sessions of this repository to read: [(session id, log path)], oldest first,
    those already processed left out."""
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
        done = sid in ids or os.path.realpath(path) in logs
        if done or any(payloads for payloads, _ in ctx.host.history_turns(path)):
            picked.append((started, sid, path, done))
    return [(sid, path) for _, sid, path, done in sorted(picked) if not done]


def replay_turns(ctx, path):
    """Each prompt-turn of a past session through the loop guard, as `controller` runs a live
    one: every call folded from the payload its PostToolUse hook would have received, and a turn
    that ended green (not interrupted, no failing check after its last edit) recorded in the
    envelope. The session ran unguarded, so nothing is decided here: no envelope, no judge."""
    for payloads, stopped in ctx.host.history_turns(path):
        g = loopguard.TurnGuard()
        for p in payloads:
            g.fold(*ctx.host.tool_call(p))
        if g.cum_turns and stopped:
            loopguard.record_turn(ctx.store, g, level=ctx.config.loopguard_aggr,
                                  overrides=ctx.config.loopguard_thresholds)


def run(ctx, announce=True):
    """Read the window's sessions. Returns how many were folded, or None while another run
    holds the lock. With `announce`, the next session start says how many."""
    with ctx.store.locked("backfill", timeout=0) as got:
        if not got:
            return None
        n = 0
        for sid, path in window(ctx):
            if fold.fold_session(ctx, sid, path, source="backfill") is None:
                continue
            if ctx.config.loopguard != "off":
                replay_turns(ctx, path)
            n += 1
        doc = ctx.store.read(STATE) or {}
        ctx.store.write(STATE, dict(doc, finished=time.time(), read=n,
                                    noticed=not (announce and n)))
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
    doc = ctx.store.read(STATE) or {}
    if doc.get("noticed", True):
        return None
    with ctx.store.locked("backfill", timeout=0) as got:
        if not got:
            return None
        doc = ctx.store.read(STATE) or {}
        if doc.get("noticed", True):
            return None
        ctx.store.write(STATE, dict(doc, noticed=True))
    return (f"TokenEconomizer read {doc.get('read', 0)} earlier {ctx.host.name} session(s) of "
            f"{ctx.repo.label} from this agent's own logs (`{command}`).")
