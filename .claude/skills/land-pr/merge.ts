/**
 * The merge decision: may this PR head merge, judged from the review bot's check-runs.
 * Pure, so the gate and tests share it. The gate fetches the state over the REST API.
 *
 * Pullfrog is the active review bot. Its `pullfrog-approval` check-run concluding
 * `success` on the exact head is the pass signal. The gate reads the latest run per check
 * suite, and Pullfrog has one suite per commit, so its reruns collapse to one run. A second
 * run of that name, or a run from another app, means another producer used the name, and
 * no verdict on that commit is trusted. Every merge requires a successful verdict on the
 * exact head, including a merge commit that resolves conflicts.
 *
 * The kit's source repo is the owner's curated settings repo, and no review bot gates it.
 * The owner also exempted two repos with no review bot installed, which take local reviews only
 * (2026-10-27), and six product meta (planning) repos with no review bot installed (2026-10-29). The list lives here in the kit, never in a file the target repo could edit. It holds SHA-256
 * digests of lowercase `owner/name`, so the kit can ship to public repos without naming a
 * private one. The exemption holds only when the merge command names the repo itself,
 * because a repo guessed from the cwd or the environment can differ from the one gh merges.
 */

export const PASS_SIGNAL = "pullfrog-approval";
/** Slug of the GitHub App that publishes PASS_SIGNAL. */
export const PASS_APP = "pullfrog";

const BOT_EXEMPT = new Set([
  "sha256:61:59:f3:8c:55:de:a3:14:c9:6c:15:a2:b5:dd:f2:ed:d3:2a:5b:51:01:9c:9d:0b:45:21:e2:ff:07:7c:08:15",
  "sha256:84:25:c2:e9:1b:73:ef:6b:8e:23:37:0c:43:b9:c5:d4:bb:3d:b3:76:68:4c:ba:dc:c1:31:c2:69:df:f4:7b:be",
  "sha256:2a:c1:f2:9e:c5:a3:38:67:97:c9:ab:31:e1:3b:8c:4f:63:46:b3:92:11:eb:40:66:e9:3f:44:8b:4c:f7:96:74",
  "sha256:9b:98:ef:a7:4d:ad:15:2a:b3:62:db:66:30:bb:27:60:3c:d2:ac:3c:13:b5:a5:65:44:32:a4:1e:3e:ca:46:3b",
  "sha256:c8:17:59:66:37:f0:0a:16:33:ec:6c:b2:54:96:61:8f:52:15:4b:43:eb:1a:e2:84:28:bc:b0:24:7a:34:72:6a",
  "sha256:00:cb:41:cb:ec:c8:f6:49:69:37:8b:d6:74:1e:bc:84:8b:e3:0a:77:3b:9e:14:b9:9f:36:09:18:07:20:3f:4a",
  "sha256:d0:b2:20:80:6e:04:c3:c3:e1:fb:f9:d5:b5:b1:47:14:10:c1:92:3d:b0:f0:10:c7:1c:5f:a0:1d:61:99:fb:a4",
  "sha256:31:4d:64:65:db:3e:85:9f:55:3a:7d:87:a9:0f:a9:55:9a:c6:81:1b:8f:a7:14:b9:69:d0:9a:53:6e:82:69:03",
  "sha256:8a:60:35:cf:9f:fe:e9:f6:e6:73:7e:2b:f7:84:b6:13:af:92:3f:a7:e4:f4:28:ba:d5:3e:d5:de:f6:7a:9b:07",
  "sha256:51:1f:ed:46:6f:52:26:97:e0:71:43:ad:a6:19:ec:44:c7:32:52:01:90:5d:9c:ac:b5:5f:b5:76:35:fa:7c:a8",
]);

/** True for a repo that no review bot gates. `repo` is `owner/name`, any case. */
export const botExempt = (repo: string): boolean =>
  BOT_EXEMPT.has(new Bun.CryptoHasher("sha256").update(repo.toLowerCase()).digest("hex").replace(/(..)(?=.)/g, "$1:").replace(/^/, "sha256:"));

export interface CheckRun {
  name: string;
  status: string;
  conclusion: string | null;
  /** Slug of the GitHub App that created the run. */
  app: string;
}

export interface HeadState {
  /** The PR's base repo as GitHub reports it, `owner/name`. */
  repo: string;
  /** The merge command named the repo with `-R`, a PR URL, or a literal REST path. */
  pinned: boolean;
  sha: string;
  /** Latest check-runs named PASS_SIGNAL on the head. */
  runs: CheckRun[];
}

export type MergeVerdict =
  | { ok: true; via: "approval" | "exempt" }
  | { ok: false; reason: string };

/** The bot's current verdict on one commit. Only a completed exact-head success passes. */
type Verdict = { state: "pass" } | { state: "refused"; why: string };

function verdictOf(runs: CheckRun[]): Verdict {
  const named = runs.filter((r) => r.name === PASS_SIGNAL);
  if (named.length === 0) return { state: "refused", why: `no ${PASS_SIGNAL} check-run exists` };
  if (named.length > 1) {
    return { state: "refused", why: `${named.length} latest ${PASS_SIGNAL} check-runs exist (apps: ${named.map((r) => r.app).join(", ")}), so the verdict is ambiguous` };
  }
  const [run] = named;
  if (run.app !== PASS_APP) return { state: "refused", why: `${PASS_SIGNAL} comes from the ${run.app} app, not ${PASS_APP}` };
  if (run.status !== "completed") return { state: "refused", why: `${PASS_SIGNAL} is ${run.status}` };
  return run.conclusion === "success" ? { state: "pass" } : { state: "refused", why: `${PASS_SIGNAL} concluded ${run.conclusion}` };
}

export function decideMerge(head: HeadState): MergeVerdict {
  if (botExempt(head.repo)) {
    if (head.pinned) return { ok: true, via: "exempt" };
    return { ok: false, reason: `${head.repo} skips the review bot only when the merge names it. Pass -R ${head.repo} or the PR URL.` };
  }
  const verdict = verdictOf(head.runs);
  if (verdict.state === "pass") return { ok: true, via: "approval" };
  return { ok: false, reason: `${verdict.why} on head ${head.sha.slice(0, 7)}. A missing, running, failed, or untrusted verdict does not pass.` };
}
