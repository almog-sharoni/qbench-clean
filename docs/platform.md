# Multi-user platform

The platform is for an internal research team on **one trusted host and one
application process**. It adds authentication and application-level isolation;
it is not an operating-system sandbox for hostile tenant code.

## Bootstrap and launch

```bash
python -m pip install -e ".[dashboard]"
qbench-admin init --database .qbench/platform.sqlite3 --username owner
qbench-dashboard --database .qbench/platform.sqlite3 --host 127.0.0.1 --port 8501
```

Choose a private directory on persistent local storage. Bootstrap creates a new
database with mode `0600` and, when needed, its immediate directory with mode
`0700`. Existing databases are never overwritten. Symlinked or group/world-readable
database files and invalid databases are refused before the server starts. Back up a live database with SQLite's
backup mechanism; copying only the main file can omit WAL transactions.

Initialization/recovery prompt privately for passwords instead of accepting them
in command-line arguments. No production account or default password is included.
Passwords use Argon2id (64 MiB, three iterations, parallelism two), not plaintext
or reversible encryption. See the [OWASP password-storage guide](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html).

## Account administration

Open **Administration → Create account**. Choose a username, display name,
temporary password, role, and features. Share the temporary password through a
trusted channel. First sign-in permits only password change and sign-out.

In **Users & feature access**, administrators can enable/deactivate accounts,
assign the `user`/`admin` role, grant features, reset passwords, and revoke sessions.
Saving access settings revokes every session for that user, including your own
if you edit yourself. Password changes/resets also revoke sessions. Reset passwords
must be changed at the next sign-in. Accounts are deactivated rather than deleted,
preserving audit attribution.

At least one enabled administrator must remain. That protection is transactional,
including concurrent edits. The role grants administration, while feature flags
control workbench actions even for administrators.

## Feature dependencies

| Feature | Requires | Grants |
| --- | --- | --- |
| `inspect` | — | Demo inspection and private support reports |
| `convert` | inspect | Simulator construction and verification |
| `evaluate` | convert | Paired output comparisons |
| `downloads` | inspect | Support/plan/ledger/evaluation ZIP exports |
| `vision` | inspect | Approved torchvision/timm catalog |
| `pretrained` | vision | Library-managed pretrained weight downloads |
| `gpu` | inspect | GPU model/input placement |
| `quantization` | gpu | Quantization-enabled runs |
| `detailed` | evaluate | Detailed diagnostics; no retained activation samples |
| `fallback` | inspect | Explicitly partial FP32 fallback |
| `datasets` | vision, evaluate | Approved ImageNet-folder evaluation |

New users default to inspect, convert, evaluate, and downloads. Dependencies must
be selected together. Hidden/disabled controls are only presentation: the server
rechecks the live session and permissions for each operation. A long-running job
finishing after revocation does not deliver its result to the revoked session.

## Approved models and datasets

The approved models are the tiny demo, ResNet-18, ViT-B/16, and MobileViT-S.
Adding model code is a maintainer change, not a user upload. Arbitrary Python
providers, checkpoints and filesystem paths are excluded from shared mode.

For labeled evaluation, a host operator creates a private JSON catalog:

```json
{
  "imagenet-validation": "/srv/qbench/datasets/imagenet/val"
}
```

Then set its path before launching:

```bash
export QBENCH_DATASET_CATALOG=/srv/qbench/config/datasets.json
qbench-dashboard --database /srv/qbench/private/platform.sqlite3
```

Users select catalog IDs, not paths. IDs are simple names; roots must be absolute
existing directories. Make datasets read-only to the application account where
practical. The platform accepts canonical ImageNet WordNet-ID class folders and
remaps labels to the packaged ImageNet class index. Preprocessing uses model-specific
normalization and the chosen image size. Runs use seed 7 and at most 64 samples.
These small runs are convenience checks, not the full pretrained acceptance gate.

Without a labeled dataset, evaluation uses two synthetic batches and reports
output comparisons—not task accuracy. The UI identifies the data behind the
displayed results. Downloaded ZIPs exclude model state and retained activations.

## Sessions and resource isolation

- Random opaque session tokens remain in server-side Streamlit session state;
  SQLite stores only their SHA-256 digests. Tokens do not appear in query strings.
- Sessions expire after eight hours or 30 minutes of inactivity. Browser refresh
  starts a new session and requires sign-in again.
- Five failed sign-ins per normalized account name within 15 minutes trigger a
  persistent throttle. Errors do not disclose unknown or disabled accounts.
- Each browser session owns its model, simulator, reports and ZIP. Another user,
  or even a second session for the same user, cannot operate on that workspace.
- One model operation runs at a time to protect process-global PyTorch/RNG state.
  Concurrent attempts receive a busy message rather than waiting in an unbounded queue.
- Four loaded workspaces are allowed by default. Set `QBENCH_MAX_WORKSPACES` to
  1–32 according to host memory. Release unused models. Expired/revoked workspaces
  are reclaimed on the next inspection.
- Batch size is capped at four and image sizes at 224/256. There is no distributed
  scheduler, background queue, hard timeout or forced kernel cancellation.
  An in-flight kernel may finish after revocation, but its result is withheld.

The compute lock/capacity are process-local: do not run multiple replicas against
the same database expecting distributed resource coordination.

## Usage metrics and privacy

Admins see account counts/status, last sign-in, per-user action counts/outcomes,
total elapsed time and mean elapsed time over 7/30/90 days. The latest 200 events
are displayed; aggregates include the whole selected window. Users see only their
own actor-scoped activity.

Audit data includes authentication/account changes and model operation outcomes,
not passwords, session tokens, tensor values, dataset paths or raw exception
messages. Retention is 90 days, pruned on sign-in. Timing is operation wall time,
not GPU utilization or model-only throughput. SQLite is not encrypted by QBench;
host/database administrators can access its contents.

Only grant downloads to users permitted to receive model artifacts. Already-delivered
bytes and Streamlit media responses cannot be recalled by revoking a session;
permissions apply to future server actions. Detailed summary statistics can be
sensitive even without retained activation tensors.

## Host-only recovery

Keep a second administrator where practical. An operator with filesystem access
to the private database can recover an enabled administrator:

```bash
qbench-admin recover-admin --database .qbench/platform.sqlite3 --username owner
```

Recovery prompts for a temporary password, revokes sessions, clears the sign-in
throttle, and requires another password change. It does not promote users or
enable disabled accounts and is not exposed in the web UI.

## Remote deployment checklist

The launcher binds loopback by default, keeps Streamlit CORS/XSRF protection
enabled, and suppresses browser-visible exception details. Streamlit 1.55 or newer
is required. Before sharing authenticated mode beyond the host:

1. Terminate **HTTPS** at a trusted reverse proxy with WebSocket support.
2. Restrict network access and add proxy-level connection/IP rate limits. Account
   throttling alone is not a comprehensive internet-abuse defense.
3. Use an unprivileged dedicated OS account, a private persistent database directory,
   a writable extension cache, and only the approved read-only dataset mounts.
4. Run one application process, set conservative model capacity, and monitor host
   memory/disk/GPU resources independently.
5. Back up the database, pin deployment dependencies, and test recovery.
6. Never proxy `--single-user` or set `QBENCH_SINGLE_USER=1` on a shared service.
   Do not disable CORS/XSRF protection.

SSO/OIDC, MFA, email invitation/reset delivery, hostile-code sandboxing,
distributed workers and penetration testing are **not included**. Authentication
does not replace authorization; see Streamlit's [authentication documentation](https://docs.streamlit.io/develop/api-reference/user).
The platform enforces its own account and feature policies at the action boundary.
