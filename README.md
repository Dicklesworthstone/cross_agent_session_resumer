# casr

Cross Agent Session Resumer for coding agents: resume a session created in one provider (Claude Code, Codex, Gemini, and more) using a different provider by converting through a canonical session model.

![Rust](https://img.shields.io/badge/Rust-2024%20nightly-orange)
![Status](https://img.shields.io/badge/status-active-green)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-blue)

## TL;DR

**The Problem**: AI coding sessions are siloed by provider. A useful Codex session cannot be resumed directly in Claude Code, and vice versa.

**The Solution**: `casr` discovers a session across installed providers, reads it into a canonical IR, writes a native session file for your target provider, verifies read-back fidelity, and prints the exact resume command.

### Why Use casr?

| Feature | What It Does |
|---|---|
| Cross-provider resume | `casr cc resume <codex-session-id>` and similar conversions in one command |
| Canonical IR | Normalizes provider formats into a common model, then exports back to native format |
| Native-format writers | Produces plausible provider-native session files, not intermediate-only exports |
| Safety-first writes | Atomic temp-then-rename writes, conflict detection, optional `.bak` backup with `--force` |
| Provider auto-detection | Finds which provider owns a session ID without user guesswork |
| Verification step | Re-reads written output to catch writer bugs before you try to resume |
| Machine-friendly output | `--json` mode for scripts and automation |
| Debuggability | `--verbose`, `--trace`, and structured tracing with `RUST_LOG` |

## Quick Example

```bash
# 1) See what providers are available
casr providers

# 2) Find a session from any provider
casr list --limit 20 --sort date

# 3) Inspect a single session
casr info 019c3eae-94c3-7d73-9b2a-9edb18f1563b

# 4) Convert that session to Claude Code format
casr cc resume 019c3eae-94c3-7d73-9b2a-9edb18f1563b

# 5) Resume in Claude Code using the generated ID
claude --resume <new-session-id>
```

## Design Philosophy

1. **Provider fungibility over lock-in**: sessions are portable assets.
2. **Native fidelity over lossy export**: writers target real provider session formats.
3. **Safety over convenience**: atomic writes, conflict checks, read-back verification.
4. **Permissive conversion over brittle strictness**: warnings for imperfect input when conversion is still useful.
5. **Observability by default**: rich logs and actionable errors for every pipeline stage.

## How casr Compares

| Capability | casr | Manual copy/paste | Read-only session search tools | Ad-hoc one-off scripts |
|---|---|---|---|---|
| Convert sessions between providers | Yes | No | No | Partial |
| Provider-native output files | Yes | No | No | Usually brittle |
| Auto-detect source provider by session ID | Yes | No | Sometimes | Rare |
| Atomic writes and conflict handling | Yes | No | N/A | Rare |
| Round-trip testable architecture | Yes | No | N/A | Rare |
| Structured JSON mode for automation | Yes | No | Sometimes | Depends |

## Supported Providers

| Provider | Alias | Read | Write | Resume command |
|---|---|---|---|---|
| Claude Code | `cc` | Yes | Yes | `claude --resume <session-id>` |
| Codex | `cod` | Yes | Yes | `codex resume <session-id>` |
| Gemini CLI | `gmi` | Yes | Yes | `gemini resume <session-id>` |
| Cursor | `cur` | Yes | Yes | Provider-specific |
| Cline | `cln` | Yes | Yes | Provider-specific |
| Aider | `aid` | Yes | Yes | Provider-specific |
| Amp | `amp` | Yes | Yes | Provider-specific |
| OpenCode | `opc` | Yes | Yes | Provider-specific |

Notes:
- Initial core focus is Claude Code, Codex, and Gemini CLI.
- Additional providers are implemented through the same `Provider` trait model.

## Installation

### From Source (Recommended)

```bash
git clone <your-repo-url>
cd cross_agent_sessions_resumer
cargo build --release
./target/release/casr --help
```

### Install as a Cargo Binary

```bash
cargo install --path .
casr --help
```

### Development Mode

```bash
cargo run -- --help
```

## Quick Start

1. Confirm provider detection.
```bash
casr providers
```

2. List discoverable sessions.
```bash
casr list --sort date --limit 50
```

3. Inspect the source session.
```bash
casr info <session-id>
```

4. Convert to your target provider.
```bash
casr <target-alias> resume <session-id>
```

5. Resume in target provider.
```bash
# Examples
claude --resume <new-session-id>
codex resume <new-session-id>
gemini resume <new-session-id>
```

## Commands

Global flags:

```bash
--dry-run                 # Show what would happen without writing
--force                   # Overwrite existing target session (creates .bak backup)
--json                    # Structured JSON output
--verbose                 # Debug-level logging (casr=debug)
--trace                   # Trace-level logging (casr=trace)
--source <alias_or_path>  # Explicit source provider alias or direct session path
--enrich                  # Add optional synthetic context/orientation messages
```

### `casr <target> resume <session-id>`

Convert a source session into target provider format and print the target resume command.

```bash
casr cc resume 019c3eae-94c3-7d73-9b2a-9edb18f1563b
casr cod resume 40f2cb68-fed7-4cee-83de-2b63ba9b7813 --dry-run
casr gmi resume 40f2cb68-fed7-4cee-83de-2b63ba9b7813 --source cc
casr cc resume <session-id> --force
casr cc resume <session-id> --json
```

### `casr list`

List sessions across installed providers.

```bash
casr list
casr list --provider codex
casr list --workspace /data/projects/myapp
casr list --limit 100 --sort messages
```

### `casr info <session-id>`

Show non-converting session details.

```bash
casr info 019c3eae-94c3-7d73-9b2a-9edb18f1563b
casr info 019c3eae-94c3-7d73-9b2a-9edb18f1563b --json
```

### `casr providers`

Show provider detection and installation evidence.

```bash
casr providers
```

### `casr completions <shell>`

Generate shell completions.

```bash
casr completions bash > /tmp/casr.bash
casr completions zsh > "${fpath[1]}/_casr"
casr completions fish > ~/.config/fish/completions/casr.fish
```

## Configuration

`casr` is primarily configured by environment variables.

```bash
# Optional provider home overrides for non-standard locations
export CLAUDE_HOME="$HOME/.claude"
export CODEX_HOME="$HOME/.codex"
export GEMINI_HOME="$HOME/.gemini"

# Logging verbosity (alternative to --verbose / --trace)
export RUST_LOG="casr=debug"
# or:
export RUST_LOG="casr=trace"
```

## Canonical Session Model

Core model (conceptual):

```text
CanonicalSession
  - session_id: String
  - provider_slug: String
  - workspace: Option<PathBuf>
  - title: Option<String>
  - started_at: Option<epoch_millis>
  - ended_at: Option<epoch_millis>
  - messages: Vec<CanonicalMessage>
  - metadata: serde_json::Value
  - source_path: PathBuf
  - model_name: Option<String>

CanonicalMessage
  - idx: usize
  - role: User | Assistant | Tool | System | Other(String)
  - content: String
  - timestamp: Option<epoch_millis>
  - author: Option<String>
  - tool_calls: Vec<ToolCall>
  - tool_results: Vec<ToolResult>
  - extra: serde_json::Value
```

Important helpers:
- `flatten_content`: normalizes mixed string/block content representations.
- `parse_timestamp`: normalizes ISO strings, epoch seconds, and epoch millis.
- `normalize_role`: maps provider-specific roles to canonical roles.
- `reindex_messages`: keeps message indices contiguous after filtering.

## Architecture

```text
Input CLI
  casr <target> resume <session-id>
          |
          v
Provider Registry + Detection
  - discover installed providers
  - optional --source narrowing
          |
          v
Session Discovery
  - find owning provider + source path
          |
          v
Reader (Provider-specific native format -> CanonicalSession)
  Claude/Codex/Gemini/etc.
          |
          v
Validation
  - hard errors: empty / one-sided sessions
  - warnings/info: missing workspace, timestamp gaps, metadata loss
          |
          v
Writer (CanonicalSession -> target native format)
  - generate target session id
  - preserve provider-specific extras when possible
          |
          v
Atomic Write + Conflict Handling
  - temp file -> fsync -> rename
  - optional --force backup (.bak)
          |
          v
Read-Back Verification
  - re-read written session via target reader
  - compare structural fidelity
          |
          v
Output
  - human output with actionable steps
  - optional JSON output for automation
```

## Provider Format Notes

### Claude Code
- Source path pattern: `~/.claude/projects/<project-hash>/<session-id>.jsonl`
- JSONL events: `user`, `assistant`, and other event types (skipped when non-message)
- Writer emits provider-plausible JSONL with expected fields and timestamps.

### Codex
- Source path pattern: `~/.codex/sessions/YYYY/MM/DD/rollout-N.jsonl`
- JSONL events include `session_meta`, `response_item`, and `event_msg` variants.
- Writer emits `session_meta` and response events plus token-count events when available.

### Gemini CLI
- Source path pattern: `~/.gemini/tmp/<hash>/chats/session-<id>.json`
- JSON includes `sessionId`, `projectHash`, `messages`, and temporal fields.
- Writer emits `user` and `model` message types with provider-compatible structure.

## Validation Rules

Hard-stop errors:
- No messages.
- Missing either user or assistant messages.

Warnings (conversion continues):
- Missing workspace.
- Missing timestamps.
- Unusual role ordering.
- Very short sessions.
- High malformed-line skip ratio.

Verbose info:
- Tool-call/result mismatch notes.
- Metadata-loss notes.

## Round-Trip and Fidelity Guarantees

Core invariant for each provider `P`:

```text
read_P(write_P(canonical)) ~= canonical
```

Cross-provider invariant:

```text
read_target(write_target(read_source(input))) preserves
  - message order
  - message role intent
  - message text content
  - timestamps (within normalization tolerance)
```

Known expected differences:
- New target session ID is generated.
- Some provider-specific metadata may not map one-to-one.
- Workspace extraction for some providers may be best-effort.

## Testing

### Unit and Integration

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

### End-to-End

```bash
bash scripts/e2e_test.sh
```

Test suite coverage includes:
- Reader and writer tests for all provider adapters.
- Canonical model helper tests (`flatten_content`, `parse_timestamp`, etc.).
- Conversion pipeline tests with mock providers.
- Cross-provider round-trip fidelity matrix tests.
- CLI integration tests with fixture-backed temp directories.
- Full shell-level e2e conversion paths and error scenarios.

## Troubleshooting

### "Session not found"

```bash
casr list
casr info <session-id>
casr cc resume <session-id> --source cod
```

### "Target provider not installed"

Check provider availability:

```bash
casr providers
```

Install the missing provider, then retry.

### "Session already exists in target"

Use force mode to back up and overwrite:

```bash
casr cc resume <session-id> --force
```

### "Write verification failed"

Run in trace mode and inspect JSON diagnostics:

```bash
casr cc resume <session-id> --trace --json
```

### "Wrong source provider was detected"

Pin source provider or session path explicitly:

```bash
casr cc resume <session-id> --source cod
casr cc resume <session-id> --source ~/.codex/sessions/2026/02/06/rollout-1.jsonl
```

## Limitations

- Provider-specific metadata cannot always be preserved perfectly across all provider pairs.
- Provider internal format changes can require reader/writer updates.
- Some workspace extraction paths are heuristic-based (especially when source format lacks explicit workspace).
- Resume acceptance depends on external provider behavior and may vary by provider version.

## FAQ

### Is casr only for one-way migration?

No. It supports bidirectional conversion across supported providers.

### Does casr modify my source session?

No. It reads source sessions and writes to target provider storage.

### What happens when target session file already exists?

By default it stops with a conflict error. With `--force`, it creates a `.bak` backup and overwrites.

### Can I script casr in CI or automation?

Yes. Use `--json` output and non-interactive command patterns.

### How do I debug a failed conversion?

Use `--verbose` or `--trace`, optionally with `RUST_LOG=casr=trace`.

### Can I convert within the same provider?

Yes. Same-provider conversion is handled gracefully and may return a direct resume path/no-op behavior when appropriate.

## About Contributions

*About Contributions:* Please don't take this the wrong way, but I do not accept outside contributions for any of my projects. I simply don't have the mental bandwidth to review anything, and it's my name on the thing, so I'm responsible for any problems it causes; thus, the risk-reward is highly asymmetric from my perspective. I'd also have to worry about other "stakeholders," which seems unwise for tools I mostly make for myself for free. Feel free to submit issues, and even PRs if you want to illustrate a proposed fix, but know I won't merge them directly. Instead, I'll have Claude or Codex review submissions via `gh` and independently decide whether and how to address them. Bug reports in particular are welcome. Sorry if this offends, but I want to avoid wasted time and hurt feelings. I understand this isn't in sync with the prevailing open-source ethos that seeks community contributions, but it's the only way I can move at this velocity and keep my sanity.

## License

MIT.
