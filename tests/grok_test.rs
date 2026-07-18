//! Integration tests for the Grok Build provider.
//!
//! These exercise discovery, reading, and CLI behavior against a temporary
//! `$GROK_HOME` seeded from the fixture tree. They live here rather than in
//! the in-crate `#[cfg(test)]` module because `src/lib.rs` declares
//! `#![forbid(unsafe_code)]` and `std::env::set_var` is `unsafe` in edition
//! 2024 — the shared `EnvGuard`/`EnvLock` harness (see `tests/test_env.rs`)
//! serializes process-global env mutation here, in a separate crate.

mod test_env;

use std::path::{Path, PathBuf};

use casr::discovery::ProviderRegistry;
use casr::model::MessageRole;
use casr::providers::{Provider, WriteOptions, grok::Grok};

static GROK_ENV: test_env::EnvLock = test_env::EnvLock;

struct EnvGuard {
    key: &'static str,
    original: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &Path) -> Self {
        let original = std::env::var(key).ok();
        // SAFETY: callers hold the `GROK_ENV` lock for the duration, so no
        // other thread reads or mutates the environment concurrently.
        unsafe { std::env::set_var(key, value) };
        Self { key, original }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.original {
            Some(val) => unsafe { std::env::set_var(self.key, val) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

const FIXTURE_ID: &str = "019f75d0-aaaa-7bbb-8ccc-b0a1b2c3d4e5";
const ENCODED_CWD: &str = "%2Fdata%2Fprojects%2Fdemo";
const FIXTURE_WORKSPACE: &str = "/data/projects/demo";

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/grok")
}

/// Copy the fixture session tree into `$GROK_HOME/sessions/…`, plus the
/// non-session artifacts a real `~/.grok/sessions` tree contains
/// (`session_search.sqlite`, group-level `prompt_history.jsonl`).
fn seed_grok_home(home: &Path) {
    let dst = home.join("sessions").join(ENCODED_CWD).join(FIXTURE_ID);
    std::fs::create_dir_all(&dst).unwrap();
    let src = fixtures_dir()
        .join("sessions")
        .join(ENCODED_CWD)
        .join(FIXTURE_ID);
    for name in ["updates.jsonl", "summary.json"] {
        std::fs::copy(src.join(name), dst.join(name)).unwrap();
    }
    std::fs::write(
        home.join("sessions").join("session_search.sqlite"),
        b"SQLite format 3\x00fixture",
    )
    .unwrap();
    std::fs::write(
        home.join("sessions").join(ENCODED_CWD).join("prompt_history.jsonl"),
        format!(
            "{{\"timestamp\":\"2026-07-18T15:20:56.242466991Z\",\"session_id\":\"{FIXTURE_ID}\",\"prompt\":\"Run the shell command: echo hi .\",\"is_bash\":false}}\n"
        ),
    )
    .unwrap();
}

#[test]
fn discovery_lists_and_owns_seeded_session() {
    let _lock = GROK_ENV.lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let _env = EnvGuard::set("GROK_HOME", tmp.path());
    seed_grok_home(tmp.path());

    let listed = Grok.list_sessions().expect("list_sessions");
    assert_eq!(listed.len(), 1, "exactly one seeded session: {listed:?}");
    assert_eq!(listed[0].0, FIXTURE_ID);

    let owned = Grok.owns_session(FIXTURE_ID).expect("owns seeded session");
    assert!(owned.ends_with("updates.jsonl"));

    // Case-insensitive ownership lookup.
    let upper = FIXTURE_ID.to_ascii_uppercase();
    assert!(Grok.owns_session(&upper).is_some());

    // The registry resolves the `grk` alias (and slug tokens) to Grok.
    let registry = ProviderRegistry::default_registry();
    let provider = registry.find_by_alias("grk").expect("grk alias resolves");
    assert_eq!(provider.slug(), "grok");
    let provider = registry
        .find_by_alias("grok-build")
        .expect("grok-build token resolves");
    assert_eq!(provider.slug(), "grok");
}

#[test]
fn read_session_from_seeded_home_matches_fixture_expectations() {
    let _lock = GROK_ENV.lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let _env = EnvGuard::set("GROK_HOME", tmp.path());
    seed_grok_home(tmp.path());

    let path = Grok.owns_session(FIXTURE_ID).expect("owns");
    let session = Grok.read_session(&path).expect("read");

    assert_eq!(session.session_id, FIXTURE_ID);
    assert_eq!(session.provider_slug, "grok");
    assert_eq!(session.workspace, Some(PathBuf::from(FIXTURE_WORKSPACE)));
    assert_eq!(session.model_name.as_deref(), Some("grok-build"));
    assert_eq!(session.messages.len(), 5);
    assert_eq!(session.messages[0].role, MessageRole::User);
    assert!(session.messages.iter().any(|m| !m.tool_calls.is_empty()));
    assert!(session.messages.iter().any(|m| !m.tool_results.is_empty()));
}

#[test]
fn write_session_is_refused_read_only_provider() {
    use casr::model::{CanonicalMessage, CanonicalSession};

    let _lock = GROK_ENV.lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let _env = EnvGuard::set("GROK_HOME", tmp.path());

    let session = CanonicalSession {
        session_id: "foreign".into(),
        provider_slug: "claude-code".into(),
        workspace: Some(PathBuf::from("/data/projects/foo")),
        title: Some("Hello".into()),
        started_at: Some(1_700_000_000_000),
        ended_at: Some(1_700_000_100_000),
        messages: vec![CanonicalMessage {
            idx: 0,
            role: MessageRole::User,
            content: "Hi there".into(),
            timestamp: Some(1_700_000_000_000),
            author: Some("user".into()),
            tool_calls: vec![],
            tool_results: vec![],
            extra: serde_json::Value::Null,
        }],
        metadata: serde_json::Value::Object(serde_json::Map::new()),
        source_path: PathBuf::from("/nonexistent"),
        model_name: None,
    };

    let err = Grok
        .write_session(&session, &WriteOptions { force: true })
        .expect_err("grok must refuse writes");
    let msg = err.to_string();
    assert!(msg.contains("read/resume-only"), "unhelpful error: {msg}");
    // Nothing may have been written into GROK_HOME.
    assert!(
        !tmp.path().join("sessions").exists(),
        "refused write must not create session dirs"
    );
}

/// CLI smoke test: `casr list --provider grok` finds the seeded session.
#[test]
fn cli_list_finds_seeded_grok_session() {
    let _lock = GROK_ENV.lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    seed_grok_home(tmp.path());

    // `casr list` defaults to scoping by the current working-directory
    // project; the fixture's workspace is a synthetic path, so pass it
    // explicitly via `--workspace` to take it out of cwd scope.
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_casr"))
        .args([
            "list",
            "--provider",
            "grok",
            "--workspace",
            FIXTURE_WORKSPACE,
            "--limit",
            "5",
        ])
        .env("GROK_HOME", tmp.path())
        .output()
        .expect("run casr list");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "casr list failed: status={:?}\nstdout={stdout}\nstderr={stderr}",
        output.status
    );
    assert!(
        stdout.contains(FIXTURE_ID) || stdout.contains("grok"),
        "expected the seeded Grok session in output:\n{stdout}"
    );
}

/// CLI smoke test: `casr info <id> --source grk` reports the session details.
#[test]
fn cli_info_reports_seeded_grok_session() {
    let _lock = GROK_ENV.lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    seed_grok_home(tmp.path());

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_casr"))
        .args(["info", FIXTURE_ID, "--source", "grk"])
        .env("GROK_HOME", tmp.path())
        .output()
        .expect("run casr info");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "casr info failed: status={:?}\nstdout={stdout}\nstderr={stderr}",
        output.status
    );
    assert!(
        stdout.contains(FIXTURE_ID),
        "expected session id in info output:\n{stdout}"
    );
}

/// CLI smoke test: converting INTO grok fails with the read-only message.
#[test]
fn cli_convert_into_grok_is_refused() {
    let _lock = GROK_ENV.lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    seed_grok_home(tmp.path());

    // Seed a Claude Code session as the cross-provider source (same-provider
    // conversions short-circuit without touching the writer).
    let cc_fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/claude_code/cc_simple.jsonl");
    let cc_dir = tmp.path().join("claude/projects/-data-projects-myapp");
    std::fs::create_dir_all(&cc_dir).unwrap();
    std::fs::copy(&cc_fixture, cc_dir.join("cc-simple-001.jsonl")).unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_casr"))
        .args(["resume", "grok", "cc-simple-001", "--source", "cc"])
        .env("GROK_HOME", tmp.path())
        .env("CLAUDE_HOME", tmp.path().join("claude"))
        .output()
        .expect("run casr resume grok");

    assert!(
        !output.status.success(),
        "converting into grok must fail (read-only provider)"
    );
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        combined.contains("read/resume-only"),
        "expected the read-only refusal message, got:\n{combined}"
    );
}
