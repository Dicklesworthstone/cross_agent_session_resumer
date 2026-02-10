//! End-to-end CLI integration tests for casr.
//!
//! Uses `assert_cmd` to invoke the compiled binary and validate output.
//! All tests use temp directories with env overrides (`CLAUDE_HOME`,
//! `CODEX_HOME`, `GEMINI_HOME`, `CURSOR_HOME`, `CLINE_HOME`, `AIDER_HOME`,
//! `AMP_HOME`, `OPENCODE_HOME`, `CHATGPT_HOME`, `CLAWDBOT_HOME`, `VIBE_HOME`,
//! `FACTORY_HOME`) so they never touch real provider data.

use std::fs;
use std::path::PathBuf;

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::TempDir;

/// Root of the fixtures directory.
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Build a `Command` for the casr binary with isolated provider homes.
///
/// Sets provider home overrides to subdirs of the provided temp dir so the
/// CLI never touches real provider data.
fn casr_cmd(tmp: &TempDir) -> Command {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("casr").expect("casr binary should be built");
    cmd.env("CLAUDE_HOME", tmp.path().join("claude"))
        .env("CODEX_HOME", tmp.path().join("codex"))
        .env("GEMINI_HOME", tmp.path().join("gemini"))
        .env("CURSOR_HOME", tmp.path().join("cursor"))
        .env("CLINE_HOME", tmp.path().join("cline"))
        .env("AIDER_HOME", tmp.path().join("aider"))
        .env("AMP_HOME", tmp.path().join("amp"))
        .env("OPENCODE_HOME", tmp.path().join("opencode"))
        .env("CHATGPT_HOME", tmp.path().join("chatgpt"))
        .env("CLAWDBOT_HOME", tmp.path().join("clawdbot"))
        .env("VIBE_HOME", tmp.path().join("vibe"))
        .env("FACTORY_HOME", tmp.path().join("factory"))
        .env("OPENCLAW_HOME", tmp.path().join("openclaw"))
        .env("PI_AGENT_HOME", tmp.path().join("pi-agent"))
        .env("XDG_CONFIG_HOME", tmp.path().join("xdg-config"))
        .env("XDG_DATA_HOME", tmp.path().join("xdg-data"))
        // Suppress colored output in tests.
        .env("NO_COLOR", "1");
    cmd
}

/// Set up a Claude Code session fixture in the temp dir.
///
/// Creates the expected directory structure:
/// `<claude_home>/projects/<project-key>/<session-id>.jsonl`
fn setup_cc_fixture(tmp: &TempDir, fixture_name: &str) -> String {
    let source = fixtures_dir().join(format!("claude_code/{fixture_name}.jsonl"));
    let content = fs::read_to_string(&source)
        .unwrap_or_else(|e| panic!("Failed to read fixture {fixture_name}: {e}"));

    // Extract session ID and cwd from the fixture content.
    let first_line: serde_json::Value = content
        .lines()
        .find(|l| !l.trim().is_empty())
        .and_then(|l| serde_json::from_str(l).ok())
        .expect("fixture should have valid first line");

    let session_id = first_line["sessionId"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();
    let cwd = first_line["cwd"].as_str().unwrap_or("/tmp");

    // Derive project key: replace non-alphanumeric with dash.
    let project_key: String = cwd
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect();

    let projects_dir = tmp.path().join("claude/projects").join(&project_key);
    fs::create_dir_all(&projects_dir).expect("create CC project dir");

    let target_path = projects_dir.join(format!("{session_id}.jsonl"));
    fs::write(&target_path, &content).expect("write CC fixture");

    session_id
}

/// Set up a Codex session fixture in the temp dir.
#[allow(dead_code)]
fn setup_codex_fixture(tmp: &TempDir, fixture_name: &str, ext: &str) -> String {
    let source = fixtures_dir().join(format!("codex/{fixture_name}.{ext}"));
    let content = fs::read_to_string(&source)
        .unwrap_or_else(|e| panic!("Failed to read fixture {fixture_name}: {e}"));

    // For JSONL, extract session ID from session_meta payload.
    let session_id = if ext == "jsonl" {
        content
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .find(|v| v["type"] == "session_meta")
            .and_then(|v| v["payload"]["id"].as_str().map(String::from))
            .unwrap_or_else(|| "unknown".to_string())
    } else {
        // Legacy JSON.
        let root: serde_json::Value = serde_json::from_str(&content).unwrap();
        root["session"]["id"]
            .as_str()
            .unwrap_or("unknown")
            .to_string()
    };

    // Place in sessions dir with correct hierarchy.
    let sessions_dir = tmp.path().join("codex/sessions/2026/01/01");
    fs::create_dir_all(&sessions_dir).expect("create Codex sessions dir");

    let filename = format!("rollout-2026-01-01T00-00-00-{session_id}.{ext}");
    let target_path = sessions_dir.join(&filename);
    fs::write(&target_path, &content).expect("write Codex fixture");

    session_id
}

/// Set up a Gemini session fixture in the temp dir.
#[allow(dead_code)]
fn setup_gemini_fixture(tmp: &TempDir, fixture_name: &str) -> String {
    let source = fixtures_dir().join(format!("gemini/{fixture_name}.json"));
    let content = fs::read_to_string(&source)
        .unwrap_or_else(|e| panic!("Failed to read fixture {fixture_name}: {e}"));

    let root: serde_json::Value = serde_json::from_str(&content).unwrap();
    let session_id = root["sessionId"].as_str().unwrap_or("unknown").to_string();

    // Place in <hash>/chats/ directory.
    let hash_dir = tmp.path().join("gemini/tmp/testhash123/chats");
    fs::create_dir_all(&hash_dir).expect("create Gemini chats dir");

    let filename = format!("session-{session_id}.json");
    let target_path = hash_dir.join(&filename);
    fs::write(&target_path, &content).expect("write Gemini fixture");

    session_id
}

// ---------------------------------------------------------------------------
// Basic CLI tests
// ---------------------------------------------------------------------------

#[test]
fn cli_version_outputs_metadata() {
    let tmp = TempDir::new().unwrap();
    casr_cmd(&tmp)
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("casr"));
}

#[test]
fn cli_help_outputs_usage() {
    let tmp = TempDir::new().unwrap();
    casr_cmd(&tmp)
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Cross Agent Session Resumer"))
        .stdout(predicate::str::contains("resume"))
        .stdout(predicate::str::contains("list"))
        .stdout(predicate::str::contains("providers"));
}

#[test]
fn cli_no_args_shows_error() {
    let tmp = TempDir::new().unwrap();
    casr_cmd(&tmp)
        .assert()
        .failure()
        .stderr(predicate::str::contains("Usage"));
}

#[test]
fn cli_invalid_subcommand_fails() {
    let tmp = TempDir::new().unwrap();
    casr_cmd(&tmp).arg("nonexistent").assert().failure();
}

// ---------------------------------------------------------------------------
// Providers command
// ---------------------------------------------------------------------------

#[test]
fn cli_providers_succeeds() {
    let tmp = TempDir::new().unwrap();
    casr_cmd(&tmp)
        .arg("providers")
        .assert()
        .success()
        .stdout(predicate::str::contains("Claude Code"))
        .stdout(predicate::str::contains("Codex"))
        .stdout(predicate::str::contains("Gemini"))
        .stdout(predicate::str::contains("Cursor"))
        .stdout(predicate::str::contains("Cline"))
        .stdout(predicate::str::contains("Aider"))
        .stdout(predicate::str::contains("Amp"))
        .stdout(predicate::str::contains("OpenCode"));
}

#[test]
fn cli_providers_json_is_valid() {
    let tmp = TempDir::new().unwrap();
    let output = casr_cmd(&tmp)
        .args(["--json", "providers"])
        .output()
        .expect("providers should run");

    assert!(output.status.success(), "providers --json should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("providers --json should emit valid JSON");
    assert!(parsed.is_array(), "providers JSON should be an array");
}

// ---------------------------------------------------------------------------
// List command
// ---------------------------------------------------------------------------

#[test]
fn cli_list_empty_shows_helpful_message() {
    let tmp = TempDir::new().unwrap();
    casr_cmd(&tmp)
        .arg("list")
        .assert()
        .success()
        .stdout(predicate::str::contains("No sessions found"));
}

#[test]
fn cli_list_finds_cc_sessions() {
    let tmp = TempDir::new().unwrap();
    let session_id = setup_cc_fixture(&tmp, "cc_simple");
    casr_cmd(&tmp)
        .arg("list")
        .assert()
        .success()
        .stdout(predicate::str::contains(&session_id));
}

#[test]
fn cli_list_json_is_valid_array() {
    let tmp = TempDir::new().unwrap();
    setup_cc_fixture(&tmp, "cc_simple");
    let output = casr_cmd(&tmp)
        .args(["--json", "list"])
        .output()
        .expect("list should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("list --json should emit valid JSON");
    assert!(parsed.is_array());
    assert!(!parsed.as_array().unwrap().is_empty());
}

#[test]
fn cli_list_limit_respects_bound() {
    let tmp = TempDir::new().unwrap();
    setup_cc_fixture(&tmp, "cc_simple");
    setup_cc_fixture(&tmp, "cc_malformed");
    let output = casr_cmd(&tmp)
        .args(["--json", "list", "--limit", "1"])
        .output()
        .expect("list should run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(parsed.as_array().unwrap().len(), 1);
}

#[test]
fn cli_list_workspace_filter_filters_sessions() {
    let tmp = TempDir::new().unwrap();
    let myapp_id = setup_cc_fixture(&tmp, "cc_simple"); // /data/projects/myapp
    let webapp_id = setup_cc_fixture(&tmp, "cc_complex"); // /data/projects/webapp

    let output = casr_cmd(&tmp)
        .args(["--json", "list", "--workspace", "/data/projects/myapp"])
        .output()
        .expect("list should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let sessions = parsed.as_array().expect("list --json should be an array");

    assert!(
        sessions
            .iter()
            .any(|s| s["session_id"].as_str() == Some(&myapp_id)),
        "expected myapp session to be present"
    );
    assert!(
        !sessions
            .iter()
            .any(|s| s["session_id"].as_str() == Some(&webapp_id)),
        "expected webapp session to be filtered out"
    );
}

#[test]
fn cli_list_sort_messages_orders_descending() {
    let tmp = TempDir::new().unwrap();
    let simple_id = setup_cc_fixture(&tmp, "cc_simple");
    let complex_id = setup_cc_fixture(&tmp, "cc_complex");

    let output = casr_cmd(&tmp)
        .args(["--json", "list", "--sort", "messages", "--limit", "2"])
        .output()
        .expect("list should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let sessions = parsed.as_array().expect("list --json should be an array");

    assert_eq!(sessions.len(), 2);
    assert_eq!(
        sessions[0]["session_id"].as_str(),
        Some(complex_id.as_str())
    );
    assert_eq!(sessions[1]["session_id"].as_str(), Some(simple_id.as_str()));
}

// ---------------------------------------------------------------------------
// Info command
// ---------------------------------------------------------------------------

#[test]
fn cli_info_shows_session_details() {
    let tmp = TempDir::new().unwrap();
    let session_id = setup_cc_fixture(&tmp, "cc_simple");
    casr_cmd(&tmp)
        .args(["info", &session_id])
        .assert()
        .success()
        .stdout(predicate::str::contains(&session_id))
        .stdout(predicate::str::contains("claude-code"))
        .stdout(predicate::str::contains("Messages:"));
}

#[test]
fn cli_info_json_is_valid() {
    let tmp = TempDir::new().unwrap();
    let session_id = setup_cc_fixture(&tmp, "cc_simple");
    let output = casr_cmd(&tmp)
        .args(["--json", "info", &session_id])
        .output()
        .expect("info should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("info --json should emit valid JSON");
    assert_eq!(parsed["session_id"].as_str().unwrap(), session_id);
    assert_eq!(parsed["provider"].as_str().unwrap(), "claude-code");
}

#[test]
fn cli_info_unknown_session_fails() {
    let tmp = TempDir::new().unwrap();
    casr_cmd(&tmp)
        .args(["info", "nonexistent-session-id"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Error"));
}

#[test]
fn cli_info_unknown_session_json_error() {
    let tmp = TempDir::new().unwrap();
    let output = casr_cmd(&tmp)
        .args(["--json", "info", "nonexistent-session-id"])
        .output()
        .expect("info should run");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    let parsed: serde_json::Value =
        serde_json::from_str(&stderr).expect("JSON error should be valid JSON");
    assert_eq!(parsed["ok"], false);
    assert!(parsed["error_type"].as_str().is_some());
}

// ---------------------------------------------------------------------------
// Resume command
// ---------------------------------------------------------------------------

#[test]
fn cli_resume_dry_run_does_not_write() {
    let tmp = TempDir::new().unwrap();
    let session_id = setup_cc_fixture(&tmp, "cc_simple");

    // Resume CC→Codex with dry run.
    casr_cmd(&tmp)
        .args(["resume", "cod", &session_id, "--dry-run"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Would convert"));

    // Verify no Codex session files were written.
    let codex_sessions = tmp.path().join("codex/sessions");
    if codex_sessions.exists() {
        let entries: Vec<_> = walkdir::WalkDir::new(&codex_sessions)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
            .collect();
        assert!(
            entries.is_empty(),
            "Dry run should not write any files, but found: {:?}",
            entries
        );
    }
}

#[test]
fn cli_resume_writes_target_session() {
    let tmp = TempDir::new().unwrap();
    let session_id = setup_cc_fixture(&tmp, "cc_simple");

    // Resume CC→Codex (actual write).
    casr_cmd(&tmp)
        .args(["resume", "cod", &session_id])
        .assert()
        .success()
        .stdout(predicate::str::contains("Converted"))
        .stdout(predicate::str::contains("claude-code"))
        .stdout(predicate::str::contains("codex"))
        .stdout(predicate::str::contains("Resume:"));

    // Verify a Codex session file was written.
    let codex_sessions = tmp.path().join("codex/sessions");
    assert!(
        codex_sessions.exists(),
        "Codex sessions dir should exist after conversion"
    );
    let files: Vec<_> = walkdir::WalkDir::new(&codex_sessions)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .collect();
    assert_eq!(
        files.len(),
        1,
        "Exactly one Codex session file should be written"
    );
}

#[test]
fn cli_resume_json_output_is_valid() {
    let tmp = TempDir::new().unwrap();
    let session_id = setup_cc_fixture(&tmp, "cc_simple");

    let output = casr_cmd(&tmp)
        .args(["--json", "resume", "cod", &session_id, "--dry-run"])
        .output()
        .expect("resume should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("resume --json should emit valid JSON");
    assert_eq!(parsed["ok"], true);
    assert_eq!(parsed["source_provider"].as_str().unwrap(), "claude-code");
    assert_eq!(parsed["target_provider"].as_str().unwrap(), "codex");
    assert_eq!(parsed["dry_run"], true);
}

#[test]
fn cli_resume_unknown_target_fails() {
    let tmp = TempDir::new().unwrap();
    let session_id = setup_cc_fixture(&tmp, "cc_simple");

    casr_cmd(&tmp)
        .args(["resume", "nonexistent", &session_id])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Error"));
}

#[test]
fn cli_resume_unknown_session_fails() {
    let tmp = TempDir::new().unwrap();
    casr_cmd(&tmp)
        .args(["resume", "cod", "nonexistent-session"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Error"));
}

#[test]
fn cli_resume_cc_to_gemini_works() {
    let tmp = TempDir::new().unwrap();
    let session_id = setup_cc_fixture(&tmp, "cc_simple");

    casr_cmd(&tmp)
        .args(["resume", "gmi", &session_id])
        .assert()
        .success()
        .stdout(predicate::str::contains("Converted"))
        .stdout(predicate::str::contains("gemini"));

    // Verify Gemini file was written.
    let gemini_tmp = tmp.path().join("gemini/tmp");
    assert!(gemini_tmp.exists());
    let files: Vec<_> = walkdir::WalkDir::new(&gemini_tmp)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| {
            e.file_type().is_file() && e.path().extension().is_some_and(|ext| ext == "json")
        })
        .collect();
    assert_eq!(
        files.len(),
        1,
        "Exactly one Gemini session file should be written"
    );
}

#[test]
fn cli_resume_cc_to_cursor_works_and_is_discoverable() {
    let tmp = TempDir::new().unwrap();
    let session_id = setup_cc_fixture(&tmp, "cc_simple");

    let output = casr_cmd(&tmp)
        .args(["--json", "resume", "cur", &session_id])
        .output()
        .expect("resume should run");
    assert!(
        output.status.success(),
        "CC→Cursor conversion should succeed"
    );

    let parsed: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("resume --json output should parse");
    assert_eq!(parsed["ok"], true);
    assert_eq!(parsed["target_provider"].as_str().unwrap(), "cursor");
    let cursor_session_id = parsed["target_session_id"]
        .as_str()
        .expect("target_session_id should be present for non-dry-run");

    let cursor_db = tmp.path().join("cursor/User/globalStorage/state.vscdb");
    assert!(
        cursor_db.exists(),
        "Cursor DB should exist after CC→Cursor conversion"
    );

    casr_cmd(&tmp)
        .args(["--json", "info", cursor_session_id])
        .assert()
        .success();
}

#[test]
fn cli_resume_cursor_to_cc_works_with_source_hint() {
    let tmp = TempDir::new().unwrap();
    let source_id = setup_cc_fixture(&tmp, "cc_simple");

    let cursor_result = casr_cmd(&tmp)
        .args(["--json", "resume", "cur", &source_id])
        .output()
        .expect("CC→Cursor seed conversion should run");
    assert!(cursor_result.status.success());
    let cursor_json: serde_json::Value =
        serde_json::from_slice(&cursor_result.stdout).expect("seed conversion JSON should parse");
    let cursor_session_id = cursor_json["target_session_id"]
        .as_str()
        .expect("cursor target_session_id should be present");

    casr_cmd(&tmp)
        .args(["resume", "cc", cursor_session_id, "--source", "cur"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Converted"))
        .stdout(predicate::str::contains("cursor"))
        .stdout(predicate::str::contains("claude-code"));
}

#[test]
fn cli_resume_cc_to_cline_works_and_is_discoverable() {
    let tmp = TempDir::new().unwrap();
    let session_id = setup_cc_fixture(&tmp, "cc_simple");

    let output = casr_cmd(&tmp)
        .args(["--json", "resume", "cln", &session_id])
        .output()
        .expect("resume should run");
    assert!(
        output.status.success(),
        "CC→Cline conversion should succeed"
    );

    let parsed: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("resume --json output should parse");
    assert_eq!(parsed["ok"], true);
    assert_eq!(parsed["target_provider"].as_str().unwrap(), "cline");
    let cline_session_id = parsed["target_session_id"]
        .as_str()
        .expect("target_session_id should be present for non-dry-run");

    let cline_api = tmp
        .path()
        .join("cline/tasks")
        .join(cline_session_id)
        .join("api_conversation_history.json");
    assert!(
        cline_api.exists(),
        "Cline task API history should exist after CC→Cline conversion"
    );

    casr_cmd(&tmp)
        .args([
            "--json",
            "resume",
            "cc",
            cline_session_id,
            "--source",
            "cln",
            "--dry-run",
        ])
        .assert()
        .success();
}

#[test]
fn cli_resume_cline_to_cc_works_with_source_hint() {
    let tmp = TempDir::new().unwrap();
    let source_id = setup_cc_fixture(&tmp, "cc_simple");

    let cline_result = casr_cmd(&tmp)
        .args(["--json", "resume", "cln", &source_id])
        .output()
        .expect("CC→Cline seed conversion should run");
    assert!(cline_result.status.success());
    let cline_json: serde_json::Value =
        serde_json::from_slice(&cline_result.stdout).expect("seed conversion JSON should parse");
    let cline_session_id = cline_json["target_session_id"]
        .as_str()
        .expect("cline target_session_id should be present");

    casr_cmd(&tmp)
        .args(["resume", "cc", cline_session_id, "--source", "cln"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Converted"))
        .stdout(predicate::str::contains("cline"))
        .stdout(predicate::str::contains("claude-code"));
}

#[test]
fn cli_resume_cc_to_amp_works_and_is_discoverable() {
    let tmp = TempDir::new().unwrap();
    let session_id = setup_cc_fixture(&tmp, "cc_simple");

    let output = casr_cmd(&tmp)
        .args(["--json", "resume", "amp", &session_id])
        .output()
        .expect("resume should run");
    assert!(output.status.success(), "CC→Amp conversion should succeed");

    let parsed: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("resume --json output should parse");
    assert_eq!(parsed["ok"], true);
    assert_eq!(parsed["target_provider"].as_str().unwrap(), "amp");
    let amp_session_id = parsed["target_session_id"]
        .as_str()
        .expect("target_session_id should be present for non-dry-run");

    let amp_thread = tmp
        .path()
        .join("amp/threads")
        .join(format!("{amp_session_id}.json"));
    assert!(
        amp_thread.exists(),
        "Amp thread file should exist after CC→Amp conversion"
    );

    casr_cmd(&tmp)
        .args([
            "--json",
            "resume",
            "cc",
            amp_session_id,
            "--source",
            "amp",
            "--dry-run",
        ])
        .assert()
        .success();
}

#[test]
fn cli_resume_amp_to_cc_works_with_source_hint() {
    let tmp = TempDir::new().unwrap();
    let source_id = setup_cc_fixture(&tmp, "cc_simple");

    let amp_result = casr_cmd(&tmp)
        .args(["--json", "resume", "amp", &source_id])
        .output()
        .expect("CC→Amp seed conversion should run");
    assert!(amp_result.status.success());
    let amp_json: serde_json::Value =
        serde_json::from_slice(&amp_result.stdout).expect("seed conversion JSON should parse");
    let amp_session_id = amp_json["target_session_id"]
        .as_str()
        .expect("amp target_session_id should be present");

    casr_cmd(&tmp)
        .args(["resume", "cc", amp_session_id, "--source", "amp"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Converted"))
        .stdout(predicate::str::contains("amp"))
        .stdout(predicate::str::contains("claude-code"));
}

#[test]
fn cli_resume_cc_to_aider_works_and_is_discoverable() {
    let tmp = TempDir::new().unwrap();
    let session_id = setup_cc_fixture(&tmp, "cc_simple");

    let output = casr_cmd(&tmp)
        .args(["--json", "resume", "aid", &session_id])
        .output()
        .expect("resume should run");
    assert!(
        output.status.success(),
        "CC→Aider conversion should succeed"
    );

    let parsed: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("resume --json output should parse");
    assert_eq!(parsed["ok"], true);
    assert_eq!(parsed["target_provider"].as_str().unwrap(), "aider");
    let aider_session_id = parsed["target_session_id"]
        .as_str()
        .expect("target_session_id should be present for non-dry-run");

    let aider_history = tmp.path().join("aider/.aider.chat.history.md");
    assert!(
        aider_history.exists(),
        "Aider history file should exist after CC→Aider conversion"
    );

    casr_cmd(&tmp)
        .args([
            "--json",
            "resume",
            "cc",
            aider_session_id,
            "--source",
            "aid",
            "--dry-run",
        ])
        .assert()
        .success();
}

#[test]
fn cli_resume_aider_to_cc_works_with_source_hint() {
    let tmp = TempDir::new().unwrap();
    let source_id = setup_cc_fixture(&tmp, "cc_simple");

    let aider_result = casr_cmd(&tmp)
        .args(["--json", "resume", "aid", &source_id])
        .output()
        .expect("CC→Aider seed conversion should run");
    assert!(aider_result.status.success());
    let aider_json: serde_json::Value =
        serde_json::from_slice(&aider_result.stdout).expect("seed conversion JSON should parse");
    let aider_session_id = aider_json["target_session_id"]
        .as_str()
        .expect("aider target_session_id should be present");

    casr_cmd(&tmp)
        .args(["resume", "cc", aider_session_id, "--source", "aid"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Converted"))
        .stdout(predicate::str::contains("aider"))
        .stdout(predicate::str::contains("claude-code"));
}

#[test]
fn cli_resume_cc_to_opencode_works_and_is_discoverable() {
    let tmp = TempDir::new().unwrap();
    let session_id = setup_cc_fixture(&tmp, "cc_simple");

    let output = casr_cmd(&tmp)
        .args(["--json", "resume", "opc", &session_id])
        .output()
        .expect("resume should run");
    assert!(
        output.status.success(),
        "CC→OpenCode conversion should succeed"
    );

    let parsed: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("resume --json output should parse");
    assert_eq!(parsed["ok"], true);
    assert_eq!(parsed["target_provider"].as_str().unwrap(), "opencode");
    let opencode_session_id = parsed["target_session_id"]
        .as_str()
        .expect("target_session_id should be present for non-dry-run");

    let opencode_db = tmp.path().join("opencode/opencode.db");
    assert!(
        opencode_db.exists(),
        "OpenCode DB should exist after CC→OpenCode conversion"
    );

    casr_cmd(&tmp)
        .args(["--json", "info", opencode_session_id])
        .assert()
        .success();
}

#[test]
fn cli_resume_opencode_to_cc_works_with_source_hint() {
    let tmp = TempDir::new().unwrap();
    let source_id = setup_cc_fixture(&tmp, "cc_simple");

    let opencode_result = casr_cmd(&tmp)
        .args(["--json", "resume", "opc", &source_id])
        .output()
        .expect("CC→OpenCode seed conversion should run");
    assert!(opencode_result.status.success());
    let opencode_json: serde_json::Value =
        serde_json::from_slice(&opencode_result.stdout).expect("seed conversion JSON should parse");
    let opencode_session_id = opencode_json["target_session_id"]
        .as_str()
        .expect("opencode target_session_id should be present");

    casr_cmd(&tmp)
        .args(["resume", "cc", opencode_session_id, "--source", "opc"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Converted"))
        .stdout(predicate::str::contains("opencode"))
        .stdout(predicate::str::contains("claude-code"));
}

// ---------------------------------------------------------------------------
// Completions command
// ---------------------------------------------------------------------------

#[test]
fn cli_completions_bash() {
    let tmp = TempDir::new().unwrap();
    casr_cmd(&tmp)
        .args(["completions", "bash"])
        .assert()
        .success()
        .stdout(predicate::str::contains("casr"));
}

#[test]
fn cli_completions_invalid_shell() {
    let tmp = TempDir::new().unwrap();
    casr_cmd(&tmp)
        .args(["completions", "ksh"])
        .assert()
        .failure();
}

// ---------------------------------------------------------------------------
// Verbose / trace flags
// ---------------------------------------------------------------------------

#[test]
fn cli_verbose_flag_accepted() {
    let tmp = TempDir::new().unwrap();
    casr_cmd(&tmp)
        .args(["--verbose", "providers"])
        .assert()
        .success();
}

#[test]
fn cli_trace_flag_accepted() {
    let tmp = TempDir::new().unwrap();
    casr_cmd(&tmp)
        .args(["--trace", "providers"])
        .assert()
        .success();
}
