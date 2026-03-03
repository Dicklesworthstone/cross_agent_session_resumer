#![forbid(unsafe_code)]

//! casr — Cross Agent Session Resumer.
//!
//! CLI entry point: parses arguments, dispatches subcommands, renders output.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use chrono::{Local, Utc};
use clap::Parser;
use colored::Colorize;
use rich_rust::prelude::{Column, Console, JustifyMethod, Style, Table};
use tracing_subscriber::EnvFilter;

use casr::discovery::ProviderRegistry;
use casr::pipeline::{ConversionPipeline, ConvertOptions};

/// Cross Agent Session Resumer — resume AI coding sessions across providers.
///
/// Convert sessions between Claude Code, Codex, Gemini CLI, Cursor, Cline, Aider, Amp, OpenCode, and ChatGPT so you can
/// pick up where you left off with a different agent.
#[derive(Parser, Debug)]
#[command(
    name = "casr",
    version = long_version(),
    about,
    long_about = None,
)]
struct Cli {
    /// Show detailed conversion progress.
    #[arg(long, global = true)]
    verbose: bool,

    /// Show everything including per-message parsing details.
    #[arg(long, global = true)]
    trace: bool,

    /// Output as JSON for machine consumption.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// Convert and resume a session from another provider.
    Resume {
        /// Target provider alias (cc, cod, gmi, cur, cln, aid, amp, opc, gpt).
        target: String,
        /// Session ID to convert.
        session_id: String,

        /// Show what would happen without writing anything.
        #[arg(long)]
        dry_run: bool,

        /// Overwrite existing session in target if it exists.
        #[arg(long)]
        force: bool,

        /// Explicitly specify source provider alias or session file path.
        #[arg(long)]
        source: Option<String>,

        /// Add context messages to help the target agent understand the conversion.
        #[arg(long)]
        enrich: bool,
    },

    /// List all discoverable sessions across installed providers.
    List {
        /// Filter by provider slug.
        #[arg(long)]
        provider: Option<String>,

        /// Filter by workspace path.
        #[arg(long)]
        workspace: Option<String>,

        /// Maximum sessions to show.
        #[arg(long, default_value = "10")]
        limit: usize,

        /// Sort field (date, messages, provider).
        #[arg(long, default_value = "date")]
        sort: String,
    },

    /// Show details for a specific session.
    Info {
        /// Session ID to inspect.
        session_id: String,
    },

    /// List detected providers and their installation status.
    Providers,

    /// Generate shell completions.
    Completions {
        /// Shell to generate completions for (bash, zsh, fish).
        shell: String,
    },
}

/// Build the long version string with embedded build metadata.
///
/// vergen-gix always emits these env vars (uses placeholders when values are
/// unavailable), so `env!()` is safe here.
fn long_version() -> &'static str {
    concat!(
        env!("CARGO_PKG_VERSION"),
        " (",
        env!("VERGEN_GIT_SHA"),
        " ",
        env!("VERGEN_BUILD_TIMESTAMP"),
        " ",
        env!("VERGEN_CARGO_TARGET_TRIPLE"),
        ")",
    )
}

/// Initialize the tracing subscriber based on CLI flags.
///
/// Priority: `--trace` > `--verbose` > `RUST_LOG` env var > default (warn).
fn init_tracing(cli: &Cli) {
    let filter = if cli.trace {
        EnvFilter::new("casr=trace")
    } else if cli.verbose {
        EnvFilter::new("casr=debug")
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"))
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_writer(std::io::stderr)
        .init();
}

/// Rewrite ergonomic shorthand target flags into canonical resume commands.
///
/// Supports:
/// - `casr -cc <session-id> ...`
/// - `casr -cod <session-id> ...`
/// - `casr -gmi <session-id> ...`
///
/// Rewritten form:
/// `casr [global-options] resume <target> <session-id> ...`
fn rewrite_shorthand_resume_args(args: Vec<OsString>) -> Vec<OsString> {
    if args.len() < 2 {
        return args;
    }

    let mut shorthand_idx: Option<usize> = None;
    let mut target_alias: Option<&'static str> = None;

    // Only scan option-like tokens before the first positional token.
    // This preserves regular subcommand behavior (e.g., `casr list`).
    for (idx, arg) in args.iter().enumerate().skip(1) {
        let raw = arg.to_string_lossy();
        if raw == "--" {
            break;
        }
        if !raw.starts_with('-') {
            break;
        }

        let alias = match raw.as_ref() {
            "-cc" => Some("cc"),
            "-cod" => Some("cod"),
            "-gmi" => Some("gmi"),
            _ => None,
        };

        if let Some(a) = alias {
            shorthand_idx = Some(idx);
            target_alias = Some(a);
            break;
        }
    }

    let (idx, alias) = match (shorthand_idx, target_alias) {
        (Some(i), Some(a)) => (i, a),
        _ => return args,
    };

    let mut rewritten = Vec::with_capacity(args.len() + 1);
    rewritten.push(args[0].clone());

    // Preserve any global options before the shorthand flag.
    rewritten.extend(args.iter().take(idx).skip(1).cloned());

    rewritten.push(OsString::from("resume"));
    rewritten.push(OsString::from(alias));

    // Preserve the remaining args after shorthand (session id + options).
    rewritten.extend(args.into_iter().skip(idx + 1));

    rewritten
}

fn main() -> ExitCode {
    let argv = rewrite_shorthand_resume_args(std::env::args_os().collect());
    let cli = Cli::parse_from(argv);
    init_tracing(&cli);

    let result = match cli.command {
        Command::Resume {
            target,
            session_id,
            dry_run,
            force,
            source,
            enrich,
        } => cmd_resume(
            &target,
            &session_id,
            dry_run,
            force,
            source,
            enrich,
            cli.json,
        ),
        Command::List {
            provider,
            workspace,
            limit,
            sort,
        } => cmd_list(
            provider.as_deref(),
            workspace.as_deref(),
            limit,
            &sort,
            cli.json,
        ),
        Command::Info { session_id } => cmd_info(&session_id, cli.json),
        Command::Providers => cmd_providers(cli.json),
        Command::Completions { shell } => cmd_completions(&shell),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            if cli.json {
                let json = serde_json::json!({
                    "ok": false,
                    "error_type": error_type_name(&e),
                    "message": format!("{e}"),
                });
                eprintln!(
                    "{}",
                    serde_json::to_string_pretty(&json).unwrap_or_default()
                );
            } else {
                eprintln!("{} {e}", "Error:".red().bold());
            }
            ExitCode::FAILURE
        }
    }
}

/// Extract a short error type name for JSON output.
fn error_type_name(e: &anyhow::Error) -> &'static str {
    if let Some(casr_err) = e.downcast_ref::<casr::error::CasrError>() {
        match casr_err {
            casr::error::CasrError::SessionNotFound { .. } => "SessionNotFound",
            casr::error::CasrError::AmbiguousSessionId { .. } => "AmbiguousSessionId",
            casr::error::CasrError::UnknownProviderAlias { .. } => "UnknownProviderAlias",
            casr::error::CasrError::ProviderUnavailable { .. } => "ProviderUnavailable",
            casr::error::CasrError::SessionReadError { .. } => "SessionReadError",
            casr::error::CasrError::SessionWriteError { .. } => "SessionWriteError",
            casr::error::CasrError::SessionConflict { .. } => "SessionConflict",
            casr::error::CasrError::ValidationError { .. } => "ValidationError",
            casr::error::CasrError::VerifyFailed { .. } => "VerifyFailed",
        }
    } else {
        "InternalError"
    }
}

// ---------------------------------------------------------------------------
// Command implementations
// ---------------------------------------------------------------------------

fn cmd_resume(
    target: &str,
    session_id: &str,
    dry_run: bool,
    force: bool,
    source: Option<String>,
    enrich: bool,
    json_mode: bool,
) -> anyhow::Result<()> {
    let registry = ProviderRegistry::default_registry();
    let pipeline = ConversionPipeline { registry };

    let opts = ConvertOptions {
        dry_run,
        force,
        verbose: false,
        enrich,
        source_hint: source,
    };

    let result = pipeline.convert(target, session_id, opts)?;

    if json_mode {
        let json = serde_json::json!({
            "ok": true,
            "source_provider": result.source_provider,
            "target_provider": result.target_provider,
            "source_session_id": result.canonical_session.session_id,
            "target_session_id": result.written.as_ref().map(|w| &w.session_id),
            "written_paths": result.written.as_ref().map(|w| &w.paths),
            "resume_command": result.written.as_ref().map(|w| &w.resume_command),
            "dry_run": result.written.is_none(),
            "warnings": result.warnings,
        });
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else if let Some(ref written) = result.written {
        println!(
            "{} Converted {} session to {}",
            "✓".green().bold(),
            result.source_provider.cyan(),
            result.target_provider.cyan()
        );
        println!(
            "  {} → {}",
            "Source".dimmed(),
            result.canonical_session.session_id
        );
        println!("  {} → {}", "Target".dimmed(), written.session_id);
        println!(
            "  {} → {}",
            "Messages".dimmed(),
            result.canonical_session.messages.len()
        );
        for path in &written.paths {
            println!("  {} → {}", "Written".dimmed(), path.display());
        }
        for warning in &result.warnings {
            println!("  {} {warning}", "⚠".yellow());
        }
        println!();
        println!(
            "  {} {}",
            "Resume:".green().bold(),
            written.resume_command.bold()
        );
    } else {
        // Dry run.
        println!(
            "{} Would convert {} session to {}",
            "⊘".cyan().bold(),
            result.source_provider.cyan(),
            result.target_provider.cyan()
        );
        println!(
            "  {} → {} messages",
            "Messages".dimmed(),
            result.canonical_session.messages.len()
        );
        for warning in &result.warnings {
            println!("  {} {warning}", "⚠".yellow());
        }
    }

    Ok(())
}

fn cmd_list(
    provider_filter: Option<&str>,
    workspace_filter: Option<&str>,
    limit: usize,
    sort: &str,
    json_mode: bool,
) -> anyhow::Result<()> {
    let registry = ProviderRegistry::default_registry();
    let installed = registry.installed_providers();
    let provider_filter_slug = provider_filter
        .and_then(|filter| registry.find_by_alias(filter).map(|p| p.slug().to_string()))
        .or_else(|| provider_filter.map(|filter| filter.to_ascii_lowercase()));

    #[derive(Debug)]
    struct SessionSummary {
        session_id: String,
        provider: String,
        title: Option<String>,
        messages: usize,
        workspace: Option<PathBuf>,
        started_at: Option<i64>,
        last_active_at: Option<i64>,
        path: PathBuf,
    }

    impl SessionSummary {
        fn recency_value(&self) -> i64 {
            self.last_active_at.or(self.started_at).unwrap_or(0)
        }

        fn started_at_display(&self) -> String {
            self.started_at
                .and_then(chrono::DateTime::<Utc>::from_timestamp_millis)
                .map(|dt| {
                    dt.with_timezone(&Local)
                        .format("%Y-%m-%d %H:%M")
                        .to_string()
                })
                .unwrap_or_else(|| "-".to_string())
        }

        fn last_active_display(&self, now_millis: i64) -> String {
            self.last_active_at
                .map(|timestamp| format_relative_age(timestamp, now_millis))
                .unwrap_or_else(|| "-".to_string())
        }

        fn to_json(&self) -> serde_json::Value {
            serde_json::json!({
                "session_id": self.session_id,
                "provider": self.provider,
                "title": self.title,
                "messages": self.messages,
                "workspace": self.workspace.as_ref().map(|w| w.display().to_string()),
                "started_at": self.started_at,
                "path": self.path.display().to_string(),
            })
        }
    }

    fn expand_tilde_path(value: &str) -> PathBuf {
        if let Some(rest) = value.strip_prefix("~/")
            && let Some(home) = dirs::home_dir()
        {
            home.join(rest)
        } else {
            PathBuf::from(value)
        }
    }

    fn system_time_to_epoch_millis(time: std::time::SystemTime) -> Option<i64> {
        time.duration_since(std::time::UNIX_EPOCH)
            .ok()
            .and_then(|dur| i64::try_from(dur.as_millis()).ok())
    }

    fn file_mtime_millis(path: &Path) -> i64 {
        path.metadata()
            .ok()
            .and_then(|meta| meta.modified().ok())
            .and_then(system_time_to_epoch_millis)
            .unwrap_or(0)
    }

    fn file_last_activity_millis(path: &Path) -> Option<i64> {
        let meta = path.metadata().ok()?;
        let modified = meta.modified().ok().and_then(system_time_to_epoch_millis);
        let accessed = meta.accessed().ok().and_then(system_time_to_epoch_millis);
        match (modified, accessed) {
            (Some(modified), Some(accessed)) => Some(modified.max(accessed)),
            (Some(modified), None) => Some(modified),
            (None, Some(accessed)) => Some(accessed),
            (None, None) => None,
        }
    }

    fn format_relative_age(timestamp_millis: i64, now_millis: i64) -> String {
        let (delta_millis, suffix) = if now_millis >= timestamp_millis {
            (now_millis.saturating_sub(timestamp_millis), "ago")
        } else {
            (timestamp_millis.saturating_sub(now_millis), "from now")
        };
        let total_seconds = u64::try_from(delta_millis / 1000).unwrap_or(0);
        let days = total_seconds / 86_400;
        let hours = (total_seconds % 86_400) / 3_600;
        let minutes = (total_seconds % 3_600) / 60;
        let seconds = total_seconds % 60;
        format!("{days}d {hours:02}h {minutes:02}m {seconds:02}s {suffix}")
    }

    fn provider_display(provider: &str) -> String {
        match provider {
            "claude-code" => provider.bright_magenta().bold().to_string(),
            "codex" => provider.bright_cyan().bold().to_string(),
            "gemini" => provider.bright_yellow().bold().to_string(),
            "cursor" => provider.bright_blue().bold().to_string(),
            "cline" => provider.bright_green().bold().to_string(),
            "aider" => provider.bright_red().bold().to_string(),
            "amp" => provider.green().bold().to_string(),
            "opencode" => provider.magenta().bold().to_string(),
            "chatgpt" => provider.yellow().bold().to_string(),
            "clawdbot" => provider.cyan().bold().to_string(),
            "vibe" => provider.bright_white().bold().to_string(),
            "factory" => provider.blue().bold().to_string(),
            "openclaw" => provider.red().bold().to_string(),
            "pi-agent" => provider.white().bold().to_string(),
            _ => provider.bold().to_string(),
        }
    }

    fn probe_limit_for_sort(limit: usize, sort: &str, workspace_scoped: bool) -> usize {
        if sort == "date" {
            // Cap expensive provider scans while preserving high confidence for
            // "most recent" results. Workspace-scoped lists can use a tighter cap.
            let multiplier = if workspace_scoped { 3 } else { 8 };
            std::cmp::max(limit.saturating_mul(multiplier), 30)
        } else {
            usize::MAX
        }
    }

    fn workspace_hint_matches(
        provider_slug: &str,
        path: &Path,
        workspace_filter: Option<&PathBuf>,
    ) -> bool {
        let Some(ws) = workspace_filter else {
            return true;
        };

        match provider_slug {
            "claude-code" => {
                let expected = casr::providers::claude_code::project_dir_key(ws.as_path());
                path.parent()
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    == Some(expected.as_str())
            }
            "gemini" => {
                let expected_hash = casr::providers::gemini::project_hash(ws.as_path());
                let observed_hash = path
                    .parent()
                    .and_then(|p| p.parent())
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str());
                match observed_hash {
                    Some(hash) if hash == expected_hash => true,
                    Some(hash)
                        if hash.len() == 64 && hash.chars().all(|c| c.is_ascii_hexdigit()) =>
                    {
                        false
                    }
                    // Keep fixture/legacy layouts permissive.
                    _ => true,
                }
            }
            _ => true,
        }
    }

    fn provider_has_workspace_path_hint(provider_slug: &str) -> bool {
        matches!(provider_slug, "claude-code" | "gemini")
    }

    fn workspace_scoped_listed_sessions(
        provider_slug: &str,
        workspace_filter: Option<&PathBuf>,
    ) -> Option<Vec<(String, PathBuf)>> {
        let ws = workspace_filter?;
        match provider_slug {
            "claude-code" => {
                let claude_home = std::env::var("CLAUDE_HOME")
                    .ok()
                    .map(PathBuf::from)
                    .or_else(|| dirs::home_dir().map(|h| h.join(".claude")))?;
                let expected_dir = claude_home
                    .join("projects")
                    .join(casr::providers::claude_code::project_dir_key(ws.as_path()));
                if !expected_dir.is_dir() {
                    return Some(vec![]);
                }

                let mut sessions: Vec<(String, PathBuf)> = Vec::new();
                let entries = match std::fs::read_dir(&expected_dir) {
                    Ok(entries) => entries,
                    Err(_) => return Some(vec![]),
                };
                for entry in entries.flatten() {
                    let path = entry.path();
                    if !path.is_file() || path.extension().and_then(|e| e.to_str()) != Some("jsonl")
                    {
                        continue;
                    }
                    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                        continue;
                    };
                    sessions.push((stem.to_string(), path));
                }
                Some(sessions)
            }
            "gemini" => {
                let gemini_home = std::env::var("GEMINI_HOME")
                    .ok()
                    .map(PathBuf::from)
                    .or_else(|| dirs::home_dir().map(|h| h.join(".gemini")))?;
                let tmp_root = gemini_home.join("tmp");
                let hash = casr::providers::gemini::project_hash(ws.as_path());
                let chats_dir = tmp_root.join(hash).join("chats");
                if !chats_dir.is_dir() {
                    // Fallback to generic provider enumeration when tmp/ has
                    // legacy/non-hash chat roots (fixtures or older layouts).
                    // Otherwise, return empty early to avoid an expensive scan.
                    let has_legacy_chat_roots =
                        std::fs::read_dir(&tmp_root).ok().is_some_and(|entries| {
                            entries.flatten().any(|entry| {
                                let path = entry.path();
                                if !path.is_dir() || !path.join("chats").is_dir() {
                                    return false;
                                }
                                let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                                    return true;
                                };
                                !(name.len() == 64 && name.chars().all(|c| c.is_ascii_hexdigit()))
                            })
                        });
                    return if has_legacy_chat_roots {
                        None
                    } else {
                        Some(vec![])
                    };
                }

                let mut sessions: Vec<(String, PathBuf)> = Vec::new();
                let entries = match std::fs::read_dir(&chats_dir) {
                    Ok(entries) => entries,
                    Err(_) => return Some(vec![]),
                };
                for entry in entries.flatten() {
                    let path = entry.path();
                    if !path.is_file() {
                        continue;
                    }
                    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                        continue;
                    };
                    if !(name.starts_with("session-") && name.ends_with(".json")) {
                        continue;
                    }
                    let session_id = name
                        .strip_prefix("session-")
                        .and_then(|n| n.strip_suffix(".json"))
                        .unwrap_or(name)
                        .to_string();
                    sessions.push((session_id, path));
                }
                Some(sessions)
            }
            _ => None,
        }
    }

    let workspace_filter_explicit = workspace_filter.is_some();
    let workspace_filter = workspace_filter
        .map(expand_tilde_path)
        .or_else(|| std::env::current_dir().ok());
    let workspace_scope = workspace_filter
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "all workspaces".to_string());
    let workspace_scope_label = if workspace_filter_explicit {
        "workspace project (--workspace)"
    } else {
        "current working-directory project"
    };
    tracing::debug!(
        provider_filter = ?provider_filter_slug,
        workspace = %workspace_scope,
        scope = %workspace_scope_label,
        sort,
        limit,
        "listing sessions"
    );

    let mut sessions: Vec<SessionSummary> = Vec::new();

    for provider in &installed {
        tracing::debug!(provider = provider.slug(), "scanning provider for sessions");
        if let Some(filter_slug) = provider_filter_slug.as_deref()
            && provider.slug() != filter_slug
            && provider.cli_alias() != filter_slug
        {
            continue;
        }

        // Prefer list_sessions() for providers that store multiple sessions
        // in a single file/DB (avoids undercounting).
        let scoped_listed =
            workspace_scoped_listed_sessions(provider.slug(), workspace_filter.as_ref());
        if let Some(listed) = scoped_listed.or_else(|| provider.list_sessions()) {
            let mut listed = listed;
            let probe_limit = probe_limit_for_sort(limit, sort, workspace_filter.is_some());
            if listed.len() > probe_limit {
                listed.sort_by_key(|(_, path)| std::cmp::Reverse(file_mtime_millis(path)));
                listed.truncate(probe_limit);
            }

            for (session_id, path) in listed {
                if !workspace_hint_matches(provider.slug(), &path, workspace_filter.as_ref()) {
                    continue;
                }
                match provider.read_session(&path) {
                    Ok(session) => {
                        let last_active_at = file_last_activity_millis(&path);
                        sessions.push(SessionSummary {
                            session_id: session.session_id,
                            provider: provider.slug().to_string(),
                            title: session.title,
                            messages: session.messages.len(),
                            workspace: session.workspace,
                            started_at: session.started_at,
                            last_active_at,
                            path,
                        });
                    }
                    Err(_) => continue,
                }
                let _ = session_id; // returned by provider for reference
            }
            continue;
        }

        let mut candidate_paths: Vec<PathBuf> = Vec::new();

        for root in provider.session_roots() {
            let walker = walkdir::WalkDir::new(&root)
                .max_depth(4)
                .into_iter()
                .filter_map(Result::ok);

            for entry in walker {
                if !entry.file_type().is_file() {
                    continue;
                }
                let path = entry.path();
                let ext = path.extension().and_then(|e| e.to_str());
                if !matches!(
                    ext,
                    Some("jsonl")
                        | Some("json")
                        | Some("vscdb")
                        | Some("md")
                        | Some("db")
                        | Some("sqlite")
                ) {
                    continue;
                }

                if !workspace_hint_matches(provider.slug(), path, workspace_filter.as_ref()) {
                    continue;
                }

                candidate_paths.push(path.to_path_buf());
            }
        }

        let probe_limit = probe_limit_for_sort(limit, sort, workspace_filter.is_some());
        if candidate_paths.len() > probe_limit {
            candidate_paths.sort_by_key(|path| std::cmp::Reverse(file_mtime_millis(path)));
            candidate_paths.truncate(probe_limit);
        }

        for path in candidate_paths {
            // Try to read session metadata.
            match provider.read_session(&path) {
                Ok(session) => {
                    let last_active_at = file_last_activity_millis(&path);
                    sessions.push(SessionSummary {
                        session_id: session.session_id,
                        provider: provider.slug().to_string(),
                        title: session.title,
                        messages: session.messages.len(),
                        workspace: session.workspace,
                        started_at: session.started_at,
                        last_active_at,
                        path,
                    });
                }
                Err(_) => continue,
            }
        }
    }

    if let Some(filter) = workspace_filter.as_ref() {
        sessions.retain(|s| {
            s.workspace.as_ref().is_some_and(|w| w.starts_with(filter))
                || (provider_has_workspace_path_hint(&s.provider)
                    && workspace_hint_matches(&s.provider, &s.path, Some(filter)))
        });
    }

    match sort {
        "date" => sessions.sort_by_key(|s| std::cmp::Reverse(s.recency_value())),
        "messages" => sessions.sort_by(|a, b| {
            b.messages
                .cmp(&a.messages)
                .then_with(|| b.recency_value().cmp(&a.recency_value()))
        }),
        "provider" => sessions.sort_by(|a, b| {
            a.provider
                .cmp(&b.provider)
                .then_with(|| b.recency_value().cmp(&a.recency_value()))
        }),
        other => {
            return Err(anyhow::anyhow!(
                "Unknown sort field '{other}'. Expected one of: date, messages, provider."
            ));
        }
    }
    sessions.truncate(limit);
    tracing::debug!(sessions = sessions.len(), sort, "list sessions complete");

    if json_mode {
        let json: Vec<serde_json::Value> = sessions.iter().map(SessionSummary::to_json).collect();
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        if sessions.is_empty() {
            println!(
                "No sessions found for {} {}. Run {} to check provider status.",
                workspace_scope_label.cyan(),
                workspace_scope.cyan(),
                "casr providers".cyan(),
            );
            return Ok(());
        }

        let console = Console::new();
        console.print(&format!(
            "[bold cyan]Project-scoped sessions[/] for [bold]{workspace_scope}[/]"
        ));
        console.print(&format!("[dim]Scope:[/] [bold]{workspace_scope_label}[/]"));

        let mut table = Table::new()
            .title(format!(
                "Top {} Most Recently Active Sessions in This Project",
                sessions.len()
            ))
            .header_style(Style::parse("bold white on blue").unwrap_or_default())
            .border_style(Style::parse("cyan").unwrap_or_default())
            .with_column(Column::new("#").justify(JustifyMethod::Right).width(3))
            .with_column(Column::new("Provider").min_width(12))
            .with_column(Column::new("Session ID").min_width(36))
            .with_column(Column::new("Msgs").justify(JustifyMethod::Right).width(6))
            .with_column(Column::new("Started").min_width(16))
            .with_column(Column::new("Last Active").min_width(22));

        let now_millis = Utc::now().timestamp_millis();

        for (idx, s) in sessions.iter().enumerate() {
            let rank = (idx + 1).to_string();
            let provider = provider_display(&s.provider);
            let session_id = s.session_id.clone();
            let messages = s.messages.to_string();
            let started = s.started_at_display();
            let last_active = s.last_active_display(now_millis);
            table.add_row_cells([
                rank.as_str(),
                provider.as_str(),
                session_id.as_str(),
                messages.as_str(),
                started.as_str(),
                last_active.as_str(),
            ]);
        }

        console.print_renderable(&table);
        console.print("[dim]Tip:[/] run [bold]casr info <session-id>[/] for full metadata.");
    }

    Ok(())
}

fn cmd_info(session_id: &str, json_mode: bool) -> anyhow::Result<()> {
    let registry = ProviderRegistry::default_registry();
    let resolved = registry.resolve_session(session_id, None)?;
    let session = resolved.provider.read_session(&resolved.path)?;

    if json_mode {
        let json = serde_json::json!({
            "session_id": session.session_id,
            "provider": session.provider_slug,
            "title": session.title,
            "workspace": session.workspace.as_ref().map(|w| w.display().to_string()),
            "messages": session.messages.len(),
            "started_at": session.started_at,
            "ended_at": session.ended_at,
            "model_name": session.model_name,
            "source_path": session.source_path.display().to_string(),
            "metadata": session.metadata,
        });
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        println!("{}\n", "Session Info".bold());
        println!("  {} {}", "ID:".dimmed(), session.session_id.cyan());
        println!("  {} {}", "Provider:".dimmed(), session.provider_slug);
        if let Some(ref title) = session.title {
            println!("  {} {title}", "Title:".dimmed());
        }
        if let Some(ref ws) = session.workspace {
            println!("  {} {}", "Workspace:".dimmed(), ws.display());
        }
        println!("  {} {}", "Messages:".dimmed(), session.messages.len());
        if let Some(ref model) = session.model_name {
            println!("  {} {model}", "Model:".dimmed());
        }
        println!("  {} {}", "Path:".dimmed(), session.source_path.display());

        // Show role breakdown.
        let user_count = session
            .messages
            .iter()
            .filter(|m| m.role == casr::model::MessageRole::User)
            .count();
        let asst_count = session
            .messages
            .iter()
            .filter(|m| m.role == casr::model::MessageRole::Assistant)
            .count();
        println!(
            "  {} {user_count} user, {asst_count} assistant",
            "Roles:".dimmed()
        );
    }

    Ok(())
}

fn cmd_providers(json_mode: bool) -> anyhow::Result<()> {
    let registry = ProviderRegistry::default_registry();
    let results = registry.detect_all();

    if json_mode {
        let providers: Vec<serde_json::Value> = results
            .iter()
            .map(|(p, det)| {
                serde_json::json!({
                    "name": p.name(),
                    "slug": p.slug(),
                    "alias": p.cli_alias(),
                    "installed": det.installed,
                    "version": det.version,
                    "evidence": det.evidence,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&providers)?);
    } else {
        println!("{}\n", "Detected Providers".bold());
        for (provider, detection) in &results {
            let status = if detection.installed {
                "✓".green().bold().to_string()
            } else {
                "✗".red().bold().to_string()
            };
            println!(
                "  {status} {} ({}) — alias: {}",
                provider.name(),
                provider.slug(),
                provider.cli_alias().cyan()
            );
            for ev in &detection.evidence {
                println!("    {ev}");
            }
        }
    }

    Ok(())
}

fn cmd_completions(shell: &str) -> anyhow::Result<()> {
    use clap::CommandFactory;
    use clap_complete::{Shell, generate};

    let parsed_shell: Shell = shell
        .parse()
        .map_err(|_| anyhow::anyhow!("Unknown shell '{shell}'. Use: bash, zsh, fish"))?;

    let mut cmd = Cli::command();
    generate(parsed_shell, &mut cmd, "casr", &mut std::io::stdout());

    Ok(())
}
