#![forbid(unsafe_code)]

//! casr — Cross Agent Session Resumer.
//!
//! CLI entry point: parses arguments, dispatches subcommands, renders output.

use std::process::ExitCode;

use clap::Parser;
use colored::Colorize;
use tracing_subscriber::EnvFilter;

use casr::discovery::ProviderRegistry;
use casr::pipeline::{ConversionPipeline, ConvertOptions};

/// Cross Agent Session Resumer — resume AI coding sessions across providers.
///
/// Convert sessions between Claude Code, Codex, Gemini CLI, and Cursor so you can
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
        /// Target provider alias (cc, cod, gmi, cur).
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
        #[arg(long, default_value = "50")]
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

fn main() -> ExitCode {
    let cli = Cli::parse();
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
            workspace: _,
            limit,
            sort: _,
        } => cmd_list(provider.as_deref(), limit, cli.json),
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

fn cmd_list(provider_filter: Option<&str>, limit: usize, json_mode: bool) -> anyhow::Result<()> {
    let registry = ProviderRegistry::default_registry();
    let installed = registry.installed_providers();

    let mut sessions: Vec<serde_json::Value> = Vec::new();

    for provider in &installed {
        if let Some(filter) = provider_filter
            && provider.cli_alias() != filter
            && provider.slug() != filter
        {
            continue;
        }

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
                if !matches!(ext, Some("jsonl") | Some("json") | Some("vscdb")) {
                    continue;
                }

                // Try to read session metadata.
                match provider.read_session(path) {
                    Ok(session) => {
                        sessions.push(serde_json::json!({
                            "session_id": session.session_id,
                            "provider": provider.slug(),
                            "title": session.title,
                            "messages": session.messages.len(),
                            "workspace": session.workspace.as_ref().map(|w| w.display().to_string()),
                            "started_at": session.started_at,
                            "path": path.display().to_string(),
                        }));
                    }
                    Err(_) => continue,
                }

                if sessions.len() >= limit {
                    break;
                }
            }
        }

        if sessions.len() >= limit {
            break;
        }
    }

    // Sort by started_at descending.
    sessions.sort_by(|a, b| {
        let ta = a.get("started_at").and_then(|v| v.as_i64()).unwrap_or(0);
        let tb = b.get("started_at").and_then(|v| v.as_i64()).unwrap_or(0);
        tb.cmp(&ta)
    });
    sessions.truncate(limit);

    if json_mode {
        println!("{}", serde_json::to_string_pretty(&sessions)?);
    } else {
        if sessions.is_empty() {
            println!(
                "No sessions found. Run {} to check provider status.",
                "casr providers".cyan()
            );
            return Ok(());
        }
        println!(
            "{} ({} sessions)\n",
            "Discoverable sessions".bold(),
            sessions.len()
        );
        for s in &sessions {
            let sid = s["session_id"].as_str().unwrap_or("?");
            let prov = s["provider"].as_str().unwrap_or("?");
            let title = s["title"].as_str().unwrap_or("");
            let msgs = s["messages"].as_u64().unwrap_or(0);
            let display_title = if title.len() > 60 {
                format!("{}...", &title[..57])
            } else {
                title.to_string()
            };

            println!(
                "  {} {} {} {}",
                sid.cyan(),
                format!("[{prov}]").dimmed(),
                format!("{msgs}msg").dimmed(),
                display_title
            );
        }
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
