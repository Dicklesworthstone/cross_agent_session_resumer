//! Provider detection and cross-provider session lookup.
//!
//! The [`ProviderRegistry`] knows about all supported providers and can
//! detect which ones are installed, then locate sessions by ID across
//! all of them.
//!
//! The main entry point for session resolution is [`ProviderRegistry::resolve_session`],
//! which implements a deterministic multi-step algorithm:
//!
//! 1. If `--source <path>` → bypass discovery, resolve directly to file.
//! 2. If `--source <alias>` → only search that provider.
//! 3. Otherwise → search all installed providers, detect ambiguity.

use std::path::{Path, PathBuf};

use tracing::{debug, info, trace, warn};

use crate::error::{Candidate, CasrError};
use crate::model::{CanonicalSession, MessageRole};
use crate::providers::Provider;

// ---------------------------------------------------------------------------
// Source hint — parsed from `--source` CLI flag
// ---------------------------------------------------------------------------

/// Hint from the `--source` CLI flag to constrain session resolution.
#[derive(Debug, Clone)]
pub enum SourceHint {
    /// Provider alias (e.g., `"cc"`, `"cod"`, `"gmi"`) or slug.
    Alias(String),
    /// Direct path to a native session file.
    Path(PathBuf),
}

impl SourceHint {
    /// Parse a `--source` value into a hint.
    ///
    /// Heuristic: if the value contains a path separator or starts with `.`/`~`/`/`,
    /// treat it as a path. Otherwise, treat it as a provider alias.
    pub fn parse(value: &str) -> Self {
        if value.contains(std::path::MAIN_SEPARATOR)
            || value.starts_with('.')
            || value.starts_with('~')
            || value.starts_with('/')
        {
            // Expand leading `~/` to the user's home directory.
            let expanded = if let Some(rest) = value.strip_prefix("~/") {
                dirs::home_dir()
                    .map(|h| h.join(rest))
                    .unwrap_or_else(|| PathBuf::from(value))
            } else {
                PathBuf::from(value)
            };
            SourceHint::Path(expanded)
        } else {
            SourceHint::Alias(value.to_string())
        }
    }
}

// ---------------------------------------------------------------------------
// Resolved session
// ---------------------------------------------------------------------------

/// A successfully resolved session: source provider + file path.
pub struct ResolvedSession<'a> {
    /// The provider that owns this session.
    pub provider: &'a dyn Provider,
    /// Path to the native session file.
    pub path: PathBuf,
}

impl std::fmt::Debug for ResolvedSession<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedSession")
            .field("provider", &self.provider.slug())
            .field("path", &self.path)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Provider registry
// ---------------------------------------------------------------------------

/// Central registry of all known providers.
pub struct ProviderRegistry {
    providers: Vec<Box<dyn Provider>>,
}

impl ProviderRegistry {
    /// Create a registry with all known providers.
    pub fn new(providers: Vec<Box<dyn Provider>>) -> Self {
        Self { providers }
    }

    /// Create the default registry with all built-in providers.
    pub fn default_registry() -> Self {
        Self::new(vec![
            Box::new(crate::providers::claude_code::ClaudeCode),
            Box::new(crate::providers::codex::Codex),
            Box::new(crate::providers::gemini::Gemini),
            Box::new(crate::providers::cursor::Cursor),
            Box::new(crate::providers::cline::Cline),
            Box::new(crate::providers::aider::Aider),
            Box::new(crate::providers::amp::Amp),
            Box::new(crate::providers::opencode::OpenCode),
            Box::new(crate::providers::chatgpt::ChatGpt),
            Box::new(crate::providers::clawdbot::ClawdBot),
            Box::new(crate::providers::vibe::Vibe),
            Box::new(crate::providers::factory::Factory),
            Box::new(crate::providers::openclaw::OpenClaw),
            Box::new(crate::providers::pi_agent::PiAgent),
        ])
    }

    /// Probe each provider for installation status.
    pub fn detect_all(&self) -> Vec<(&dyn Provider, DetectionResult)> {
        self.providers
            .iter()
            .map(|p| {
                let result = p.detect();
                debug!(
                    provider = p.name(),
                    installed = result.installed,
                    "provider detection"
                );
                (p.as_ref(), result)
            })
            .collect()
    }

    /// Return only providers that are currently installed.
    pub fn installed_providers(&self) -> Vec<&dyn Provider> {
        self.providers
            .iter()
            .filter(|p| p.detect().installed)
            .map(|p| p.as_ref())
            .collect()
    }

    /// Return all registered providers regardless of installation status.
    pub fn all_providers(&self) -> Vec<&dyn Provider> {
        self.providers.iter().map(|p| p.as_ref()).collect()
    }

    /// Find a provider by its slug (e.g. `"claude-code"`).
    pub fn find_by_slug(&self, slug: &str) -> Option<&dyn Provider> {
        self.providers
            .iter()
            .find(|p| p.slug() == slug)
            .map(|p| p.as_ref())
    }

    /// Find a provider by its CLI alias (e.g. `"cc"`) or slug.
    pub fn find_by_alias(&self, alias: &str) -> Option<&dyn Provider> {
        self.providers
            .iter()
            .find(|p| p.cli_alias() == alias || p.slug() == alias)
            .map(|p| p.as_ref())
    }

    // -----------------------------------------------------------------------
    // Session resolution — the full algorithm
    // -----------------------------------------------------------------------

    /// Resolve a session ID to its source provider and file path.
    ///
    /// This is the main entry point for the `casr <target> resume <session-id>`
    /// flow. It implements a deterministic multi-step algorithm:
    ///
    /// 1. If `source_hint` is a [`SourceHint::Path`], bypass discovery entirely.
    /// 2. If `source_hint` is a [`SourceHint::Alias`], search only that provider.
    /// 3. Otherwise, search all installed providers via fast-path ownership checks.
    /// 4. Exactly one match → return it.
    /// 5. Multiple matches → [`CasrError::AmbiguousSessionId`].
    /// 6. No matches → [`CasrError::SessionNotFound`] with diagnostics.
    pub fn resolve_session(
        &self,
        session_id: &str,
        source_hint: Option<&SourceHint>,
    ) -> Result<ResolvedSession<'_>, CasrError> {
        match source_hint {
            Some(SourceHint::Path(path)) => self.resolve_from_path(session_id, path),
            Some(SourceHint::Alias(alias)) => self.resolve_with_alias(session_id, alias),
            None => self.resolve_auto(session_id),
        }
    }

    /// Resolve by direct file path — bypass all discovery.
    ///
    /// Identifies the owning provider by checking which provider's session roots
    /// contain the path. Falls back to file extension heuristics.
    fn resolve_from_path(
        &self,
        session_id: &str,
        path: &Path,
    ) -> Result<ResolvedSession<'_>, CasrError> {
        debug!(path = %path.display(), "resolving session from explicit path");

        if !path.is_file() {
            return Err(CasrError::SessionNotFound {
                session_id: session_id.to_string(),
                providers_checked: vec!["(direct path)".to_string()],
                sessions_scanned: 0,
            });
        }

        // Try to identify the owning provider by checking session roots.
        for provider in &self.providers {
            for root in provider.session_roots() {
                if path.starts_with(&root) {
                    info!(
                        provider = provider.name(),
                        path = %path.display(),
                        "resolved session from explicit path"
                    );
                    return Ok(ResolvedSession {
                        provider: provider.as_ref(),
                        path: path.to_path_buf(),
                    });
                }
            }
        }

        // Path exists but does not live under any known provider session root.
        // This can happen when users move/copy session files for archival or sharing.
        //
        // First, try lightweight file signature inference. If that fails, probe all
        // providers and pick the most plausible parser.
        if let Some(provider) = self.infer_provider_for_path(path) {
            info!(
                provider = provider.name(),
                path = %path.display(),
                "resolved session from explicit path via file signature"
            );
            return Ok(ResolvedSession {
                provider,
                path: path.to_path_buf(),
            });
        }

        let mut best: Option<(&dyn Provider, usize, bool)> = None;
        let mut providers_tried: Vec<String> = Vec::new();

        for provider in &self.providers {
            providers_tried.push(provider.slug().to_string());
            let parsed = provider.read_session(path);
            let Ok(session) = parsed else {
                continue;
            };

            if session.messages.is_empty() {
                continue;
            }

            let plausible = is_plausible_session(&session);

            let is_better = best.is_none_or(|(best_provider, best_len, best_plausible)| {
                (plausible, session.messages.len(), provider.slug())
                    > (best_plausible, best_len, best_provider.slug())
            });

            if is_better {
                best = Some((provider.as_ref(), session.messages.len(), plausible));
            }
        }

        if let Some((provider, _len, plausible)) = best {
            if !plausible {
                warn!(
                    provider = provider.name(),
                    path = %path.display(),
                    "no provider root matched path; selected best-effort parser (session may not be resumable)"
                );
            } else {
                info!(
                    provider = provider.name(),
                    path = %path.display(),
                    "resolved session from explicit path via provider probing"
                );
            }
            return Ok(ResolvedSession {
                provider,
                path: path.to_path_buf(),
            });
        }

        Err(CasrError::SessionReadError {
            path: path.to_path_buf(),
            provider: "(unknown)".to_string(),
            detail: format!(
                "Path is not under any provider root and could not be parsed as a session by any provider. Tried: {providers_tried:?}"
            ),
        })
    }

    /// Resolve by alias hint — only search the specified provider.
    fn resolve_with_alias(
        &self,
        session_id: &str,
        alias: &str,
    ) -> Result<ResolvedSession<'_>, CasrError> {
        debug!(
            alias,
            session_id, "resolving session with source alias hint"
        );

        let provider =
            self.find_by_alias(alias)
                .ok_or_else(|| CasrError::UnknownProviderAlias {
                    alias: alias.to_string(),
                    known_aliases: self.known_aliases(),
                })?;

        match provider.owns_session(session_id) {
            Some(path) => {
                info!(
                    provider = provider.name(),
                    path = %path.display(),
                    session_id,
                    "resolved session via alias hint"
                );
                Ok(ResolvedSession { provider, path })
            }
            None => {
                let roots: Vec<String> = provider
                    .session_roots()
                    .iter()
                    .map(|r| r.display().to_string())
                    .collect();
                debug!(
                    provider = provider.name(),
                    ?roots,
                    "session not found in hinted provider"
                );
                Err(CasrError::SessionNotFound {
                    session_id: session_id.to_string(),
                    providers_checked: vec![provider.name().to_string()],
                    sessions_scanned: 0,
                })
            }
        }
    }

    /// Fully automatic resolution — search all installed providers.
    ///
    /// Collects ALL matches (does not short-circuit) to detect ambiguity.
    fn resolve_auto(&self, session_id: &str) -> Result<ResolvedSession<'_>, CasrError> {
        debug!(session_id, "auto-resolving session across all providers");

        let mut matches: Vec<(&dyn Provider, PathBuf)> = Vec::new();
        let mut providers_checked: Vec<String> = Vec::new();

        for provider in &self.providers {
            let detection = provider.detect();
            if !detection.installed {
                trace!(provider = provider.name(), "skipping — not installed");
                continue;
            }

            providers_checked.push(provider.name().to_string());
            trace!(provider = provider.name(), session_id, "searching");

            if let Some(path) = provider.owns_session(session_id) {
                debug!(
                    provider = provider.name(),
                    path = %path.display(),
                    session_id,
                    "candidate match"
                );
                matches.push((provider.as_ref(), path));
            }
        }

        match matches.len() {
            0 => {
                debug!(
                    session_id,
                    ?providers_checked,
                    "session not found in any provider"
                );
                Err(CasrError::SessionNotFound {
                    session_id: session_id.to_string(),
                    providers_checked,
                    sessions_scanned: 0,
                })
            }
            1 => {
                let (provider, path) = matches.into_iter().next().expect("checked len==1");
                info!(
                    provider = provider.name(),
                    path = %path.display(),
                    session_id,
                    "unique session match"
                );
                Ok(ResolvedSession { provider, path })
            }
            _ => {
                let candidates: Vec<Candidate> = matches
                    .iter()
                    .map(|(p, path)| Candidate {
                        provider: p.slug().to_string(),
                        path: path.to_path_buf(),
                    })
                    .collect();
                warn!(
                    session_id,
                    candidate_count = candidates.len(),
                    "ambiguous session ID — multiple providers match"
                );
                Err(CasrError::AmbiguousSessionId {
                    session_id: session_id.to_string(),
                    candidates,
                })
            }
        }
    }

    /// Collect the CLI aliases of all registered providers (for error messages).
    pub fn known_aliases(&self) -> Vec<String> {
        self.providers
            .iter()
            .map(|p| format!("{} ({})", p.cli_alias(), p.name()))
            .collect()
    }
}

fn is_plausible_session(session: &CanonicalSession) -> bool {
    if session.messages.is_empty() {
        return false;
    }
    let has_user = session.messages.iter().any(|m| m.role == MessageRole::User);
    let has_assistant = session
        .messages
        .iter()
        .any(|m| m.role == MessageRole::Assistant);
    has_user && has_assistant
}

impl ProviderRegistry {
    fn infer_provider_for_path(&self, path: &Path) -> Option<&dyn Provider> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        match ext.as_str() {
            "vscdb" => return self.find_by_slug("cursor"),
            "jsonl" => {
                let file = std::fs::File::open(path).ok()?;
                let reader = std::io::BufReader::new(file);
                for line in std::io::BufRead::lines(reader).map_while(Result::ok) {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let value: serde_json::Value = serde_json::from_str(trimmed).ok()?;
                    if value.get("type").and_then(|v| v.as_str()) == Some("session_meta") {
                        return self.find_by_slug("codex");
                    }
                    // Factory: JSONL with session_start typed entry.
                    if value.get("type").and_then(|v| v.as_str()) == Some("session_start") {
                        return self.find_by_slug("factory");
                    }
                    // OpenClaw: type:"session" with version field.
                    if value.get("type").and_then(|v| v.as_str()) == Some("session")
                        && value.get("version").is_some()
                    {
                        return self.find_by_slug("openclaw");
                    }
                    // Pi-Agent: type:"session" with provider/modelId fields.
                    if value.get("type").and_then(|v| v.as_str()) == Some("session")
                        && (value.get("provider").is_some() || value.get("modelId").is_some())
                    {
                        return self.find_by_slug("pi-agent");
                    }
                    // OpenClaw/Pi-Agent: type:"message" with nested "message" object.
                    if value.get("type").and_then(|v| v.as_str()) == Some("message")
                        && value.get("message").is_some()
                    {
                        // Disambiguate by filename pattern: Pi-Agent uses underscore.
                        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                        if stem.contains('_') {
                            return self.find_by_slug("pi-agent");
                        }
                        return self.find_by_slug("openclaw");
                    }
                    if value.get("sessionId").is_some()
                        && value.get("uuid").is_some()
                        && value.get("cwd").is_some()
                    {
                        return self.find_by_slug("claude-code");
                    }
                    // ClawdBot: bare JSONL messages with role+content, no type field.
                    if value.get("role").is_some()
                        && value.get("content").is_some()
                        && value.get("type").is_none()
                    {
                        return self.find_by_slug("clawdbot");
                    }
                    break;
                }
            }
            "json" => {
                let content = std::fs::read_to_string(path).ok()?;
                let value: serde_json::Value = serde_json::from_str(&content).ok()?;

                if value.get("sessionId").is_some() && value.get("messages").is_some() {
                    return self.find_by_slug("gemini");
                }

                // ChatGPT mapping-based conversation format.
                if value.get("mapping").is_some()
                    && (value.get("id").is_some() || value.get("conversation_id").is_some())
                {
                    return self.find_by_slug("chatgpt");
                }

                if value.get("session").is_some() {
                    return self.find_by_slug("codex");
                }
            }
            _ => {}
        }
        None
    }
}

/// Result of probing a provider for installation.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub installed: bool,
    pub version: Option<String>,
    pub evidence: Vec<String>,
}
