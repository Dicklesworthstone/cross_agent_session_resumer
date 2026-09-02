//! OpenCode provider — reads/writes sessions from SQLite `opencode.db`.
//!
//! OpenCode stores session state in a SQLite database named `opencode.db`.
//! Two on-disk schemas exist, and the reader detects which one a DB carries
//! by introspecting `sqlite_master` (issue #26):
//!
//! - **Legacy (plural)** — the Go-era layout: `sessions`, `messages` (with a
//!   `parts` JSON column) and `files`. Read and written.
//! - **1.x (singular)** — the event-sourced layout: `session`, `message` and
//!   `part`, where each message/part row carries a `data` JSON blob. Read
//!   only; direct writes into a live 1.x DB are refused.
//!
//! A DB matching neither schema fails loudly naming the missing tables and
//! the tables actually present, instead of reporting zero sessions.
//!
//! casr addresses specific OpenCode sessions using a virtual path form:
//! `<db-path>/<urlencoded-session-id>`
//! This mirrors the approach used by Cursor and Aider providers.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::Context;
use rusqlite::{Connection, OpenFlags};
use tracing::{debug, info, trace};

use crate::discovery::DetectionResult;
use crate::model::{
    CanonicalMessage, CanonicalSession, MessageRole, ToolCall, ToolResult, flatten_content,
    normalize_role, parse_timestamp, reindex_messages, truncate_title,
};
use crate::providers::{Provider, WriteOptions, WrittenSession};

/// OpenCode provider implementation.
pub struct OpenCode;

/// Which on-disk layout an `opencode.db` carries (see module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DbSchema {
    /// Go-era plural tables: `sessions` / `messages` (`parts` JSON column).
    Legacy,
    /// OpenCode 1.x singular tables: `session` / `message` / `part`, each
    /// message and part row carrying a `data` JSON blob.
    V1,
}

impl DbSchema {
    /// Tables a DB must carry to be read as this schema.
    const fn required_tables(self) -> &'static [&'static str] {
        match self {
            Self::Legacy => &["sessions", "messages"],
            Self::V1 => &["session", "message", "part"],
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::V1 => "1.x",
        }
    }

    /// Metadata tag stored in `session.metadata.opencode_schema`.
    const fn metadata_tag(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::V1 => "v1",
        }
    }
}

const DB_FILENAME: &str = "opencode.db";
const DATA_DIRNAME: &str = ".opencode";

impl OpenCode {
    /// Parse OPENCODE environment overrides into a target DB path.
    ///
    /// Supported overrides:
    /// - `OPENCODE_DB_PATH` (direct file path)
    /// - `OPENCODE_HOME` (directory containing `opencode.db`, or a direct `.db` path)
    fn env_db_path() -> Option<PathBuf> {
        if let Ok(path) = std::env::var("OPENCODE_DB_PATH")
            && !path.trim().is_empty()
        {
            return Some(PathBuf::from(path));
        }

        if let Ok(home) = std::env::var("OPENCODE_HOME")
            && !home.trim().is_empty()
        {
            let home_path = PathBuf::from(home);
            if home_path.extension().is_some_and(|ext| ext == "db") {
                return Some(home_path);
            }
            return Some(home_path.join(DB_FILENAME));
        }

        None
    }

    /// Candidate global config files that may contain `data.directory`.
    fn config_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();
        if let Some(home) = dirs::home_dir() {
            paths.push(home.join(".opencode.json"));
            paths.push(home.join(".config/opencode/.opencode.json"));
        }
        if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME")
            && !xdg.trim().is_empty()
        {
            paths.push(PathBuf::from(xdg).join("opencode/.opencode.json"));
        }
        paths
    }

    /// Parse absolute `data.directory` values from OpenCode config files.
    fn configured_data_dirs() -> Vec<PathBuf> {
        let mut dirs = Vec::new();

        for cfg in Self::config_paths() {
            let Ok(text) = std::fs::read_to_string(&cfg) else {
                continue;
            };
            let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
                continue;
            };
            let Some(dir) = json
                .pointer("/data/directory")
                .and_then(serde_json::Value::as_str)
            else {
                continue;
            };

            let data_dir = PathBuf::from(dir);
            if data_dir.is_absolute() {
                dirs.push(data_dir);
            }
        }

        dirs
    }

    /// Candidate DB paths derived from the XDG data directory.
    ///
    /// OpenCode 1.x stores its database under the XDG data home
    /// (`$XDG_DATA_HOME/opencode/opencode.db`, defaulting to
    /// `~/.local/share/opencode/opencode.db`). Users who redirect
    /// `XDG_DATA_HOME` per project otherwise get no discovery hit at all
    /// (issue #26).
    fn xdg_data_db_candidates(xdg_data_home: Option<&str>, home: Option<&Path>) -> Vec<PathBuf> {
        let mut candidates = Vec::new();
        if let Some(xdg) = xdg_data_home.map(str::trim).filter(|s| !s.is_empty()) {
            candidates.push(PathBuf::from(xdg).join("opencode").join(DB_FILENAME));
        }
        if let Some(home) = home {
            candidates.push(home.join(".local/share/opencode").join(DB_FILENAME));
        }
        candidates
    }

    /// Candidate DB paths from current directory and parents (`.opencode/opencode.db`).
    fn cwd_ancestor_db_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();
        let Ok(cwd) = std::env::current_dir() else {
            return paths;
        };

        for ancestor in cwd.ancestors() {
            paths.push(ancestor.join(DATA_DIRNAME).join(DB_FILENAME));
        }

        paths
    }

    /// Discover existing OpenCode DB files.
    ///
    /// If env override is set, discovery is constrained to that location.
    fn find_db_files() -> Vec<PathBuf> {
        if let Some(env_db) = Self::env_db_path() {
            return if env_db.is_file() {
                vec![env_db]
            } else {
                Vec::new()
            };
        }

        let mut candidates = Vec::new();
        candidates.extend(Self::cwd_ancestor_db_paths());
        if let Some(home) = dirs::home_dir() {
            candidates.push(home.join(DATA_DIRNAME).join(DB_FILENAME));
        }
        candidates.extend(Self::xdg_data_db_candidates(
            std::env::var("XDG_DATA_HOME").ok().as_deref(),
            dirs::home_dir().as_deref(),
        ));
        for data_dir in Self::configured_data_dirs() {
            candidates.push(data_dir.join(DB_FILENAME));
        }

        dedup_existing_files(candidates)
    }

    /// Resolve target DB path for writes.
    fn choose_target_db_path(session: &CanonicalSession) -> anyhow::Result<PathBuf> {
        if let Some(env_db) = Self::env_db_path() {
            return Ok(env_db);
        }

        if let Some(workspace) = &session.workspace {
            // Discovery builds its candidates from `current_dir()`, which
            // resolves symlinks (macOS spells `/var/...` cwds as
            // `/private/var/...`). Canonicalize so the DB this write creates
            // is the very path discovery will report; otherwise the same
            // session round-trips under two spellings that compare unequal.
            let workspace = workspace
                .canonicalize()
                .unwrap_or_else(|_| workspace.clone());
            return Ok(workspace.join(DATA_DIRNAME).join(DB_FILENAME));
        }

        if let Some(existing) = Self::find_db_files().into_iter().next() {
            return Ok(existing);
        }

        let cwd = std::env::current_dir().context("failed to determine current directory")?;
        Ok(cwd.join(DATA_DIRNAME).join(DB_FILENAME))
    }

    /// Build virtual per-session path: `<db-path>/<urlencoded-session-id>`.
    fn virtual_session_path(db_path: &Path, session_id: &str) -> PathBuf {
        let encoded = urlencoding::encode(session_id);
        db_path.join(encoded.as_ref())
    }

    /// Parse virtual path back into `(db_path, session_id)`.
    fn parse_virtual_path(path: &Path) -> Option<(PathBuf, String)> {
        let parent = path.parent()?;
        if !parent.is_file() {
            return None;
        }
        if parent.file_name().and_then(|n| n.to_str()) != Some(DB_FILENAME) {
            return None;
        }

        let encoded = path.file_name()?.to_str()?;
        let decoded = urlencoding::decode(encoded).ok()?;
        Some((parent.to_path_buf(), decoded.into_owned()))
    }

    /// Open DB in read-only mode.
    fn open_db(path: &Path) -> anyhow::Result<Connection> {
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("failed to open OpenCode DB: {}", path.display()))?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        Ok(conn)
    }

    /// Open DB in read-write/create mode.
    fn open_db_rw(path: &Path) -> anyhow::Result<Connection> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory: {}", parent.display()))?;
        }

        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("failed to open OpenCode DB for writing: {}", path.display()))?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        Ok(conn)
    }

    fn table_exists(conn: &Connection, table: &str) -> bool {
        conn.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1")
            .and_then(|mut stmt| stmt.exists(rusqlite::params![table]))
            .unwrap_or(false)
    }

    /// List every table name in the DB, sorted (from `sqlite_master`).
    fn list_tables(conn: &Connection) -> Vec<String> {
        let Ok(mut stmt) =
            conn.prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        else {
            return Vec::new();
        };
        let Ok(rows) = stmt.query_map([], |row| row.get::<_, String>(0)) else {
            return Vec::new();
        };
        rows.flatten().collect()
    }

    /// Detect which OpenCode schema a DB carries by introspecting
    /// `sqlite_master`, or build a loud diagnostic naming the missing
    /// table(s) and the tables actually present.
    ///
    /// The silent-zero behavior this replaces was indistinguishable from
    /// "no sessions yet" (issue #26).
    fn detect_schema(conn: &Connection, db_path: &Path) -> anyhow::Result<DbSchema> {
        let has = |table: &str| Self::table_exists(conn, table);

        let missing_legacy: Vec<&str> = DbSchema::Legacy
            .required_tables()
            .iter()
            .copied()
            .filter(|table| !has(table))
            .collect();
        if missing_legacy.is_empty() {
            return Ok(DbSchema::Legacy);
        }

        let missing_v1: Vec<&str> = DbSchema::V1
            .required_tables()
            .iter()
            .copied()
            .filter(|table| !has(table))
            .collect();
        if missing_v1.is_empty() {
            return Ok(DbSchema::V1);
        }

        let found = Self::list_tables(conn);
        let found_desc = if found.is_empty() {
            "no tables at all".to_string()
        } else {
            format!("tables present: {}", found.join(", "))
        };

        // Report against whichever schema the DB is closer to, so a partial
        // 1.x DB names its own missing tables rather than the legacy ones.
        let (schema, missing) = if missing_v1.len() < DbSchema::V1.required_tables().len() {
            (DbSchema::V1, missing_v1)
        } else {
            (DbSchema::Legacy, missing_legacy)
        };

        anyhow::bail!(
            "OpenCode DB {} does not match either schema casr reads \
             (legacy plural sessions/messages, or 1.x singular session/message/part); \
             closest is the {} schema, missing table(s): {}; {}",
            db_path.display(),
            schema.label(),
            missing.join(", "),
            found_desc,
        )
    }

    /// The schema diagnostic as a plain message, or `None` when readable.
    #[cfg(test)]
    fn schema_mismatch(conn: &Connection, db_path: &Path) -> Option<String> {
        Self::detect_schema(conn, db_path)
            .err()
            .map(|err| err.to_string())
    }

    /// Pull a column by name, tolerating columns absent in older or newer
    /// OpenCode 1.x revisions (returns `None` instead of failing the row).
    fn col<T: rusqlite::types::FromSql>(row: &rusqlite::Row<'_>, name: &str) -> Option<T> {
        let idx = row.as_ref().column_index(name).ok()?;
        row.get::<_, Option<T>>(idx).ok().flatten()
    }

    fn trigger_exists(conn: &Connection, trigger: &str) -> bool {
        conn.prepare("SELECT 1 FROM sqlite_master WHERE type='trigger' AND name=?1")
            .and_then(|mut stmt| stmt.exists(rusqlite::params![trigger]))
            .unwrap_or(false)
    }

    /// Ensure core OpenCode tables exist.
    fn ensure_schema(conn: &Connection) -> anyhow::Result<()> {
        conn.execute_batch(
            r#"
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    parent_session_id TEXT,
    title TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0 CHECK (message_count >= 0),
    prompt_tokens INTEGER NOT NULL DEFAULT 0 CHECK (prompt_tokens >= 0),
    completion_tokens INTEGER NOT NULL DEFAULT 0 CHECK (completion_tokens >= 0),
    cost REAL NOT NULL DEFAULT 0.0 CHECK (cost >= 0.0),
    updated_at INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    summary_message_id TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    parts TEXT NOT NULL DEFAULT '[]',
    model TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    finished_at INTEGER,
    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    path TEXT NOT NULL,
    content TEXT NOT NULL,
    version TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE,
    UNIQUE(path, session_id, version)
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages (session_id);
CREATE INDEX IF NOT EXISTS idx_files_session_id ON files (session_id);
"#,
        )
        .context("failed to initialize OpenCode schema")?;
        Ok(())
    }

    fn session_exists(conn: &Connection, schema: DbSchema, session_id: &str) -> bool {
        let sql = match schema {
            DbSchema::Legacy => "SELECT 1 FROM sessions WHERE id = ?1 LIMIT 1",
            DbSchema::V1 => "SELECT 1 FROM session WHERE id = ?1 LIMIT 1",
        };
        conn.prepare(sql)
            .and_then(|mut stmt| stmt.exists(rusqlite::params![session_id]))
            .unwrap_or(false)
    }

    fn newest_root_session_id(conn: &Connection, schema: DbSchema) -> Option<String> {
        let sql = match schema {
            DbSchema::Legacy => {
                "SELECT id FROM sessions WHERE parent_session_id IS NULL \
                 ORDER BY created_at DESC LIMIT 1"
            }
            DbSchema::V1 => {
                "SELECT id FROM session WHERE parent_id IS NULL \
                 ORDER BY time_created DESC, id DESC LIMIT 1"
            }
        };
        conn.query_row(sql, [], |row| row.get(0)).ok()
    }

    /// All session ids in the DB, newest first.
    fn all_session_ids(conn: &Connection, schema: DbSchema) -> Vec<String> {
        let sql = match schema {
            DbSchema::Legacy => "SELECT id FROM sessions ORDER BY created_at DESC",
            DbSchema::V1 => "SELECT id FROM session ORDER BY time_created DESC, id DESC",
        };
        let Ok(mut stmt) = conn.prepare(sql) else {
            return Vec::new();
        };
        let Ok(rows) = stmt.query_map([], |row| row.get::<_, String>(0)) else {
            return Vec::new();
        };
        rows.flatten().collect()
    }

    fn workspace_from_db_path(db_path: &Path) -> Option<PathBuf> {
        let data_dir = db_path.parent()?;
        if data_dir.file_name().and_then(|n| n.to_str()) == Some(DATA_DIRNAME) {
            return data_dir.parent().map(Path::to_path_buf);
        }
        None
    }

    fn read_session_by_id(
        conn: &Connection,
        db_path: &Path,
        session_id: &str,
    ) -> anyhow::Result<CanonicalSession> {
        match Self::detect_schema(conn, db_path)? {
            DbSchema::Legacy => Self::read_legacy_session(conn, db_path, session_id),
            DbSchema::V1 => Self::read_v1_session(conn, db_path, session_id),
        }
    }

    /// Read a session from the OpenCode 1.x singular schema.
    ///
    /// `session` columns are pulled by name with tolerance for revisions that
    /// lack some of them; `message.data` / `part.data` are the JSON blobs
    /// OpenCode hydrates its `Message`/`Part` values from (ids and
    /// `session_id`/`message_id` live in dedicated columns, not in `data`).
    fn read_v1_session(
        conn: &Connection,
        db_path: &Path,
        session_id: &str,
    ) -> anyhow::Result<CanonicalSession> {
        let session_row = conn
            .query_row(
                "SELECT * FROM session WHERE id = ?1 LIMIT 1",
                rusqlite::params![session_id],
                |row| {
                    Ok(V1SessionRow {
                        title: Self::col(row, "title"),
                        directory: Self::col(row, "directory"),
                        parent_id: Self::col(row, "parent_id"),
                        time_created: Self::col(row, "time_created"),
                        time_updated: Self::col(row, "time_updated"),
                        cost: Self::col(row, "cost"),
                        tokens_input: Self::col(row, "tokens_input"),
                        tokens_output: Self::col(row, "tokens_output"),
                        tokens_reasoning: Self::col(row, "tokens_reasoning"),
                        slug: Self::col(row, "slug"),
                        version: Self::col(row, "version"),
                        project_id: Self::col(row, "project_id"),
                        model_json: Self::col(row, "model"),
                    })
                },
            )
            .with_context(|| {
                format!("session '{session_id}' not found in {}", db_path.display())
            })?;

        // Parts grouped by message id. Join through `message` rather than
        // relying on `part.session_id`, which older 1.x revisions lack.
        let mut parts_by_message: HashMap<String, Vec<serde_json::Value>> = HashMap::new();
        {
            let mut stmt = conn
                .prepare(
                    "SELECT p.message_id, p.id, p.data FROM part p
                     JOIN message m ON m.id = p.message_id
                     WHERE m.session_id = ?1
                     ORDER BY p.message_id ASC, p.id ASC",
                )
                .context("failed to prepare OpenCode 1.x part query")?;
            let rows = stmt.query_map(rusqlite::params![session_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })?;
            for row in rows {
                let (message_id, part_id, data_json) = row?;
                let mut part = data_json
                    .as_deref()
                    .and_then(|json| serde_json::from_str::<serde_json::Value>(json).ok())
                    .unwrap_or_else(|| serde_json::json!({}));
                if let Some(obj) = part.as_object_mut() {
                    obj.insert("id".to_string(), serde_json::Value::from(part_id));
                }
                parts_by_message.entry(message_id).or_default().push(part);
            }
        }

        let mut started_at = session_row
            .time_created
            .and_then(|ts| parse_timestamp(&serde_json::Value::from(ts)));
        let mut ended_at = session_row
            .time_updated
            .and_then(|ts| parse_timestamp(&serde_json::Value::from(ts)))
            .or(started_at);
        let mut model_counts: HashMap<String, usize> = HashMap::new();
        let mut messages = Vec::new();

        let mut stmt = conn
            .prepare(
                "SELECT * FROM message WHERE session_id = ?1
                 ORDER BY time_created ASC, id ASC",
            )
            .context("failed to prepare OpenCode 1.x message query")?;
        let rows = stmt.query_map(rusqlite::params![session_id], |row| {
            Ok((
                row.get::<_, String>("id")?,
                Self::col::<i64>(row, "time_created"),
                Self::col::<String>(row, "data"),
            ))
        })?;

        for row in rows {
            let (message_id, row_time_created, data_json) = row?;
            let info = data_json
                .as_deref()
                .and_then(|json| serde_json::from_str::<serde_json::Value>(json).ok())
                .unwrap_or_else(|| serde_json::json!({}));

            let role_raw = info
                .get("role")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown");
            let created_raw = info
                .pointer("/time/created")
                .and_then(serde_json::Value::as_i64)
                .or(row_time_created);
            let timestamp =
                created_raw.and_then(|ts| parse_timestamp(&serde_json::Value::from(ts)));
            if let Some(ts) = timestamp {
                started_at = Some(started_at.map_or(ts, |current| current.min(ts)));
                ended_at = Some(ended_at.map_or(ts, |current| current.max(ts)));
            }
            if let Some(completed) = info
                .pointer("/time/completed")
                .and_then(serde_json::Value::as_i64)
                .and_then(|ts| parse_timestamp(&serde_json::Value::from(ts)))
            {
                ended_at = Some(ended_at.map_or(completed, |current| current.max(completed)));
            }

            // Assistant rows carry a flat `modelID`; user rows nest it under
            // `model.modelID`. Only assistant turns count toward the session
            // model, matching what actually produced the output.
            let model_id = info
                .get("modelID")
                .or_else(|| info.pointer("/model/modelID"))
                .and_then(serde_json::Value::as_str)
                .filter(|m| !m.is_empty())
                .map(ToString::to_string);
            if role_raw == "assistant"
                && let Some(model_name) = &model_id
            {
                *model_counts.entry(model_name.clone()).or_insert(0) += 1;
            }

            let raw_parts =
                serde_json::Value::Array(parts_by_message.remove(&message_id).unwrap_or_default());
            let (content, tool_calls, tool_results) = parse_v1_parts(&raw_parts);

            messages.push(CanonicalMessage {
                idx: 0,
                role: normalize_role(role_raw),
                content,
                timestamp,
                author: model_id,
                tool_calls,
                tool_results,
                extra: serde_json::json!({
                    "opencode_message_id": message_id,
                    "opencode_message": info,
                    "opencode_parts": raw_parts,
                }),
            });
        }

        reindex_messages(&mut messages);

        let title = session_row
            .title
            .filter(|t| !t.trim().is_empty())
            .or_else(|| {
                messages
                    .iter()
                    .find(|m| m.role == MessageRole::User)
                    .map(|m| truncate_title(&m.content, 80))
                    .filter(|t| !t.is_empty())
            });

        let model_name = model_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(name, _)| name)
            .or_else(|| {
                session_row
                    .model_json
                    .as_deref()
                    .and_then(|json| serde_json::from_str::<serde_json::Value>(json).ok())
                    .and_then(|model| {
                        model
                            .get("id")
                            .and_then(serde_json::Value::as_str)
                            .map(ToString::to_string)
                    })
            });

        let workspace = session_row
            .directory
            .as_deref()
            .map(str::trim)
            .filter(|d| !d.is_empty())
            .map(PathBuf::from)
            .or_else(|| Self::workspace_from_db_path(db_path));

        Ok(CanonicalSession {
            session_id: session_id.to_string(),
            provider_slug: "opencode".to_string(),
            workspace,
            title,
            started_at,
            ended_at,
            messages,
            metadata: serde_json::json!({
                "opencode_db": db_path.display().to_string(),
                "opencode_schema": DbSchema::V1.metadata_tag(),
                "parent_session_id": session_row.parent_id,
                "prompt_tokens": session_row.tokens_input.unwrap_or(0),
                "completion_tokens": session_row.tokens_output.unwrap_or(0),
                "reasoning_tokens": session_row.tokens_reasoning.unwrap_or(0),
                "cost": session_row.cost.unwrap_or(0.0),
                "directory": session_row.directory,
                "slug": session_row.slug,
                "opencode_version": session_row.version,
                "project_id": session_row.project_id,
            }),
            source_path: Self::virtual_session_path(db_path, session_id),
            model_name,
        })
    }

    /// Read a session from the legacy (plural) schema.
    fn read_legacy_session(
        conn: &Connection,
        db_path: &Path,
        session_id: &str,
    ) -> anyhow::Result<CanonicalSession> {
        let (title_raw, created_raw, updated_raw, parent_session_id, prompt_tokens, completion_tokens, cost): (
            String,
            i64,
            i64,
            Option<String>,
            i64,
            i64,
            f64,
        ) = conn
            .query_row(
                "SELECT title, created_at, updated_at, parent_session_id, prompt_tokens, completion_tokens, cost
                 FROM sessions
                 WHERE id = ?1
                 LIMIT 1",
                rusqlite::params![session_id],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                    ))
                },
            )
            .with_context(|| format!("session '{session_id}' not found in {}", db_path.display()))?;

        let mut started_at = parse_timestamp(&serde_json::Value::from(created_raw));
        let mut ended_at = parse_timestamp(&serde_json::Value::from(updated_raw)).or(started_at);
        let mut model_counts: HashMap<String, usize> = HashMap::new();
        let mut messages = Vec::new();

        let mut stmt = conn
            .prepare(
                "SELECT id, role, parts, model, created_at, updated_at, finished_at
                 FROM messages
                 WHERE session_id = ?1
                 ORDER BY created_at ASC, id ASC",
            )
            .context("failed to prepare message query")?;

        let rows = stmt.query_map(rusqlite::params![session_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, Option<i64>>(6)?,
            ))
        })?;

        for row in rows {
            let (
                message_id,
                role_raw,
                parts_json,
                model,
                created_at_raw,
                _updated_at_raw,
                finished_at_raw,
            ) = row?;

            let timestamp =
                parse_timestamp(&serde_json::Value::from(created_at_raw)).or(Some(created_at_raw));
            if let Some(ts) = timestamp {
                started_at = Some(started_at.map_or(ts, |current| current.min(ts)));
                ended_at = Some(ended_at.map_or(ts, |current| current.max(ts)));
            }

            if let Some(finished_raw) = finished_at_raw
                && let Some(finished_ts) = parse_timestamp(&serde_json::Value::from(finished_raw))
            {
                ended_at = Some(ended_at.map_or(finished_ts, |current| current.max(finished_ts)));
            }

            let raw_parts = serde_json::from_str::<serde_json::Value>(&parts_json)
                .unwrap_or_else(|_| serde_json::json!([]));
            let (content, tool_calls, tool_results) = parse_parts(&raw_parts);

            if let Some(model_name) = model.as_deref().filter(|m| !m.is_empty()) {
                *model_counts.entry(model_name.to_string()).or_insert(0) += 1;
            }

            messages.push(CanonicalMessage {
                idx: 0,
                role: normalize_role(&role_raw),
                content,
                timestamp,
                author: model.clone(),
                tool_calls,
                tool_results,
                extra: serde_json::json!({
                    "opencode_message_id": message_id,
                    "opencode_parts": raw_parts,
                }),
            });
        }

        reindex_messages(&mut messages);

        let title = (!title_raw.trim().is_empty())
            .then_some(title_raw)
            .or_else(|| {
                messages
                    .iter()
                    .find(|m| m.role == MessageRole::User)
                    .map(|m| truncate_title(&m.content, 80))
                    .filter(|t| !t.is_empty())
            });

        let model_name = model_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(name, _)| name);

        let source = Self::virtual_session_path(db_path, session_id);

        Ok(CanonicalSession {
            session_id: session_id.to_string(),
            provider_slug: "opencode".to_string(),
            workspace: Self::workspace_from_db_path(db_path),
            title,
            started_at,
            ended_at,
            messages,
            metadata: serde_json::json!({
                "opencode_db": db_path.display().to_string(),
                "opencode_schema": DbSchema::Legacy.metadata_tag(),
                "parent_session_id": parent_session_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
            }),
            source_path: source,
            model_name,
        })
    }
}

impl Provider for OpenCode {
    fn name(&self) -> &str {
        "OpenCode"
    }

    fn slug(&self) -> &str {
        "opencode"
    }

    fn cli_alias(&self) -> &str {
        "opc"
    }

    fn detect(&self) -> DetectionResult {
        let mut installed = false;
        let mut evidence = Vec::new();

        if which::which("opencode").is_ok() {
            installed = true;
            evidence.push("opencode binary found in PATH".to_string());
        }

        if let Some(env_path) = Self::env_db_path() {
            evidence.push(format!("env override target: {}", env_path.display()));
        }

        let dbs = Self::find_db_files();
        if !dbs.is_empty() {
            installed = true;
            evidence.push(format!("found {} opencode.db database(s)", dbs.len()));
        }

        trace!(provider = "opencode", installed, ?evidence, "detection");
        DetectionResult {
            installed,
            version: None,
            evidence,
        }
    }

    fn session_roots(&self) -> Vec<PathBuf> {
        Self::find_db_files()
    }

    fn owns_session(&self, session_id: &str) -> Option<PathBuf> {
        for db_path in Self::find_db_files() {
            let Ok(conn) = Self::open_db(&db_path) else {
                continue;
            };

            let schema = match Self::detect_schema(&conn, &db_path) {
                Ok(schema) => schema,
                Err(mismatch) => {
                    tracing::warn!("skipping unreadable OpenCode DB: {mismatch}");
                    continue;
                }
            };

            if Self::session_exists(&conn, schema, session_id) {
                let virtual_path = Self::virtual_session_path(&db_path, session_id);
                debug!(
                    db = %db_path.display(),
                    session = %virtual_path.display(),
                    session_id,
                    "found OpenCode session"
                );
                return Some(virtual_path);
            }
        }
        None
    }

    fn read_session(&self, path: &Path) -> anyhow::Result<CanonicalSession> {
        debug!(path = %path.display(), "reading OpenCode session");

        // Virtual path (`.../opencode.db/<encoded-session-id>`) from discovery.
        if let Some((db_path, session_id)) = Self::parse_virtual_path(path) {
            let conn = Self::open_db(&db_path)?;
            return Self::read_session_by_id(&conn, &db_path, &session_id);
        }

        // Direct DB path (`.../opencode.db`) — choose newest root session.
        let conn = Self::open_db(path)?;
        // Fail loudly on a schema mismatch instead of reporting an empty DB.
        let schema = Self::detect_schema(&conn, path)?;
        let Some(session_id) = Self::newest_root_session_id(&conn, schema) else {
            anyhow::bail!("no OpenCode sessions found in {}", path.display());
        };
        Self::read_session_by_id(&conn, path, &session_id)
    }

    fn write_session(
        &self,
        session: &CanonicalSession,
        opts: &WriteOptions,
    ) -> anyhow::Result<WrittenSession> {
        let db_path = Self::choose_target_db_path(session)?;
        let mut conn = Self::open_db_rw(&db_path)?;

        // Never graft the legacy plural tables into a live OpenCode 1.x DB:
        // its `session`/`message`/`part` rows are projections of an event
        // log, so rows inserted behind its back would not be picked up and
        // would leave the DB with two disagreeing schemas.
        if matches!(Self::detect_schema(&conn, &db_path), Ok(DbSchema::V1)) {
            anyhow::bail!(
                "OpenCode DB {} uses the 1.x event-sourced schema (session/message/part); \
                 casr can read it but does not write into it. Point OPENCODE_DB_PATH at a \
                 separate database, or use a different target provider.",
                db_path.display()
            );
        }
        Self::ensure_schema(&conn)?;

        let has_count_trigger =
            Self::trigger_exists(&conn, "update_session_message_count_on_insert");

        // Derive a STABLE target id from the source session so re-converting the
        // same session targets the same row (matching the clawdbot/cursor/pi_agent
        // idiom). This makes `--force` meaningful: without a stable id every run
        // would silently create an orphaned duplicate row, and with a colliding id
        // the INSERT would otherwise fail on the PRIMARY KEY. Fall back to a random
        // UUID only when the source has no id.
        let target_session_id = if session.session_id.is_empty() {
            uuid::Uuid::new_v4().to_string()
        } else {
            session.session_id.clone()
        };

        // Honor `--force`: if the target session already exists, either overwrite
        // it (delete-then-insert; `ON DELETE CASCADE` clears messages/files) or
        // return a clean conflict error, matching the cursor provider's behavior.
        if Self::session_exists(&conn, DbSchema::Legacy, &target_session_id) {
            if opts.force {
                // `ensure_schema` already enabled `PRAGMA foreign_keys = ON` on
                // this connection, so deleting the session cascades to messages
                // and files. Delete dependents explicitly too, in case the live
                // DB predates the FK constraint or has the pragma disabled.
                let _ = conn.execute(
                    "DELETE FROM files WHERE session_id = ?1",
                    rusqlite::params![target_session_id],
                );
                let _ = conn.execute(
                    "DELETE FROM messages WHERE session_id = ?1",
                    rusqlite::params![target_session_id],
                );
                conn.execute(
                    "DELETE FROM sessions WHERE id = ?1",
                    rusqlite::params![target_session_id],
                )
                .context("failed to delete existing OpenCode session for --force overwrite")?;
            } else {
                return Err(crate::error::CasrError::SessionConflict {
                    session_id: target_session_id,
                    existing_path: db_path,
                }
                .into());
            }
        }

        let now = chrono::Utc::now().timestamp_millis();
        let created_at = session.started_at.unwrap_or(now);
        let updated_at = session.ended_at.unwrap_or(now);

        let title = session.title.clone().or_else(|| {
            session
                .messages
                .iter()
                .find(|m| m.role == MessageRole::User)
                .map(|m| truncate_title(&m.content, 80))
                .filter(|t| !t.is_empty())
        });
        let title = title.unwrap_or_else(|| "Converted session".to_string());

        let tx = conn.transaction().context("failed to begin transaction")?;

        tx.execute(
            "INSERT INTO sessions (
                id, parent_session_id, title, message_count, prompt_tokens, completion_tokens, cost,
                summary_message_id, updated_at, created_at
             ) VALUES (?1, NULL, ?2, ?3, 0, 0, 0.0, NULL, ?4, ?5)",
            rusqlite::params![
                target_session_id,
                title,
                if has_count_trigger {
                    0_i64
                } else {
                    i64::try_from(session.messages.len()).unwrap_or(i64::MAX)
                },
                updated_at,
                created_at,
            ],
        )
        .context("failed to insert OpenCode session")?;

        let default_model = session.model_name.clone();
        for msg in &session.messages {
            let message_id = uuid::Uuid::new_v4().to_string();
            let parts = build_parts(msg);
            let parts_json =
                serde_json::to_string(&parts).context("failed to serialize OpenCode parts")?;
            let timestamp = msg.timestamp.unwrap_or(created_at);
            let model = msg.author.clone().or_else(|| default_model.clone());

            tx.execute(
                "INSERT INTO messages (
                    id, session_id, role, parts, model, created_at, updated_at, finished_at
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL)",
                rusqlite::params![
                    message_id,
                    target_session_id,
                    role_to_opencode(&msg.role),
                    parts_json,
                    model,
                    timestamp,
                    timestamp,
                ],
            )
            .with_context(|| format!("failed to insert OpenCode message {}", msg.idx))?;
        }

        // If the DB has no count trigger, set message_count explicitly.
        if !has_count_trigger {
            tx.execute(
                "UPDATE sessions SET message_count = ?1 WHERE id = ?2",
                rusqlite::params![
                    i64::try_from(session.messages.len()).unwrap_or(i64::MAX),
                    target_session_id
                ],
            )
            .context("failed to update OpenCode session message_count")?;
        }

        tx.commit().context("failed to commit transaction")?;

        let virtual_path = Self::virtual_session_path(&db_path, &target_session_id);
        info!(
            session_id = target_session_id,
            path = %db_path.display(),
            messages = session.messages.len(),
            "OpenCode session written"
        );

        Ok(WrittenSession {
            paths: vec![virtual_path],
            session_id: target_session_id.clone(),
            resume_command: self.resume_command(&target_session_id),
            backup_path: None,
            warnings: Vec::new(),
        })
    }

    fn resume_command(&self, _session_id: &str) -> String {
        // OpenCode has no session-id-specific resume flag.
        "opencode".to_string()
    }

    fn list_sessions(&self) -> Option<Vec<(String, PathBuf)>> {
        let db_files = Self::find_db_files();
        if db_files.is_empty() {
            return Some(Vec::new());
        }

        let mut results = Vec::new();
        for db_path in &db_files {
            let Ok(conn) = Self::open_db(db_path) else {
                continue;
            };
            let schema = match Self::detect_schema(&conn, db_path) {
                Ok(schema) => schema,
                Err(mismatch) => {
                    // Surface the mismatch instead of silently listing 0
                    // sessions (the default log filter shows warnings, so
                    // this is visible in plain `casr list` output).
                    tracing::warn!("skipping unreadable OpenCode DB: {mismatch}");
                    continue;
                }
            };

            for session_id in Self::all_session_ids(&conn, schema) {
                let virtual_path = Self::virtual_session_path(db_path, &session_id);
                results.push((session_id, virtual_path));
            }
        }

        Some(results)
    }
}

fn dedup_existing_files(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen: BTreeSet<PathBuf> = BTreeSet::new();
    for path in paths {
        if path.is_file() {
            seen.insert(path);
        }
    }
    seen.into_iter().collect()
}

fn parse_tool_call_arguments(input: &str) -> serde_json::Value {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return serde_json::json!({});
    }
    serde_json::from_str(trimmed).unwrap_or_else(|_| serde_json::json!({ "input": input }))
}

fn parse_parts(parts: &serde_json::Value) -> (String, Vec<ToolCall>, Vec<ToolResult>) {
    let mut text_chunks: Vec<String> = Vec::new();
    let mut reasoning_chunks: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut tool_results: Vec<ToolResult> = Vec::new();

    let Some(items) = parts.as_array() else {
        return (String::new(), tool_calls, tool_results);
    };

    for item in items {
        let part_type = item
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        let data = item.get("data").unwrap_or(&serde_json::Value::Null);

        match part_type {
            "text" => {
                if let Some(text) = data.get("text").and_then(serde_json::Value::as_str)
                    && !text.trim().is_empty()
                {
                    text_chunks.push(text.to_string());
                }
            }
            "reasoning" => {
                if let Some(thinking) = data.get("thinking").and_then(serde_json::Value::as_str)
                    && !thinking.trim().is_empty()
                {
                    reasoning_chunks.push(thinking.to_string());
                }
            }
            "tool_call" => {
                let name = data
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .filter(|name| !name.is_empty())
                    .unwrap_or("tool_call")
                    .to_string();
                let id = data
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .filter(|id| !id.is_empty())
                    .map(ToString::to_string);
                let input = data
                    .get("input")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default();

                tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments: parse_tool_call_arguments(input),
                });
            }
            "tool_result" => {
                let content = data
                    .get("content")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let call_id = data
                    .get("tool_call_id")
                    .and_then(serde_json::Value::as_str)
                    .filter(|id| !id.is_empty())
                    .map(ToString::to_string);
                let is_error = data
                    .get("is_error")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);

                tool_results.push(ToolResult {
                    call_id,
                    content,
                    is_error,
                });
            }
            _ => {
                let fallback = flatten_content(data);
                if !fallback.trim().is_empty() {
                    text_chunks.push(fallback);
                }
            }
        }
    }

    let mut content = text_chunks.join("\n");
    if content.trim().is_empty() {
        content = reasoning_chunks.join("\n");
    }
    if content.trim().is_empty() {
        let result_texts: Vec<&str> = tool_results
            .iter()
            .map(|result| result.content.as_str())
            .filter(|text| !text.trim().is_empty())
            .collect();
        content = result_texts.join("\n");
    }

    (content, tool_calls, tool_results)
}

/// One row of the OpenCode 1.x `session` table, pulled by column name so
/// revisions that lack some columns still read.
#[derive(Debug)]
struct V1SessionRow {
    title: Option<String>,
    directory: Option<String>,
    parent_id: Option<String>,
    time_created: Option<i64>,
    time_updated: Option<i64>,
    cost: Option<f64>,
    tokens_input: Option<i64>,
    tokens_output: Option<i64>,
    tokens_reasoning: Option<i64>,
    slug: Option<String>,
    version: Option<String>,
    project_id: Option<String>,
    /// Raw JSON of the `model` column (`{"providerID":…,"modelID":…}` or
    /// `{"id":…}` depending on revision), if present.
    model_json: Option<String>,
}

/// Parse OpenCode 1.x part rows (the `part.data` JSON blobs, `id` grafted
/// in) into canonical content, tool calls and tool results.
///
/// 1.x parts are flat (`{type, …}`) rather than the legacy `{type, data}`
/// envelope, and the tool call and its result share one `tool` part whose
/// `state.status` is `pending` / `running` / `completed` / `error`:
///
/// - `text` → content (`ignored` parts are skipped)
/// - `reasoning` → content only when there is no text
/// - `tool` → a [`ToolCall`] (`callID`, `tool`, `state.input`) plus a
///   [`ToolResult`] once the state is `completed` (`state.output`) or
///   `error` (`state.error`, flagged as an error)
/// - `file` → a short `[file: …]` marker so attachments stay visible
/// - `step-start` / `step-finish` / `snapshot` / `patch` / `agent` and any
///   other bookkeeping part carry no conversational content and are skipped
fn parse_v1_parts(parts: &serde_json::Value) -> (String, Vec<ToolCall>, Vec<ToolResult>) {
    let mut text_chunks: Vec<String> = Vec::new();
    let mut reasoning_chunks: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut tool_results: Vec<ToolResult> = Vec::new();

    let Some(items) = parts.as_array() else {
        return (String::new(), tool_calls, tool_results);
    };

    for item in items {
        let part_type = item
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();

        match part_type {
            "text" => {
                let ignored = item
                    .get("ignored")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                if ignored {
                    continue;
                }
                if let Some(text) = item.get("text").and_then(serde_json::Value::as_str)
                    && !text.trim().is_empty()
                {
                    text_chunks.push(text.to_string());
                }
            }
            "reasoning" => {
                if let Some(text) = item.get("text").and_then(serde_json::Value::as_str)
                    && !text.trim().is_empty()
                {
                    reasoning_chunks.push(text.to_string());
                }
            }
            "tool" => {
                let state = item.get("state").unwrap_or(&serde_json::Value::Null);
                let name = item
                    .get("tool")
                    .and_then(serde_json::Value::as_str)
                    .filter(|name| !name.is_empty())
                    .unwrap_or("tool")
                    .to_string();
                let call_id = item
                    .get("callID")
                    .and_then(serde_json::Value::as_str)
                    .filter(|id| !id.is_empty())
                    .map(ToString::to_string);
                let arguments = match state.get("input") {
                    Some(serde_json::Value::String(raw)) => parse_tool_call_arguments(raw),
                    Some(serde_json::Value::Null) | None => serde_json::json!({}),
                    Some(other) => other.clone(),
                };
                tool_calls.push(ToolCall {
                    id: call_id.clone(),
                    name,
                    arguments,
                });

                let status = state
                    .get("status")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default();
                match status {
                    "completed" => {
                        let content = state.get("output").map(flatten_content).unwrap_or_default();
                        tool_results.push(ToolResult {
                            call_id,
                            content,
                            is_error: false,
                        });
                    }
                    "error" => {
                        let content = state.get("error").map(flatten_content).unwrap_or_default();
                        tool_results.push(ToolResult {
                            call_id,
                            content,
                            is_error: true,
                        });
                    }
                    // `pending` / `running`: the call exists but has no result yet.
                    _ => {}
                }
            }
            "file" => {
                let label = item
                    .get("filename")
                    .or_else(|| item.get("url"))
                    .and_then(serde_json::Value::as_str)
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or("attachment");
                text_chunks.push(format!("[file: {label}]"));
            }
            _ => {}
        }
    }

    let mut content = text_chunks.join("\n");
    if content.trim().is_empty() {
        content = reasoning_chunks.join("\n");
    }
    if content.trim().is_empty() {
        let result_texts: Vec<&str> = tool_results
            .iter()
            .map(|result| result.content.as_str())
            .filter(|text| !text.trim().is_empty())
            .collect();
        content = result_texts.join("\n");
    }

    (content, tool_calls, tool_results)
}

fn build_parts(message: &CanonicalMessage) -> serde_json::Value {
    let mut parts = Vec::new();

    if !message.content.trim().is_empty() {
        parts.push(serde_json::json!({
            "type": "text",
            "data": { "text": message.content },
        }));
    }

    for call in &message.tool_calls {
        let input = if let Some(s) = call.arguments.as_str() {
            s.to_string()
        } else {
            serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_string())
        };

        parts.push(serde_json::json!({
            "type": "tool_call",
            "data": {
                "id": call.id.clone().unwrap_or_default(),
                "name": call.name,
                "input": input,
                "type": "function",
                "finished": true
            }
        }));
    }

    for result in &message.tool_results {
        parts.push(serde_json::json!({
            "type": "tool_result",
            "data": {
                "tool_call_id": result.call_id.clone().unwrap_or_default(),
                "name": "tool",
                "content": result.content,
                "metadata": "",
                "is_error": result.is_error
            }
        }));
    }

    serde_json::Value::Array(parts)
}

fn role_to_opencode(role: &MessageRole) -> &str {
    match role {
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
        MessageRole::System => "system",
        MessageRole::Other(role) => role.as_str(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::Provider;
    use std::sync::{LazyLock, Mutex};

    static OPENCODE_ENV: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    struct CwdGuard {
        original: PathBuf,
    }

    impl CwdGuard {
        fn change_to(path: &Path) -> Self {
            let original = std::env::current_dir().expect("read current dir");
            std::env::set_current_dir(path).expect("set current dir");
            Self { original }
        }
    }

    impl Drop for CwdGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.original);
        }
    }

    fn sample_session(workspace: &Path) -> CanonicalSession {
        CanonicalSession {
            session_id: "source-session".to_string(),
            provider_slug: "claude-code".to_string(),
            workspace: Some(workspace.to_path_buf()),
            title: Some("Fix OpenCode adapter".to_string()),
            started_at: Some(1_700_000_000_000),
            ended_at: Some(1_700_000_010_000),
            messages: vec![
                CanonicalMessage {
                    idx: 0,
                    role: MessageRole::User,
                    content: "Please inspect src/main.rs".to_string(),
                    timestamp: Some(1_700_000_000_000),
                    author: None,
                    tool_calls: vec![],
                    tool_results: vec![],
                    extra: serde_json::json!({}),
                },
                CanonicalMessage {
                    idx: 1,
                    role: MessageRole::Assistant,
                    content: "Inspecting now.".to_string(),
                    timestamp: Some(1_700_000_005_000),
                    author: Some("gpt-5".to_string()),
                    tool_calls: vec![ToolCall {
                        id: Some("call-1".to_string()),
                        name: "Read".to_string(),
                        arguments: serde_json::json!({"path":"src/main.rs"}),
                    }],
                    tool_results: vec![ToolResult {
                        call_id: Some("call-1".to_string()),
                        content: "Read complete".to_string(),
                        is_error: false,
                    }],
                    extra: serde_json::json!({}),
                },
            ],
            metadata: serde_json::json!({}),
            source_path: workspace.join("source.jsonl"),
            model_name: Some("gpt-5".to_string()),
        }
    }

    #[test]
    fn provider_metadata_and_resume_command() {
        let provider = OpenCode;
        assert_eq!(provider.name(), "OpenCode");
        assert_eq!(provider.slug(), "opencode");
        assert_eq!(provider.cli_alias(), "opc");
        assert_eq!(
            <OpenCode as Provider>::resume_command(&provider, "sid"),
            "opencode"
        );
    }

    #[test]
    fn virtual_path_round_trip() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(workspace.join(".opencode")).expect("data dir");
        let db = workspace.join(".opencode/opencode.db");
        std::fs::write(&db, "").expect("touch db file");

        let sid = "abc-123";
        let virtual_path = OpenCode::virtual_session_path(&db, sid);
        let parsed = OpenCode::parse_virtual_path(&virtual_path).expect("should parse");
        assert_eq!(parsed.0, db.as_path());
        assert_eq!(parsed.1, sid);
    }

    #[test]
    fn writer_reader_roundtrip_preserves_core_content() {
        let _lock = OPENCODE_ENV.lock().expect("mutex lock");
        let tmp = tempfile::tempdir().expect("tmpdir");
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).expect("workspace dir");
        let _cwd = CwdGuard::change_to(&workspace);

        let source = sample_session(&workspace);
        let written = OpenCode
            .write_session(&source, &WriteOptions { force: false })
            .expect("write should succeed");

        assert_eq!(written.resume_command, "opencode");
        assert_eq!(written.paths.len(), 1);
        let db_path = written
            .paths
            .first()
            .and_then(|p| p.parent())
            .expect("virtual path parent");
        assert!(db_path.is_file(), "db file should exist");

        let readback = OpenCode
            .read_session(&written.paths[0])
            .expect("readback should succeed");

        assert_eq!(readback.provider_slug, "opencode");
        assert_eq!(readback.messages.len(), source.messages.len());
        assert_eq!(readback.messages[0].role, MessageRole::User);
        assert_eq!(readback.messages[0].content, source.messages[0].content);
        assert_eq!(readback.messages[1].role, MessageRole::Assistant);
        assert_eq!(readback.messages[1].content, source.messages[1].content);
        // The writer canonicalizes the workspace so the DB it creates is the
        // same file discovery reports; the round trip therefore preserves the
        // *directory*, not the byte spelling (macOS: `/var` vs `/private/var`).
        assert_eq!(
            readback
                .workspace
                .as_deref()
                .and_then(|w| w.canonicalize().ok()),
            workspace.canonicalize().ok(),
            "round trip should preserve the workspace directory"
        );
        // The target id is now derived stably from the source session id so that
        // re-conversion is idempotent and `--force` can overwrite in place.
        assert_eq!(readback.session_id, source.session_id);
    }

    /// Regression for #14: writing the same OpenCode session twice must fail
    /// without `--force` (clean SessionConflict, not a raw SQLite duplicate-key
    /// error) and succeed with `--force`, overwriting the existing row in place
    /// rather than orphaning a duplicate.
    #[test]
    fn write_twice_with_force_overwrites_in_place() {
        let _lock = OPENCODE_ENV.lock().expect("mutex lock");
        let tmp = tempfile::tempdir().expect("tmpdir");
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).expect("workspace dir");
        let _cwd = CwdGuard::change_to(&workspace);

        let source = sample_session(&workspace);

        // First write succeeds.
        let first = OpenCode
            .write_session(&source, &WriteOptions { force: false })
            .expect("first write should succeed");
        let db_path = first.paths[0].parent().expect("db parent").to_path_buf();

        // Second write WITHOUT force must be a clean conflict, not a panic or a
        // raw "failed to insert OpenCode session" error.
        let conflict = OpenCode
            .write_session(&source, &WriteOptions { force: false })
            .expect_err("second write without --force should conflict");
        match conflict.downcast_ref::<crate::error::CasrError>() {
            Some(crate::error::CasrError::SessionConflict { session_id, .. }) => {
                assert_eq!(session_id, &source.session_id);
            }
            other => panic!("expected SessionConflict, got {other:?}"),
        }

        // Second write WITH force succeeds and overwrites in place.
        let second = OpenCode
            .write_session(&source, &WriteOptions { force: true })
            .expect("force write should succeed");

        // Same stable target id both times.
        assert_eq!(first.session_id, second.session_id);
        assert_eq!(second.session_id, source.session_id);

        // Exactly one session row and no orphaned/duplicated message rows.
        let conn = OpenCode::open_db(&db_path).expect("open db");
        let session_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))
            .expect("count sessions");
        assert_eq!(session_count, 1, "force must overwrite, not duplicate");

        let message_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM messages", [], |r| r.get(0))
            .expect("count messages");
        assert_eq!(
            message_count,
            source.messages.len() as i64,
            "messages from the prior write must be replaced, not accumulated"
        );

        // The overwritten session still reads back cleanly.
        let readback = OpenCode
            .read_session(&second.paths[0])
            .expect("readback after force overwrite");
        assert_eq!(readback.messages.len(), source.messages.len());
    }

    #[test]
    fn owns_session_returns_virtual_path() {
        let _lock = OPENCODE_ENV.lock().expect("mutex lock");
        let tmp = tempfile::tempdir().expect("tmpdir");
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).expect("workspace dir");
        let _cwd = CwdGuard::change_to(&workspace);

        let source = sample_session(&workspace);
        let written = OpenCode
            .write_session(&source, &WriteOptions { force: false })
            .expect("write should succeed");
        let found = OpenCode.owns_session(&written.session_id);

        assert_eq!(found.as_deref(), Some(written.paths[0].as_path()));
    }

    #[test]
    fn read_session_from_db_path_returns_latest_root_session() {
        let _lock = OPENCODE_ENV.lock().expect("mutex lock");
        let tmp = tempfile::tempdir().expect("tmpdir");
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).expect("workspace dir");
        let _cwd = CwdGuard::change_to(&workspace);

        // Distinct source ids so both land as separate root sessions in one DB
        // (target ids are now derived stably from the source session id).
        let mut first = sample_session(&workspace);
        first.session_id = "older-source".to_string();
        first.title = Some("Older Session".to_string());
        first.started_at = Some(1_700_000_000_000);
        let _first_written = OpenCode
            .write_session(&first, &WriteOptions { force: false })
            .expect("first write");

        let mut second = sample_session(&workspace);
        second.session_id = "newer-source".to_string();
        second.title = Some("Newer Session".to_string());
        second.started_at = Some(1_800_000_000_000);
        let second_written = OpenCode
            .write_session(&second, &WriteOptions { force: false })
            .expect("second write");

        let db_path = second_written
            .paths
            .first()
            .and_then(|p| p.parent())
            .expect("db path parent")
            .to_path_buf();

        let read_latest = OpenCode
            .read_session(&db_path)
            .expect("read from db should pick latest");
        assert_eq!(read_latest.title.as_deref(), Some("Newer Session"));
    }

    #[test]
    fn detect_reports_db_presence() {
        let _lock = OPENCODE_ENV.lock().expect("mutex lock");
        let tmp = tempfile::tempdir().expect("tmpdir");
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).expect("workspace dir");
        let _cwd = CwdGuard::change_to(&workspace);

        let source = sample_session(&workspace);
        OpenCode
            .write_session(&source, &WriteOptions { force: false })
            .expect("write should succeed");

        let detection = OpenCode.detect();
        assert!(
            detection.installed,
            "db presence should mark provider installed"
        );
        assert!(
            detection
                .evidence
                .iter()
                .any(|ev| ev.contains("opencode.db")),
            "evidence should include db detection"
        );
    }

    #[test]
    fn parse_parts_extracts_tool_calls_and_results() {
        let raw = serde_json::json!([
            {"type":"text","data":{"text":"hello"}},
            {"type":"tool_call","data":{"id":"c1","name":"Read","input":"{\"path\":\"src/main.rs\"}","type":"function","finished":true}},
            {"type":"tool_result","data":{"tool_call_id":"c1","name":"Read","content":"ok","metadata":"","is_error":false}}
        ]);

        let (content, tool_calls, tool_results) = parse_parts(&raw);
        assert_eq!(content, "hello");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "Read");
        assert_eq!(tool_results.len(), 1);
        assert_eq!(tool_results[0].content, "ok");
    }

    // ── parse_parts edge cases ──────────────────────────────────────────

    #[test]
    fn parse_parts_reasoning_content_when_no_text() {
        let raw = serde_json::json!([
            {"type":"reasoning","data":{"thinking":"Let me analyze this problem step by step."}}
        ]);
        let (content, tool_calls, tool_results) = parse_parts(&raw);
        assert_eq!(content, "Let me analyze this problem step by step.");
        assert!(tool_calls.is_empty());
        assert!(tool_results.is_empty());
    }

    #[test]
    fn parse_parts_text_preferred_over_reasoning() {
        let raw = serde_json::json!([
            {"type":"text","data":{"text":"The answer is 42."}},
            {"type":"reasoning","data":{"thinking":"Hmm, thinking..."}}
        ]);
        let (content, _, _) = parse_parts(&raw);
        assert_eq!(content, "The answer is 42.");
    }

    #[test]
    fn parse_parts_empty_array() {
        let raw = serde_json::json!([]);
        let (content, tool_calls, tool_results) = parse_parts(&raw);
        assert!(content.is_empty());
        assert!(tool_calls.is_empty());
        assert!(tool_results.is_empty());
    }

    #[test]
    fn parse_parts_non_array_returns_empty() {
        let raw = serde_json::json!("just a string");
        let (content, tool_calls, tool_results) = parse_parts(&raw);
        assert!(content.is_empty());
        assert!(tool_calls.is_empty());
        assert!(tool_results.is_empty());
    }

    #[test]
    fn parse_parts_unknown_type_uses_fallback() {
        // Unknown part type with a "text" field in data → flatten_content extracts it.
        let raw = serde_json::json!([
            {"type":"custom_widget","data":"Some inline text from unknown part type"}
        ]);
        let (content, _, _) = parse_parts(&raw);
        assert_eq!(content, "Some inline text from unknown part type");
    }

    #[test]
    fn parse_parts_tool_result_fallback_when_no_text_or_reasoning() {
        let raw = serde_json::json!([
            {"type":"tool_result","data":{"tool_call_id":"c1","content":"file contents here","is_error":false}}
        ]);
        let (content, _, tool_results) = parse_parts(&raw);
        assert_eq!(content, "file contents here");
        assert_eq!(tool_results.len(), 1);
    }

    #[test]
    fn parse_parts_multiple_text_chunks_joined() {
        let raw = serde_json::json!([
            {"type":"text","data":{"text":"First part."}},
            {"type":"text","data":{"text":"Second part."}}
        ]);
        let (content, _, _) = parse_parts(&raw);
        assert_eq!(content, "First part.\nSecond part.");
    }

    #[test]
    fn parse_parts_skips_empty_text() {
        let raw = serde_json::json!([
            {"type":"text","data":{"text":"  "}},
            {"type":"text","data":{"text":"real content"}}
        ]);
        let (content, _, _) = parse_parts(&raw);
        assert_eq!(content, "real content");
    }

    #[test]
    fn parse_parts_tool_call_missing_name_defaults() {
        let raw = serde_json::json!([
            {"type":"tool_call","data":{"id":"c1","name":"","input":"{}","type":"function"}}
        ]);
        let (_, tool_calls, _) = parse_parts(&raw);
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "tool_call");
    }

    #[test]
    fn parse_parts_tool_call_no_id_is_none() {
        let raw = serde_json::json!([
            {"type":"tool_call","data":{"name":"Bash","input":"{\"cmd\":\"ls\"}"}}
        ]);
        let (_, tool_calls, _) = parse_parts(&raw);
        assert_eq!(tool_calls.len(), 1);
        assert!(tool_calls[0].id.is_none());
    }

    #[test]
    fn parse_parts_tool_result_error_flag() {
        let raw = serde_json::json!([
            {"type":"tool_result","data":{"tool_call_id":"c1","content":"command failed","is_error":true}}
        ]);
        let (_, _, tool_results) = parse_parts(&raw);
        assert_eq!(tool_results.len(), 1);
        assert!(tool_results[0].is_error);
    }

    // ── parse_tool_call_arguments ───────────────────────────────────────

    #[test]
    fn parse_tool_call_arguments_valid_json() {
        let result = parse_tool_call_arguments(r#"{"path":"src/main.rs"}"#);
        assert_eq!(result["path"], "src/main.rs");
    }

    #[test]
    fn parse_tool_call_arguments_empty_returns_empty_object() {
        let result = parse_tool_call_arguments("");
        assert_eq!(result, serde_json::json!({}));
    }

    #[test]
    fn parse_tool_call_arguments_invalid_json_wraps_in_input() {
        let result = parse_tool_call_arguments("not json");
        assert_eq!(result["input"], "not json");
    }

    // ── role_to_opencode ────────────────────────────────────────────────

    #[test]
    fn role_to_opencode_all_variants() {
        assert_eq!(role_to_opencode(&MessageRole::User), "user");
        assert_eq!(role_to_opencode(&MessageRole::Assistant), "assistant");
        assert_eq!(role_to_opencode(&MessageRole::Tool), "tool");
        assert_eq!(role_to_opencode(&MessageRole::System), "system");
        assert_eq!(
            role_to_opencode(&MessageRole::Other("custom".to_string())),
            "custom"
        );
    }

    // ── build_parts ─────────────────────────────────────────────────────

    #[test]
    fn build_parts_text_only() {
        let msg = CanonicalMessage {
            idx: 0,
            role: MessageRole::User,
            content: "Hello world".to_string(),
            timestamp: None,
            author: None,
            tool_calls: vec![],
            tool_results: vec![],
            extra: serde_json::json!({}),
        };
        let parts = build_parts(&msg);
        let arr = parts.as_array().expect("should be array");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[0]["data"]["text"], "Hello world");
    }

    #[test]
    fn build_parts_with_tool_call_and_result() {
        let msg = CanonicalMessage {
            idx: 0,
            role: MessageRole::Assistant,
            content: "Let me check.".to_string(),
            timestamp: None,
            author: None,
            tool_calls: vec![ToolCall {
                id: Some("tc-1".to_string()),
                name: "Bash".to_string(),
                arguments: serde_json::json!({"cmd": "ls"}),
            }],
            tool_results: vec![ToolResult {
                call_id: Some("tc-1".to_string()),
                content: "file1.rs\nfile2.rs".to_string(),
                is_error: false,
            }],
            extra: serde_json::json!({}),
        };
        let parts = build_parts(&msg);
        let arr = parts.as_array().expect("should be array");
        assert_eq!(arr.len(), 3); // text + tool_call + tool_result
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[1]["type"], "tool_call");
        assert_eq!(arr[1]["data"]["name"], "Bash");
        assert_eq!(arr[2]["type"], "tool_result");
        assert!(!arr[2]["data"]["is_error"].as_bool().unwrap());
    }

    #[test]
    fn build_parts_empty_content_skips_text() {
        let msg = CanonicalMessage {
            idx: 0,
            role: MessageRole::Tool,
            content: "  ".to_string(),
            timestamp: None,
            author: None,
            tool_calls: vec![],
            tool_results: vec![ToolResult {
                call_id: Some("c1".to_string()),
                content: "result".to_string(),
                is_error: false,
            }],
            extra: serde_json::json!({}),
        };
        let parts = build_parts(&msg);
        let arr = parts.as_array().expect("array");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["type"], "tool_result");
    }

    // ── workspace_from_db_path ──────────────────────────────────────────

    #[test]
    fn workspace_from_db_path_valid() {
        let path = PathBuf::from("/home/user/project/.opencode/opencode.db");
        let ws = OpenCode::workspace_from_db_path(&path);
        assert_eq!(ws, Some(PathBuf::from("/home/user/project")));
    }

    #[test]
    fn workspace_from_db_path_wrong_dirname_returns_none() {
        let path = PathBuf::from("/home/user/project/data/opencode.db");
        let ws = OpenCode::workspace_from_db_path(&path);
        assert!(ws.is_none());
    }

    #[test]
    fn workspace_from_db_path_root_opencode_returns_none() {
        let path = PathBuf::from("/.opencode/opencode.db");
        let ws = OpenCode::workspace_from_db_path(&path);
        assert_eq!(ws, Some(PathBuf::from("/")));
    }

    // ── virtual_path_special_characters ─────────────────────────────────

    #[test]
    fn virtual_path_encodes_special_characters() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let db = tmp.path().join("opencode.db");
        std::fs::write(&db, "").expect("touch");

        let sid = "session/with spaces&special=chars";
        let vp = OpenCode::virtual_session_path(&db, sid);
        let (parsed_db, parsed_sid) = OpenCode::parse_virtual_path(&vp).expect("parse");
        assert_eq!(parsed_db, db);
        assert_eq!(parsed_sid, sid);
    }

    // ── writer edge cases ───────────────────────────────────────────────

    #[test]
    fn writer_no_title_generates_from_first_user_message() {
        let _lock = OPENCODE_ENV.lock().expect("mutex lock");
        let tmp = tempfile::tempdir().expect("tmpdir");
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).expect("workspace dir");
        let _cwd = CwdGuard::change_to(&workspace);

        let mut session = sample_session(&workspace);
        session.title = None;

        let written = OpenCode
            .write_session(&session, &WriteOptions { force: false })
            .expect("write");
        let readback = OpenCode.read_session(&written.paths[0]).expect("readback");

        // Title should be derived from first user message
        assert!(readback.title.is_some());
        let title = readback.title.unwrap();
        assert!(title.contains("inspect"));
    }

    #[test]
    fn writer_no_timestamps_uses_current_time() {
        let _lock = OPENCODE_ENV.lock().expect("mutex lock");
        let tmp = tempfile::tempdir().expect("tmpdir");
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).expect("workspace dir");
        let _cwd = CwdGuard::change_to(&workspace);

        let mut session = sample_session(&workspace);
        session.started_at = None;
        session.ended_at = None;
        for msg in &mut session.messages {
            msg.timestamp = None;
        }

        let written = OpenCode
            .write_session(&session, &WriteOptions { force: false })
            .expect("write");
        let readback = OpenCode.read_session(&written.paths[0]).expect("readback");

        assert!(readback.started_at.is_some());
        assert!(readback.ended_at.is_some());
    }

    #[test]
    fn writer_model_name_propagated_to_messages() {
        let _lock = OPENCODE_ENV.lock().expect("mutex lock");
        let tmp = tempfile::tempdir().expect("tmpdir");
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).expect("workspace dir");
        let _cwd = CwdGuard::change_to(&workspace);

        let session = sample_session(&workspace);
        let written = OpenCode
            .write_session(&session, &WriteOptions { force: false })
            .expect("write");
        let readback = OpenCode.read_session(&written.paths[0]).expect("readback");

        // The model_name should be detected from message authors
        assert!(readback.model_name.is_some());
    }

    // ── reader edge cases ───────────────────────────────────────────────

    #[test]
    fn reader_metadata_includes_token_counts() {
        let _lock = OPENCODE_ENV.lock().expect("mutex lock");
        let tmp = tempfile::tempdir().expect("tmpdir");
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).expect("workspace dir");
        let _cwd = CwdGuard::change_to(&workspace);

        let session = sample_session(&workspace);
        let written = OpenCode
            .write_session(&session, &WriteOptions { force: false })
            .expect("write");
        let readback = OpenCode.read_session(&written.paths[0]).expect("readback");

        // Metadata should include OpenCode-specific fields
        assert!(readback.metadata.get("opencode_db").is_some());
        assert!(readback.metadata.get("prompt_tokens").is_some());
        assert!(readback.metadata.get("completion_tokens").is_some());
        assert!(readback.metadata.get("cost").is_some());
    }

    #[test]
    fn reader_message_extra_has_opencode_fields() {
        let _lock = OPENCODE_ENV.lock().expect("mutex lock");
        let tmp = tempfile::tempdir().expect("tmpdir");
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).expect("workspace dir");
        let _cwd = CwdGuard::change_to(&workspace);

        let session = sample_session(&workspace);
        let written = OpenCode
            .write_session(&session, &WriteOptions { force: false })
            .expect("write");
        let readback = OpenCode.read_session(&written.paths[0]).expect("readback");

        for msg in &readback.messages {
            assert!(
                msg.extra.get("opencode_message_id").is_some(),
                "each message should have opencode_message_id in extra"
            );
            assert!(
                msg.extra.get("opencode_parts").is_some(),
                "each message should have opencode_parts in extra"
            );
        }
    }

    // ── dedup_existing_files ────────────────────────────────────────────

    #[test]
    fn dedup_existing_files_removes_duplicates_and_nonexistent() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let file1 = tmp.path().join("a.db");
        let file2 = tmp.path().join("b.db");
        std::fs::write(&file1, "").expect("touch");
        std::fs::write(&file2, "").expect("touch");

        let input = vec![
            file1.clone(),
            file2.clone(),
            file1.clone(),                     // duplicate
            tmp.path().join("nonexistent.db"), // doesn't exist
        ];
        let result = dedup_existing_files(input);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&file1));
        assert!(result.contains(&file2));
    }

    #[test]
    fn dedup_existing_files_empty_input() {
        let result = dedup_existing_files(Vec::new());
        assert!(result.is_empty());
    }

    // ── list_sessions ───────────────────────────────────────────────────

    #[test]
    fn list_sessions_returns_all_sessions_from_db() {
        let _lock = OPENCODE_ENV.lock().expect("mutex lock");
        let tmp = tempfile::tempdir().expect("tmpdir");
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).expect("workspace dir");
        let _cwd = CwdGuard::change_to(&workspace);

        // Write two distinct sessions (distinct source ids → distinct rows)
        let mut first = sample_session(&workspace);
        first.session_id = "first-source".to_string();
        first.title = Some("First Session".to_string());
        first.started_at = Some(1_700_000_000_000);
        let first_written = OpenCode
            .write_session(&first, &WriteOptions { force: false })
            .expect("first write");

        let mut second = sample_session(&workspace);
        second.session_id = "second-source".to_string();
        second.title = Some("Second Session".to_string());
        second.started_at = Some(1_800_000_000_000);
        let second_written = OpenCode
            .write_session(&second, &WriteOptions { force: false })
            .expect("second write");

        let listed = OpenCode.list_sessions().expect("should return Some");
        assert!(
            listed.len() >= 2,
            "expected at least 2 sessions, got {}",
            listed.len()
        );

        let ids: Vec<&str> = listed.iter().map(|(id, _)| id.as_str()).collect();
        assert!(
            ids.contains(&first_written.session_id.as_str()),
            "first session should be listed"
        );
        assert!(
            ids.contains(&second_written.session_id.as_str()),
            "second session should be listed"
        );
    }

    // ── OpenCode 1.x schema (issue #26) ─────────────────────────────────

    const V1_SESSION_ID: &str = "ses_9f2c4d81bbfe3aQwErTyUiOp";
    const V1_OLDER_SESSION_ID: &str = "ses_0a1b2c3d4e5f6aOlDeRoNe";

    /// Build a fixture DB carrying the OpenCode 1.x singular, event-sourced
    /// schema (`session`/`message`/`part`, `data` JSON blobs — no plural
    /// tables), populated with a two-turn conversation plus an older session.
    fn create_v1_schema_db(db_path: &Path, directory: &Path) {
        let conn = Connection::open(db_path).expect("create fixture db");
        conn.execute_batch(
            r#"
CREATE TABLE session (
    id TEXT PRIMARY KEY, project_id TEXT, parent_id TEXT, slug TEXT, directory TEXT,
    title TEXT, version TEXT, time_created INTEGER, time_updated INTEGER
);
CREATE TABLE message (
    id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, time_updated INTEGER, data TEXT
);
CREATE TABLE part (
    id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT, time_created INTEGER,
    time_updated INTEGER, data TEXT
);
CREATE TABLE event (id INTEGER PRIMARY KEY, type TEXT, payload TEXT);
"#,
        )
        .expect("create fixture schema");

        let dir = directory.display().to_string();
        conn.execute(
            "INSERT INTO session (id, project_id, parent_id, slug, directory, title, version, time_created, time_updated)
             VALUES (?1, 'prj_1', NULL, 'live-session', ?2, 'live session', '1.18.23', 1700000000000, 1700000020000)",
            rusqlite::params![V1_SESSION_ID, dir],
        )
        .expect("insert session");
        conn.execute(
            "INSERT INTO session (id, project_id, parent_id, slug, directory, title, version, time_created, time_updated)
             VALUES (?1, 'prj_1', NULL, 'older', ?2, 'older session', '1.18.23', 1600000000000, 1600000001000)",
            rusqlite::params![V1_OLDER_SESSION_ID, dir],
        )
        .expect("insert older session");

        let user_msg = serde_json::json!({
            "role": "user",
            "time": {"created": 1_700_000_001_000_i64},
            "agent": "build",
            "model": {"providerID": "anthropic", "modelID": "claude-sonnet-4"}
        });
        let assistant_msg = serde_json::json!({
            "role": "assistant",
            "time": {"created": 1_700_000_005_000_i64, "completed": 1_700_000_009_000_i64},
            "parentID": "msg_user1",
            "providerID": "anthropic",
            "modelID": "claude-sonnet-4",
            "mode": "build",
            "agent": "build",
            "path": {"cwd": dir, "root": dir},
            "cost": 0.01,
            "tokens": {"input": 10, "output": 20, "reasoning": 5, "cache": {"read": 0, "write": 0}}
        });
        conn.execute(
            "INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?1, ?2, 1700000001000, 1700000001000, ?3)",
            rusqlite::params![
                "msg_user1",
                V1_SESSION_ID,
                serde_json::to_string(&user_msg).unwrap()
            ],
        )
        .expect("insert user message");
        conn.execute(
            "INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?1, ?2, 1700000005000, 1700000009000, ?3)",
            rusqlite::params![
                "msg_asst1",
                V1_SESSION_ID,
                serde_json::to_string(&assistant_msg).unwrap()
            ],
        )
        .expect("insert assistant message");
        // The older session has one bare user message so it is a real session.
        conn.execute(
            "INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?1, ?2, 1600000000000, 1600000000000, ?3)",
            rusqlite::params![
                "msg_old1",
                V1_OLDER_SESSION_ID,
                serde_json::to_string(&user_msg).unwrap()
            ],
        )
        .expect("insert older message");

        let parts: Vec<(&str, &str, serde_json::Value)> = vec![
            (
                "prt_u1",
                "msg_user1",
                serde_json::json!({"type": "text", "text": "Please inspect src/main.rs"}),
            ),
            (
                "prt_a1",
                "msg_asst1",
                serde_json::json!({"type": "step-start"}),
            ),
            (
                "prt_a2",
                "msg_asst1",
                serde_json::json!({"type": "reasoning", "text": "Need to read the file first.", "time": {"start": 1_700_000_005_000_i64, "end": 1_700_000_006_000_i64}}),
            ),
            (
                "prt_a3",
                "msg_asst1",
                serde_json::json!({
                    "type": "tool",
                    "callID": "call_1",
                    "tool": "read",
                    "state": {
                        "status": "completed",
                        "input": {"filePath": "src/main.rs"},
                        "output": "fn main() {}",
                        "title": "src/main.rs",
                        "metadata": {},
                        "time": {"start": 1_700_000_006_000_i64, "end": 1_700_000_007_000_i64}
                    }
                }),
            ),
            (
                "prt_a4",
                "msg_asst1",
                serde_json::json!({"type": "text", "text": "Inspecting now."}),
            ),
            (
                "prt_a5",
                "msg_asst1",
                serde_json::json!({"type": "step-finish", "cost": 0.01, "tokens": {"input": 10, "output": 20, "reasoning": 5, "cache": {"read": 0, "write": 0}}}),
            ),
            (
                "prt_o1",
                "msg_old1",
                serde_json::json!({"type": "text", "text": "older prompt"}),
            ),
        ];
        for (id, message_id, data) in parts {
            let session_id = if message_id == "msg_old1" {
                V1_OLDER_SESSION_ID
            } else {
                V1_SESSION_ID
            };
            conn.execute(
                "INSERT INTO part (id, message_id, session_id, time_created, time_updated, data) VALUES (?1, ?2, ?3, 1700000001000, 1700000001000, ?4)",
                rusqlite::params![id, message_id, session_id, serde_json::to_string(&data).unwrap()],
            )
            .expect("insert part");
        }
    }

    #[test]
    fn detect_schema_recognizes_1x_layout() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let db_path = tmp.path().join("opencode.db");
        create_v1_schema_db(&db_path, tmp.path());
        let conn = OpenCode::open_db(&db_path).expect("open db");
        assert_eq!(
            OpenCode::detect_schema(&conn, &db_path).expect("1.x schema"),
            DbSchema::V1
        );
        assert_eq!(OpenCode::schema_mismatch(&conn, &db_path), None);
    }

    /// Regression for #26: a 1.x DB must actually be read — the id OpenCode
    /// itself prints for `opencode -s <id>` resolves to the conversation.
    #[test]
    fn read_session_by_virtual_path_on_1x_schema_reads_conversation() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let db_path = tmp.path().join("opencode.db");
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).expect("workspace dir");
        create_v1_schema_db(&db_path, &workspace);

        let virtual_path = OpenCode::virtual_session_path(&db_path, V1_SESSION_ID);
        let session = OpenCode
            .read_session(&virtual_path)
            .expect("1.x session must read");

        assert_eq!(session.session_id, V1_SESSION_ID);
        assert_eq!(session.provider_slug, "opencode");
        assert_eq!(session.title.as_deref(), Some("live session"));
        assert_eq!(session.workspace.as_deref(), Some(workspace.as_path()));
        assert_eq!(session.model_name.as_deref(), Some("claude-sonnet-4"));
        assert_eq!(session.started_at, Some(1_700_000_000_000));
        assert_eq!(session.ended_at, Some(1_700_000_020_000));
        assert_eq!(session.metadata["opencode_schema"], "v1");
        assert_eq!(session.metadata["prompt_tokens"], 0);
        assert_eq!(session.metadata["opencode_version"], "1.18.23");
        assert_eq!(session.source_path, virtual_path);

        assert_eq!(session.messages.len(), 2, "one user + one assistant turn");
        let user = &session.messages[0];
        assert_eq!(user.idx, 0);
        assert_eq!(user.role, MessageRole::User);
        assert_eq!(user.content, "Please inspect src/main.rs");
        assert_eq!(user.timestamp, Some(1_700_000_001_000));
        assert!(user.tool_calls.is_empty());

        let assistant = &session.messages[1];
        assert_eq!(assistant.idx, 1);
        assert_eq!(assistant.role, MessageRole::Assistant);
        assert_eq!(assistant.content, "Inspecting now.");
        assert_eq!(assistant.author.as_deref(), Some("claude-sonnet-4"));
        assert_eq!(assistant.timestamp, Some(1_700_000_005_000));
        assert_eq!(assistant.tool_calls.len(), 1);
        assert_eq!(assistant.tool_calls[0].name, "read");
        assert_eq!(assistant.tool_calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(assistant.tool_calls[0].arguments["filePath"], "src/main.rs");
        assert_eq!(assistant.tool_results.len(), 1);
        assert_eq!(assistant.tool_results[0].call_id.as_deref(), Some("call_1"));
        assert_eq!(assistant.tool_results[0].content, "fn main() {}");
        assert!(!assistant.tool_results[0].is_error);
        assert_eq!(assistant.extra["opencode_message_id"], "msg_asst1");
        assert_eq!(
            assistant.extra["opencode_parts"].as_array().map(Vec::len),
            Some(5),
            "raw 1.x parts are preserved in extra"
        );
    }

    #[test]
    fn read_session_on_1x_db_path_picks_newest_root_session() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let db_path = tmp.path().join("opencode.db");
        create_v1_schema_db(&db_path, tmp.path());

        let session = OpenCode
            .read_session(&db_path)
            .expect("direct db path must resolve to the newest root session");
        assert_eq!(session.session_id, V1_SESSION_ID);
    }

    #[test]
    fn read_session_on_1x_schema_unknown_id_fails_naming_session() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let db_path = tmp.path().join("opencode.db");
        create_v1_schema_db(&db_path, tmp.path());

        let virtual_path = OpenCode::virtual_session_path(&db_path, "ses_missing");
        let err = OpenCode
            .read_session(&virtual_path)
            .expect_err("unknown id must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("ses_missing"), "got: {msg}");
    }

    #[test]
    fn list_sessions_on_1x_schema_lists_every_session_newest_first() {
        let _lock = OPENCODE_ENV.lock().expect("mutex lock");
        let tmp = tempfile::tempdir().expect("tmpdir");
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(workspace.join(DATA_DIRNAME)).expect("data dir");
        let db_path = workspace.join(DATA_DIRNAME).join(DB_FILENAME);
        create_v1_schema_db(&db_path, &workspace);
        let _cwd = CwdGuard::change_to(&workspace);

        let listed = OpenCode.list_sessions().expect("should return Some");
        let ids: Vec<&str> = listed.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, vec![V1_SESSION_ID, V1_OLDER_SESSION_ID]);
        for (id, path) in &listed {
            assert!(
                path.ends_with(urlencoding::encode(id).as_ref()),
                "virtual path {} must end with the session id",
                path.display()
            );
        }
    }

    #[test]
    fn write_session_into_1x_db_is_refused() {
        let _lock = OPENCODE_ENV.lock().expect("mutex lock");
        let tmp = tempfile::tempdir().expect("tmpdir");
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(workspace.join(DATA_DIRNAME)).expect("data dir");
        let db_path = workspace.join(DATA_DIRNAME).join(DB_FILENAME);
        create_v1_schema_db(&db_path, &workspace);
        let _cwd = CwdGuard::change_to(&workspace);

        let err = OpenCode
            .write_session(&sample_session(&workspace), &WriteOptions { force: false })
            .expect_err("writing into a live 1.x DB must be refused");
        let msg = format!("{err:#}");
        assert!(msg.contains("1.x"), "got: {msg}");

        // The refusal must happen before any schema is grafted on.
        let conn = OpenCode::open_db(&db_path).expect("open db");
        assert!(!OpenCode::table_exists(&conn, "sessions"));
        assert!(!OpenCode::table_exists(&conn, "messages"));
    }

    /// A DB that matches neither schema must produce a loud diagnostic naming
    /// the missing tables and the tables actually found — not a silent
    /// "0 sessions".
    #[test]
    fn read_session_on_partial_1x_schema_fails_loudly_naming_missing_tables() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let db_path = tmp.path().join("opencode.db");
        let conn = Connection::open(&db_path).expect("create fixture db");
        conn.execute_batch(
            r#"
CREATE TABLE session (id TEXT PRIMARY KEY, title TEXT, time_created INTEGER);
CREATE TABLE event (id INTEGER PRIMARY KEY, type TEXT, payload TEXT);
INSERT INTO session (id, title, time_created) VALUES ('ses_partial', 'partial', 1700000000000);
"#,
        )
        .expect("populate fixture schema");
        drop(conn);

        let err = OpenCode
            .read_session(&db_path)
            .expect_err("partial schema must not read as an empty DB");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("closest is the 1.x schema"),
            "error must identify the closest schema, got: {msg}"
        );
        assert!(
            msg.contains("missing table(s): message, part"),
            "error must name the missing 1.x tables, got: {msg}"
        );
        assert!(
            msg.contains("tables present: event, session"),
            "error must list the tables actually present, got: {msg}"
        );
        assert!(
            !msg.contains("no OpenCode sessions found"),
            "must not be misreported as an empty DB, got: {msg}"
        );

        let virtual_path = OpenCode::virtual_session_path(&db_path, "ses_partial");
        let err = OpenCode
            .read_session(&virtual_path)
            .expect_err("partial schema must fail loudly by virtual path too");
        assert!(format!("{err:#}").contains("missing table(s): message, part"));
    }

    #[test]
    fn read_session_on_empty_db_fails_naming_legacy_tables() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let db_path = tmp.path().join("opencode.db");
        let conn = Connection::open(&db_path).expect("create fixture db");
        conn.execute_batch("CREATE TABLE unrelated (id INTEGER PRIMARY KEY);")
            .expect("populate fixture schema");
        drop(conn);

        let err = OpenCode
            .read_session(&db_path)
            .expect_err("unknown schema must fail loudly");
        let msg = format!("{err:#}");
        assert!(msg.contains("closest is the legacy schema"), "got: {msg}");
        assert!(
            msg.contains("missing table(s): sessions, messages"),
            "got: {msg}"
        );
    }

    // ── parse_v1_parts ──────────────────────────────────────────────────

    #[test]
    fn parse_v1_parts_tool_completed_yields_call_and_result() {
        let parts = serde_json::json!([
            {"type": "step-start"},
            {"type": "text", "text": "Reading."},
            {"type": "tool", "callID": "call_1", "tool": "bash",
             "state": {"status": "completed", "input": {"command": "ls"}, "output": "a\nb", "title": "ls"}},
            {"type": "step-finish", "cost": 0.0}
        ]);
        let (content, calls, results) = parse_v1_parts(&parts);
        assert_eq!(content, "Reading.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(calls[0].arguments, serde_json::json!({"command": "ls"}));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].call_id.as_deref(), Some("call_1"));
        assert_eq!(results[0].content, "a\nb");
        assert!(!results[0].is_error);
    }

    #[test]
    fn parse_v1_parts_tool_error_flags_result() {
        let parts = serde_json::json!([
            {"type": "tool", "callID": "call_2", "tool": "read",
             "state": {"status": "error", "input": {"filePath": "nope"}, "error": "file not found"}}
        ]);
        let (content, calls, results) = parse_v1_parts(&parts);
        assert_eq!(calls.len(), 1);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_error);
        assert_eq!(results[0].content, "file not found");
        // No text or reasoning: content falls back to the tool result.
        assert_eq!(content, "file not found");
    }

    #[test]
    fn parse_v1_parts_pending_tool_has_call_but_no_result() {
        let parts = serde_json::json!([
            {"type": "tool", "callID": "call_3", "tool": "edit", "state": {"status": "pending"}},
            {"type": "tool", "callID": "call_4", "tool": "grep",
             "state": {"status": "running", "input": "{\"pattern\":\"x\"}"}}
        ]);
        let (content, calls, results) = parse_v1_parts(&parts);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments, serde_json::json!({}));
        assert_eq!(calls[1].arguments, serde_json::json!({"pattern": "x"}));
        assert!(results.is_empty());
    }

    #[test]
    fn parse_v1_parts_reasoning_only_when_no_text_and_ignored_text_skipped() {
        let parts = serde_json::json!([
            {"type": "reasoning", "text": "thinking hard"},
            {"type": "text", "text": "hidden", "ignored": true}
        ]);
        let (content, calls, results) = parse_v1_parts(&parts);
        assert_eq!(content, "thinking hard");
        assert!(calls.is_empty());
        assert!(results.is_empty());

        let parts = serde_json::json!([
            {"type": "reasoning", "text": "thinking hard"},
            {"type": "text", "text": "visible"}
        ]);
        let (content, _, _) = parse_v1_parts(&parts);
        assert_eq!(content, "visible");
    }

    #[test]
    fn parse_v1_parts_file_marker_and_unknown_types() {
        let parts = serde_json::json!([
            {"type": "file", "mime": "image/png", "filename": "shot.png", "url": "data:..."},
            {"type": "snapshot", "snapshot": "abc"},
            {"type": "patch", "hash": "abc", "files": ["a.rs"]},
            {"type": "agent", "name": "build"},
            {"type": "text", "text": "see attached"}
        ]);
        let (content, calls, results) = parse_v1_parts(&parts);
        assert_eq!(content, "[file: shot.png]\nsee attached");
        assert!(calls.is_empty());
        assert!(results.is_empty());
    }

    #[test]
    fn parse_v1_parts_non_array_returns_empty() {
        let (content, calls, results) = parse_v1_parts(&serde_json::json!({"type": "text"}));
        assert!(content.is_empty());
        assert!(calls.is_empty());
        assert!(results.is_empty());
    }
    #[test]
    fn schema_mismatch_none_for_legacy_schema() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let db_path = tmp.path().join("opencode.db");
        let conn = OpenCode::open_db_rw(&db_path).expect("create db");
        OpenCode::ensure_schema(&conn).expect("schema");
        assert_eq!(OpenCode::schema_mismatch(&conn, &db_path), None);
    }

    #[test]
    fn schema_mismatch_on_empty_db_reports_no_tables() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let db_path = tmp.path().join("opencode.db");
        let conn = Connection::open(&db_path).expect("create empty db");
        let msg = OpenCode::schema_mismatch(&conn, &db_path).expect("empty db must mismatch");
        assert!(msg.contains("no tables at all"), "got: {msg}");
        assert!(
            msg.contains("closest is the legacy schema"),
            "empty db must be reported against the legacy schema: {msg}"
        );
        assert!(
            !msg.contains("closest is the 1.x schema"),
            "empty db is not a 1.x schema: {msg}"
        );
    }

    // ── XDG_DATA_HOME discovery (issue #26) ─────────────────────────────

    #[test]
    fn xdg_data_db_candidates_prefers_env_then_default() {
        let home = PathBuf::from("/home/user");
        let candidates =
            OpenCode::xdg_data_db_candidates(Some("/custom/data"), Some(home.as_path()));
        assert_eq!(
            candidates,
            vec![
                PathBuf::from("/custom/data/opencode/opencode.db"),
                PathBuf::from("/home/user/.local/share/opencode/opencode.db"),
            ]
        );
    }

    #[test]
    fn xdg_data_db_candidates_ignores_blank_env() {
        let home = PathBuf::from("/home/user");
        let candidates = OpenCode::xdg_data_db_candidates(Some("  "), Some(home.as_path()));
        assert_eq!(
            candidates,
            vec![PathBuf::from(
                "/home/user/.local/share/opencode/opencode.db"
            )]
        );
        assert!(OpenCode::xdg_data_db_candidates(None, None).is_empty());
    }

    #[test]
    fn list_sessions_empty_db_returns_empty_vec() {
        let _lock = OPENCODE_ENV.lock().expect("mutex lock");
        let tmp = tempfile::tempdir().expect("tmpdir");
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(workspace.join(".opencode")).expect("data dir");
        let _cwd = CwdGuard::change_to(&workspace);

        // Create empty DB with schema
        let db_path = workspace.join(".opencode/opencode.db");
        let conn = OpenCode::open_db_rw(&db_path).expect("create db");
        OpenCode::ensure_schema(&conn).expect("schema");
        drop(conn);

        let listed = OpenCode.list_sessions().expect("should return Some");
        assert!(listed.is_empty(), "empty DB should have no sessions");
    }
}
