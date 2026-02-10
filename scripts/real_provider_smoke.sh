#!/usr/bin/env bash
# scripts/real_provider_smoke.sh — Opt-in real-provider resume acceptance smoke harness.
#
# This script intentionally uses real provider homes and CLIs. It is NOT run in CI.
# It performs conversion + resume acceptance probes for:
#   CC <-> Codex, CC <-> Gemini, Codex <-> Gemini
#   CC <-> Cursor, CC <-> Cline, CC <-> Aider, CC <-> Amp, CC <-> OpenCode
#
# Artifacts:
#   - run.log: command transcript
#   - matrix.tsv: PASS/FAIL/SKIP rows per conversion path
#   - per-path command stdout/stderr files
#
# Usage:
#   bash scripts/real_provider_smoke.sh
# Optional:
#   VERBOSE=1 bash scripts/real_provider_smoke.sh
#   CASR_BIN=/path/to/casr bash scripts/real_provider_smoke.sh
#   SMOKE_ARTIFACTS_DIR=/tmp/casr-smoke bash scripts/real_provider_smoke.sh
#   SMOKE_ACCEPT_TIMEOUT=12 bash scripts/real_provider_smoke.sh
#   SMOKE_ACCEPT_CMD_CC='claude --resume {session_id}' bash scripts/real_provider_smoke.sh
#   SMOKE_ACCEPT_CMD_COD='codex resume {session_id}' bash scripts/real_provider_smoke.sh
#   SMOKE_ACCEPT_CMD_GMI='gemini --resume {session_id}' bash scripts/real_provider_smoke.sh
#   SMOKE_ACCEPT_CMD_CUR='cursor .' bash scripts/real_provider_smoke.sh
#   SMOKE_ACCEPT_CMD_CLN='code .' bash scripts/real_provider_smoke.sh
#   SMOKE_ACCEPT_CMD_AID='aider --restore-chat-history' bash scripts/real_provider_smoke.sh
#   SMOKE_ACCEPT_CMD_AMP='amp threads continue --execute \"Continue from @{session_id}\"' bash scripts/real_provider_smoke.sh
#   SMOKE_ACCEPT_CMD_OPC='opencode' bash scripts/real_provider_smoke.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CARGO_TARGET="${CARGO_TARGET_DIR:-$PROJECT_ROOT/target}"
CASR="${CASR_BIN:-$CARGO_TARGET/debug/casr}"
VERBOSE="${VERBOSE:-0}"
SMOKE_ACCEPT_TIMEOUT="${SMOKE_ACCEPT_TIMEOUT:-8}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
ARTIFACTS_DIR="${SMOKE_ARTIFACTS_DIR:-$PROJECT_ROOT/artifacts/real-smoke/$timestamp}"
mkdir -p "$ARTIFACTS_DIR"

RUN_LOG="$ARTIFACTS_DIR/run.log"
MATRIX_TSV="$ARTIFACTS_DIR/matrix.tsv"

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

if [[ -z "${NO_COLOR:-}" ]]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[0;33m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    RESET='\033[0m'
else
    GREEN='' RED='' YELLOW='' CYAN='' BOLD='' RESET=''
fi

declare -A PROVIDER_SLUG=(
    [cc]="claude-code"
    [cod]="codex"
    [gmi]="gemini"
    [cur]="cursor"
    [cln]="cline"
    [aid]="aider"
    [amp]="amp"
    [opc]="opencode"
)

declare -A PROVIDER_BIN=(
    [cc]="claude"
    [cod]="codex"
    [gmi]="gemini"
    [cur]="cursor"
    [cln]="code"
    [aid]="aider"
    [amp]="amp"
    [opc]="opencode"
)

declare -A PROVIDER_HOME=(
    [cc]="${CLAUDE_HOME:-$HOME/.claude}"
    [cod]="${CODEX_HOME:-$HOME/.codex}"
    [gmi]="${GEMINI_HOME:-$HOME/.gemini}"
    [cur]="${CURSOR_HOME:-$HOME/.config/Cursor}"
    [cln]="${CLINE_HOME:-$HOME/.config/Code/User/globalStorage/saoudrizwan.claude-dev}"
    [aid]="${AIDER_HOME:-$HOME/.aider}"
    [amp]="${AMP_HOME:-$HOME/.local/share/amp}"
    [opc]="${OPENCODE_HOME:-$HOME/.opencode}"
)

declare -A ACCEPT_TEMPLATE=(
    [cc]="${SMOKE_ACCEPT_CMD_CC:-}"
    [cod]="${SMOKE_ACCEPT_CMD_COD:-}"
    [gmi]="${SMOKE_ACCEPT_CMD_GMI:-}"
    [cur]="${SMOKE_ACCEPT_CMD_CUR:-}"
    [cln]="${SMOKE_ACCEPT_CMD_CLN:-}"
    [aid]="${SMOKE_ACCEPT_CMD_AID:-}"
    [amp]="${SMOKE_ACCEPT_CMD_AMP:-}"
    [opc]="${SMOKE_ACCEPT_CMD_OPC:-}"
)

ALL_ALIASES=(cc cod gmi cur cln aid amp opc)

declare -A SOURCE_SESSION_ID=()
declare -A PROVIDER_READY=()

LAST_EXIT=0
LAST_STDOUT_FILE=""
LAST_STDERR_FILE=""

log() {
    local msg="$1"
    printf "%s\n" "$msg" | tee -a "$RUN_LOG" > /dev/null
}

banner() {
    local msg="$1"
    echo -e "${CYAN}${BOLD}=== $msg ===${RESET}"
    log "=== $msg ==="
}

run_cmd() {
    local prefix="$1"
    shift
    local stdout_file="${prefix}.stdout"
    local stderr_file="${prefix}.stderr"

    log "CMD: $*"
    set +e
    "$@" > "$stdout_file" 2> "$stderr_file"
    local exit_code=$?
    set -e
    log "EXIT(${exit_code}): $*"

    if [[ "$VERBOSE" == "1" ]]; then
        [[ -s "$stdout_file" ]] && { echo "stdout ($stdout_file):"; head -40 "$stdout_file"; }
        [[ -s "$stderr_file" ]] && { echo "stderr ($stderr_file):"; head -40 "$stderr_file"; }
    fi

    LAST_EXIT=$exit_code
    LAST_STDOUT_FILE="$stdout_file"
    LAST_STDERR_FILE="$stderr_file"
}

run_accept_cmd() {
    local prefix="$1"
    local accept_cmd="$2"
    local stdout_file="${prefix}.stdout"
    local stderr_file="${prefix}.stderr"

    log "CMD: timeout ${SMOKE_ACCEPT_TIMEOUT}s bash -lc \"$accept_cmd\""
    set +e
    timeout "${SMOKE_ACCEPT_TIMEOUT}s" bash -lc "$accept_cmd" > "$stdout_file" 2> "$stderr_file"
    local exit_code=$?
    set -e
    log "EXIT(${exit_code}): timeout ${SMOKE_ACCEPT_TIMEOUT}s bash -lc \"$accept_cmd\""

    if [[ "$VERBOSE" == "1" ]]; then
        [[ -s "$stdout_file" ]] && { echo "stdout ($stdout_file):"; head -40 "$stdout_file"; }
        [[ -s "$stderr_file" ]] && { echo "stderr ($stderr_file):"; head -40 "$stderr_file"; }
    fi

    LAST_EXIT=$exit_code
    LAST_STDOUT_FILE="$stdout_file"
    LAST_STDERR_FILE="$stderr_file"
}

status_pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    echo -e "  ${GREEN}PASS${RESET}: $1"
}

status_fail() {
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo -e "  ${RED}FAIL${RESET}: $1"
}

status_skip() {
    SKIP_COUNT=$((SKIP_COUNT + 1))
    echo -e "  ${YELLOW}SKIP${RESET}: $1"
}

record_row() {
    local pair="$1"
    local status="$2"
    local source_session="$3"
    local target_session="$4"
    local convert_exit="$5"
    local accept_exit="$6"
    local written_path="$7"
    local accept_cmd="$8"
    local notes="$9"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$pair" "$status" "$source_session" "$target_session" "$convert_exit" "$accept_exit" "$written_path" "$accept_cmd" "$notes" >> "$MATRIX_TSV"
}

ensure_prereqs() {
    if ! command -v jq > /dev/null 2>&1; then
        echo "ERROR: jq is required."
        exit 1
    fi
    if ! command -v timeout > /dev/null 2>&1; then
        echo "ERROR: timeout is required."
        exit 1
    fi
    if [[ ! -x "$CASR" ]]; then
        banner "Building casr"
        (cd "$PROJECT_ROOT" && cargo build --quiet)
    fi
    if [[ ! -x "$CASR" ]]; then
        echo "ERROR: casr binary not found at $CASR"
        exit 1
    fi
}

source_session_for_alias() {
    local alias="$1"
    local slug="${PROVIDER_SLUG[$alias]}"
    local prefix="$ARTIFACTS_DIR/list_${alias}"

    run_cmd "$prefix" "$CASR" --json list --provider "$slug" --limit 25
    if [[ "$LAST_EXIT" -ne 0 ]]; then
        echo ""
        return 0
    fi
    jq -r '.[0].session_id // empty' "$LAST_STDOUT_FILE"
}

configure_provider_readiness() {
    for alias in cc cod gmi; do
        local bin="${PROVIDER_BIN[$alias]}"
        local home="${PROVIDER_HOME[$alias]}"
        local reason=""
        local ready="1"

        if ! command -v "$bin" > /dev/null 2>&1; then
            ready="0"
            reason="binary '$bin' not found"
        elif [[ ! -d "$home" ]]; then
            ready="0"
            reason="home '$home' not found"
        fi

        SOURCE_SESSION_ID["$alias"]=""
        if [[ "$ready" == "1" ]]; then
            SOURCE_SESSION_ID["$alias"]="$(source_session_for_alias "$alias")"
            if [[ -z "${SOURCE_SESSION_ID[$alias]}" ]]; then
                ready="0"
                reason="no discoverable sessions for ${PROVIDER_SLUG[$alias]}"
            fi
        fi

        PROVIDER_READY["$alias"]="$ready"
        if [[ "$ready" == "1" ]]; then
            log "Provider $alias ready: binary=$bin home=$home source_session=${SOURCE_SESSION_ID[$alias]}"
        else
            log "Provider $alias unavailable: $reason"
        fi
    done
}

expand_accept_command() {
    local alias="$1"
    local target_session="$2"
    local resume_command="$3"
    local template="${ACCEPT_TEMPLATE[$alias]}"
    if [[ -z "$template" ]]; then
        echo "$resume_command"
        return 0
    fi

    template="${template//\{session_id\}/$target_session}"
    template="${template//\{resume_command\}/$resume_command}"
    echo "$template"
}

run_pair() {
    local src="$1"
    local dst="$2"
    local pair="${src}->${dst}"
    local pair_dir="$ARTIFACTS_DIR/${src}_to_${dst}"
    mkdir -p "$pair_dir"

    local src_ready="${PROVIDER_READY[$src]}"
    local dst_bin="${PROVIDER_BIN[$dst]}"
    local dst_home="${PROVIDER_HOME[$dst]}"

    if [[ "$src_ready" != "1" ]]; then
        status_skip "$pair (source unavailable)"
        record_row "$pair" "SKIP" "" "" "-" "-" "" "" "source unavailable"
        return 0
    fi
    if ! command -v "$dst_bin" > /dev/null 2>&1; then
        status_skip "$pair (target binary '$dst_bin' missing)"
        record_row "$pair" "SKIP" "${SOURCE_SESSION_ID[$src]}" "" "-" "-" "" "" "target binary missing"
        return 0
    fi
    if [[ ! -d "$dst_home" ]]; then
        status_skip "$pair (target home '$dst_home' missing)"
        record_row "$pair" "SKIP" "${SOURCE_SESSION_ID[$src]}" "" "-" "-" "" "" "target home missing"
        return 0
    fi

    local source_session="${SOURCE_SESSION_ID[$src]}"

    run_cmd \
        "$pair_dir/convert" \
        "$CASR" --json resume "$dst" "$source_session" --source "$src" --force
    local convert_exit="$LAST_EXIT"
    if [[ "$convert_exit" -ne 0 ]]; then
        status_fail "$pair conversion failed"
        record_row "$pair" "FAIL" "$source_session" "" "$convert_exit" "-" "" "" "conversion failed (see $pair_dir/convert.stderr)"
        return 0
    fi

    local target_session
    target_session="$(jq -r '.target_session_id // empty' "$LAST_STDOUT_FILE")"
    local written_path
    written_path="$(jq -r '.written_paths[0] // empty' "$LAST_STDOUT_FILE")"
    local resume_cmd
    resume_cmd="$(jq -r '.resume_command // empty' "$LAST_STDOUT_FILE")"

    if [[ -z "$target_session" || -z "$written_path" || -z "$resume_cmd" ]]; then
        status_fail "$pair conversion output missing target session details"
        record_row "$pair" "FAIL" "$source_session" "" "$convert_exit" "-" "$written_path" "" "missing target_session_id/written_paths/resume_command"
        return 0
    fi

    local accept_cmd
    accept_cmd="$(expand_accept_command "$dst" "$target_session" "$resume_cmd")"

    run_accept_cmd "$pair_dir/accept" "$accept_cmd"
    local accept_exit="$LAST_EXIT"
    local note=""
    local status=""

    if [[ "$accept_exit" -eq 0 ]]; then
        status="PASS"
        note="acceptance command exited 0"
    elif [[ "$accept_exit" -eq 124 ]]; then
        if grep -Eqi "error|invalid|not found|no such|unknown session|failed" "$LAST_STDOUT_FILE" "$LAST_STDERR_FILE"; then
            status="FAIL"
            note="acceptance timed out with error output"
        else
            status="PASS"
            note="acceptance command entered interactive mode (timeout ${SMOKE_ACCEPT_TIMEOUT}s)"
        fi
    else
        status="FAIL"
        note="acceptance command exit=$accept_exit"
    fi

    if [[ "$status" == "PASS" ]]; then
        status_pass "$pair ($note)"
    else
        status_fail "$pair ($note)"
    fi

    record_row "$pair" "$status" "$source_session" "$target_session" "$convert_exit" "$accept_exit" "$written_path" "$accept_cmd" "$note"
}

print_matrix() {
    banner "Matrix"
    printf "Path\tStatus\tSourceSession\tTargetSession\tConvertExit\tAcceptExit\tWrittenPath\tAcceptCommand\tNotes\n"
    cat "$MATRIX_TSV" | while IFS= read -r line; do
        printf "%s\n" "$line"
    done
}

main() {
    echo -e "${BOLD}casr real-provider smoke harness${RESET}"
    echo "Binary: $CASR"
    echo "Artifacts: $ARTIFACTS_DIR"
    echo "Timeout: ${SMOKE_ACCEPT_TIMEOUT}s"
    echo ""

    ensure_prereqs

    printf "path\tstatus\tsource_session\ttarget_session\tconvert_exit\taccept_exit\twritten_path\taccept_cmd\tnotes\n" > "$MATRIX_TSV"
    : > "$RUN_LOG"

    banner "Provider readiness"
    configure_provider_readiness
    for alias in cc cod gmi; do
        if [[ "${PROVIDER_READY[$alias]}" == "1" ]]; then
            echo "  READY: $alias (${PROVIDER_SLUG[$alias]}) source_session=${SOURCE_SESSION_ID[$alias]}"
        else
            echo "  SKIP:  $alias (${PROVIDER_SLUG[$alias]})"
        fi
    done
    echo ""

    banner "Smoke conversions"
    run_pair cc cod
    run_pair cod cc
    run_pair cc gmi
    run_pair gmi cc
    run_pair cod gmi
    run_pair gmi cod
    echo ""

    print_matrix > "$ARTIFACTS_DIR/matrix.txt"
    cat "$ARTIFACTS_DIR/matrix.txt"

    local total
    total=$((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))
    echo ""
    echo -e "${BOLD}Summary:${RESET} ${GREEN}${PASS_COUNT} passed${RESET}, ${RED}${FAIL_COUNT} failed${RESET}, ${YELLOW}${SKIP_COUNT} skipped${RESET} (${total} total)"
    echo "Artifacts: $ARTIFACTS_DIR"

    if [[ "$FAIL_COUNT" -gt 0 ]]; then
        exit 1
    fi
}

main "$@"
