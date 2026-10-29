#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/run_security_checks.sh [--with-audit | --help]

Run local offline security checks. Add --with-audit to also query pip-audit.
EOF
}

WITH_AUDIT=0
case "${1:-}" in
    "")
        ;;
    --with-audit)
        if [[ "$#" -ne 1 ]]; then
            printf 'Error: --with-audit no acepta argumentos adicionales.\n' >&2
            exit 2
        fi
        WITH_AUDIT=1
        ;;
    --help|-h)
        if [[ "$#" -ne 1 ]]; then
            printf 'Error: --help no acepta argumentos adicionales.\n' >&2
            exit 2
        fi
        usage
        exit 0
        ;;
    *)
        printf 'Error: argumento desconocido: %s\n' "$1" >&2
        usage >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
else
    PYTHON="$REPO_ROOT/.venv/bin/python"
fi

if [[ ! -x "$PYTHON" ]]; then
    printf 'Error: no se encontró un Python ejecutable en %s.\n' "$PYTHON" >&2
    exit 1
fi

cd "$REPO_ROOT"
EVIDENCE_DIR=/tmp/actividad7-security
mkdir -p "$EVIDENCE_DIR"
: > "$EVIDENCE_DIR/security-checks.txt"

FAILED=0
AUDIT_FINDINGS=0

run_check() {
    local name="$1"
    local status
    shift
    printf '\n>>> %s' "$*" | tee -a "$EVIDENCE_DIR/security-checks.txt"
    printf '\n' | tee -a "$EVIDENCE_DIR/security-checks.txt"
    if "$PYTHON" "$@" 2>&1 \
        | tee "$EVIDENCE_DIR/$name.txt" \
        | tee -a "$EVIDENCE_DIR/security-checks.txt"; then
        return 0
    else
        status="$?"
        if [[ "$name" == "pip-audit" && "$status" -eq 1 ]] \
            && grep -Eq '^Found [1-9][0-9]* known vulnerabilities' "$EVIDENCE_DIR/$name.txt"; then
            printf 'RESULTADO: AUDITORÍA EJECUTADA CON HALLAZGOS (código %s).\n' "$status" \
                | tee -a "$EVIDENCE_DIR/security-checks.txt"
        else
            printf 'ERROR: %s terminó con código %s.\n' "$name" "$status" \
                | tee -a "$EVIDENCE_DIR/security-checks.txt" >&2
        fi
        return "$status"
    fi
}

if run_check pip-check -m pip check; then :; else FAILED=1; fi
if run_check bandit-gate -m bandit -r mibici_ml -c .bandit \
    --severity-level medium --confidence-level medium; then :; else FAILED=1; fi
if run_check pytest-security -m pytest -q tests/test_repository_security.py; then
    :
else
    FAILED=1
fi

if [[ "$WITH_AUDIT" -eq 1 ]]; then
    if run_check pip-audit -m pip_audit -r requirements.txt; then
        :
    else
        status="$?"
        if [[ "$status" -eq 1 ]] \
            && grep -Eq '^Found [1-9][0-9]* known vulnerabilities' "$EVIDENCE_DIR/pip-audit.txt"; then
            AUDIT_FINDINGS=1
        else
            FAILED=1
        fi
    fi
fi

if [[ "$FAILED" -ne 0 ]]; then
    printf 'ERROR: uno o más chequeos técnicos fallaron; revisa /tmp/actividad7-security/.\n' >&2
    exit 1
fi

if [[ "$AUDIT_FINDINGS" -ne 0 ]]; then
    printf 'AUDITORÍA EJECUTADA CON HALLAZGOS; código final 1. Revisa /tmp/actividad7-security/.\n'
    exit 1
fi

printf 'Chequeos completados sin fallos; código final 0.\n'
