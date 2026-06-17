"""
main.py — LangGraph-first CSV tool-call executor.

Pipeline per row:
1) Parse raw row + recover malformed Tool_Call text where possible.
2) Ask GPT (OpenAI) to normalize into executable request intent.
3) Execute request (HTTP) and classify as PASS/FAIL.
4) Aggregate reports.
"""

import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, TypedDict

import requests
from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# ---------------------------------------------------------------------------
# ENV + CONFIG
# ---------------------------------------------------------------------------

def init_env() -> None:
    dotenv_path = os.getenv("DOTENV_PATH", "").strip()
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)
        return
    auto = find_dotenv(usecwd=True)
    if auto:
        load_dotenv(dotenv_path=auto, override=False)
        return
    for candidate in (".env", "/app/.env", "/app/app/.env"):
        if Path(candidate).exists():
            load_dotenv(dotenv_path=candidate, override=False)
            return


init_env()

CSV_PATH = os.getenv("CSV_PATH", "output.csv")
STOP_ON_ERROR = os.getenv("STOP_ON_ERROR", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
ALLOWED_TEST_TYPES: list[str] = []
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))
API_BASE_URL = os.getenv("API_BASE_URL", "http://driver-safety.corazor.com/").rstrip("/")
ML_BASE_URL = os.getenv("ML_BASE_URL", "http://driver-safety.corazor.com/").rstrip("/")
REPORT_PATH = os.getenv("REPORT_PATH", "")
SUMMARY_PATH = os.getenv("SUMMARY_PATH", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
USE_LLM_NORMALIZER = os.getenv("USE_LLM_NORMALIZER", "true").lower() == "true"
USE_LLM_VALUE_SYNTHESIS = os.getenv("USE_LLM_VALUE_SYNTHESIS", "true").lower() == "true"
DOCUMENTATION_PATH = os.getenv("DOCUMENTATION_PATH", "").strip()
MAX_DOC_CHARS = int(os.getenv("MAX_DOC_CHARS", "50000"))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
session = requests.Session()

# ---------------------------------------------------------------------------
# DYNAMIC SYNTHESIS HELPERS
# ---------------------------------------------------------------------------


def _extract_placeholder_fields(payload: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    for k, v in payload.items():
        if isinstance(v, str) and "<" in v and ">" in v:
            fields.append(k)
    return fields


def _merge_dict_values(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in updates.items():
        merged[str(k)] = v
    return merged


def sanitize_placeholders(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Detect unresolved placeholders; leave payload unchanged."""
    had_any = bool(_extract_placeholder_fields(payload))
    return dict(payload), had_any


def parse_422_missing_fields(response_text: str) -> list[tuple[str, str]]:
    """Parse FastAPI 422 detail. Returns list of (location, field_name)."""
    try:
        data = json.loads(response_text)
    except (json.JSONDecodeError, ValueError):
        return []
    missing: list[tuple[str, str]] = []
    detail = data.get("detail", []) if isinstance(data, dict) else []
    if not isinstance(detail, list):
        return []
    for entry in detail:
        if not isinstance(entry, dict):
            continue
        loc = entry.get("loc", [])
        if not isinstance(loc, list) or len(loc) < 2:
            continue
        location, field = str(loc[0]), str(loc[-1])
        if location in {"body", "query", "path"}:
            missing.append((location, field))
    return missing


def _llm_generate_values_for_missing(
    state: "RowState",
    payload: dict[str, Any],
    missing: list[tuple[str, str]],
    response_snippet: str = "",
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if value_synth_chain is None:
        return None
    try:
        llm_text = value_synth_chain.invoke(
            {
                "description": state.get("description", ""),
                "documentation_context": documentation_context,
                "function_name": state.get("function_name", ""),
                "method": state.get("method", ""),
                "path": state.get("path", ""),
                "payload_json": json.dumps(payload, ensure_ascii=True),
                "missing_fields_json": json.dumps(
                    [{"location": loc, "field": field} for loc, field in missing],
                    ensure_ascii=True,
                ),
                "response_snippet": response_snippet[:500],
            }
        )
        generated = safe_json_loads(llm_text)
        body = generated.get("body", {})
        query = generated.get("query", {})
        if not isinstance(body, dict):
            body = {}
        if not isinstance(query, dict):
            query = {}
        return body, query
    except Exception as exc:  # noqa: BLE001
        log.debug("Prompt_ID=%s value synthesis failed: %s", state.get("prompt_id"), exc)
        return None


def enrich_payload_dynamic(
    state: "RowState",
    payload: dict[str, Any],
    missing: list[tuple[str, str]],
    response_snippet: str = "",
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    llm_values = _llm_generate_values_for_missing(state, payload, missing, response_snippet)
    if llm_values is not None:
        llm_body, llm_query = llm_values
        return _merge_dict_values(payload, llm_body), llm_query
    return None


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------

class RowState(TypedDict):
    row: dict[str, str]
    prompt_id: int
    test_type: str
    description: str
    raw_tool: str
    function_name: str
    tool_payload: dict[str, Any]
    method: str
    base_url: str
    path: str
    request_payload: dict[str, Any]
    should_execute: bool
    expected_status: int
    status: str
    reason: str
    http_status: str
    response_snippet: str

# ---------------------------------------------------------------------------
# LLM NORMALIZER (optional)
# ---------------------------------------------------------------------------

NORMALIZE_PROMPT = PromptTemplate(
    template=(
        "You are a strict API tool-call normalizer.\n"
        "Given one QA row, return only compact JSON with keys:\n"
        "method, base_hint, path, payload, should_execute, reason.\n\n"
        "Rules:\n"
        "- method must be one of GET, POST, PUT, PATCH, DELETE, WS, UNKNOWN.\n"
        "- base_hint must be one of api, ml.\n"
        "- path must start with '/' and have no trailing punctuation like '.', ',' or ')'.\n"
        "- should_execute=true always; placeholders like <image> are replaced automatically later.\n"
        "- If websocket or unclear mapping, keep method=WS or UNKNOWN.\n"
        "- No markdown fences.\n\n"
        "Documentation context:\n{documentation_context}\n\n"
        "Row description:\n{description}\n\n"
        "Function name:\n{function_name}\n\n"
        "Payload JSON:\n{payload_json}\n"
    ),
    input_variables=["documentation_context", "description", "function_name", "payload_json"],
)

normalizer_chain = None
value_synth_chain = None
documentation_context = ""


VALUE_SYNTH_PROMPT = PromptTemplate(
    template=(
        "You generate realistic synthetic API values for QA execution.\n"
        "Return only compact JSON object with keys: body, query.\n"
        "Each key maps to an object of field->value.\n"
        "Rules:\n"
        "- Use valid values likely accepted by typical FastAPI/Pydantic APIs.\n"
        "- Prefer plain scalar values (str/int/float/bool), short arrays/objects only if field clearly implies it.\n"
        "- If a field suggests image/base64, return a valid base64 string.\n"
        "- Do not include markdown fences.\n\n"
        "Documentation context:\n{documentation_context}\n\n"
        "Description:\n{description}\n\n"
        "Function Name:\n{function_name}\n"
        "Method: {method}\n"
        "Path: {path}\n\n"
        "Current payload JSON:\n{payload_json}\n\n"
        "Requested fields JSON (location + field):\n{missing_fields_json}\n\n"
        "Last response snippet (may include validation hints):\n{response_snippet}\n"
    ),
    input_variables=[
        "description",
        "documentation_context",
        "function_name",
        "method",
        "path",
        "payload_json",
        "missing_fields_json",
        "response_snippet",
    ],
)


def build_normalizer_chain():
    return NORMALIZE_PROMPT | ChatOpenAI(model=OPENAI_MODEL, temperature=0) | StrOutputParser()


def build_value_synth_chain():
    return VALUE_SYNTH_PROMPT | ChatOpenAI(model=OPENAI_MODEL, temperature=0) | StrOutputParser()


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

SUPPORTED_DOC_EXTENSIONS = {".txt", ".pdf", ".docx", ".md"}
DOC_LOADERS = {
    ".txt": lambda p: TextLoader(str(p), encoding="utf-8"),
    ".pdf": lambda p: PyPDFLoader(str(p)),
    ".docx": lambda p: Docx2txtLoader(str(p)),
    ".md": lambda p: UnstructuredMarkdownLoader(str(p)),
}


def _load_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in DOC_LOADERS:
        return ""
    docs = DOC_LOADERS[ext](path).load()
    content = "\n".join(d.page_content for d in docs).strip()
    if len(content) > MAX_DOC_CHARS:
        content = content[:MAX_DOC_CHARS] + "\n\n[...truncated...]"
    return content


def resolve_documentation_path(csv_file: Path) -> Path | None:
    if DOCUMENTATION_PATH:
        doc_path = Path(DOCUMENTATION_PATH)
        if doc_path.exists():
            return doc_path
        log.warning("DOCUMENTATION_PATH does not exist: %s", doc_path)
    candidates: list[Path] = []
    for root in (csv_file.parent, csv_file.parent.parent, Path.cwd()):
        for ext in SUPPORTED_DOC_EXTENSIONS:
            candidates.extend(root.glob(f"*{ext}"))
    def score(path: Path) -> tuple[int, float]:
        name = path.name.lower()
        rank = 0
        if "guide" in name or "documentation" in name or "spec" in name:
            rank -= 20
        if "qa" in name or "test" in name or "bluebird" in name or "carozor" in name:
            rank -= 10
        return rank, -path.stat().st_mtime
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=score)
    return candidates[0]

def resolve_csv_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    fallback_candidates = [
        Path.cwd() / raw_path,
        Path.cwd() / "output.csv",
        Path("/app/app/output.csv"),
        Path("/app/output.csv"),
    ]
    for path in fallback_candidates:
        if path.exists():
            return path
    searched = [str(candidate)] + [str(p) for p in fallback_candidates]
    raise FileNotFoundError(
        "CSV not found. Checked: " + ", ".join(searched)
        + ". Set CSV_PATH correctly (for docker commonly /app/app/output.csv)."
    )


def resolve_report_paths(csv_file: Path) -> tuple[Path, Path]:
    """Reports go next to the input CSV so they survive docker volume mounts."""
    csv_dir = csv_file.parent
    report = Path(REPORT_PATH) if REPORT_PATH else csv_dir / "execution_report.csv"
    summary = Path(SUMMARY_PATH) if SUMMARY_PATH else csv_dir / "execution_summary.json"
    return report, summary


def pick_first(row: dict[str, str], keys: list[str], default: str = "") -> str:
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip() != "":
            return str(v)
    return default


def extract_raw_tool_call(row: dict[str, str]) -> str:
    raw_tool = pick_first(row, ["Tool_Call", "tool_call", "tool", "Tool"], "").strip()
    if raw_tool:
        return raw_tool
    desc = pick_first(row, ["Description", "description", "prompt", "Prompt", "instruction"], "")
    match = re.search(r'(\{\s*"role"\s*:\s*"assistant".*)', desc)
    return match.group(1).strip() if match else ""


def extract_expected_status(description: str) -> int:
    """Parse expected HTTP status code from test description. Returns 0 for 'any 2xx'."""
    patterns = [
        r"(?:expected|response)[^.]*?(?:code|status)[^.]*?(\d{3})",
        r"returns?\s+(?:a\s+)?(\d{3})\b",
        r"(?:should|must)\s+(?:return|receive|get|respond)\s+(?:a\s+)?(\d{3})\b",
        r"response\s+(?:code\s+)?(?:is\s+)?(\d{3})\b",
        r"\b(\d{3})\s+(?:error|status|response|indicating)",
    ]
    for p in patterns:
        m = re.search(p, description, re.IGNORECASE)
        if m:
            code = int(m.group(1))
            if 100 <= code <= 599:
                return code
    return 0


def parse_tool_payload(raw_tool: str) -> tuple[str, dict[str, Any], str]:
    if not raw_tool:
        return "", {}, "missing_tool_call_json"
    try:
        tool_obj = json.loads(raw_tool)
    except json.JSONDecodeError as exc:
        return "", {}, f"invalid_tool_call_json: {exc}"

    calls = tool_obj.get("tool_calls", [])
    if not calls:
        return "", {}, "tool_calls_not_found"

    function = calls[0].get("function", {})
    function_name = function.get("name", "")
    if not function_name:
        return "", {}, "function_name_missing"

    args_raw = function.get("arguments", "{}")
    try:
        payload = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
    except json.JSONDecodeError:
        return function_name, {}, "invalid_function_arguments_json"

    if not isinstance(payload, dict):
        return function_name, {}, "arguments_not_object"
    return function_name, payload, ""


def safe_json_loads(text: str) -> dict[str, Any]:
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# GRAPH NODES
# ---------------------------------------------------------------------------

def parse_node(state: RowState) -> RowState:
    row = state["row"]
    raw_id = pick_first(row, ["Prompt_ID", "prompt_id", "id", "ID"], "0")
    prompt_id = int(raw_id) if str(raw_id).isdigit() else 0
    test_type = pick_first(row, ["Test_Type", "test_type", "category", "Category"], "")
    description = pick_first(row, ["Description", "description", "prompt", "Prompt", "instruction"], "")
    raw_tool = extract_raw_tool_call(row)
    function_name, payload, error = parse_tool_payload(raw_tool)
    expected = extract_expected_status(description)

    if not function_name:
        function_name = "description_inferred_call"

    recoverable_errors = {
        "missing_tool_call_json",
        "tool_calls_not_found",
        "function_name_missing",
        "arguments_not_object",
        "invalid_function_arguments_json",
    }
    is_recoverable = (
        error in recoverable_errors
        or (error or "").startswith("invalid_tool_call_json")
    )

    base = {
        "prompt_id": prompt_id,
        "test_type": test_type,
        "description": description,
        "raw_tool": raw_tool,
        "function_name": function_name,
        "expected_status": expected,
        "http_status": "",
        "response_snippet": "",
        "method": "",
        "base_url": "",
        "path": "",
        "request_payload": {},
    }

    if is_recoverable:
        base.update({"tool_payload": payload or {}, "status": "", "reason": "tool_call_missing_using_description"})
    elif error:
        base.update({"tool_payload": {}, "status": "FAIL", "reason": error, "should_execute": False})
    else:
        base.update({"tool_payload": payload, "status": "", "reason": ""})

    state.update(base)
    return state


_TRAILING_JUNK = ".,;:!?)\"'"


def _clean_path(raw: str) -> str:
    """Normalise an extracted URL path (strip trailing punctuation, whitespace)."""
    path = raw.strip().rstrip(_TRAILING_JUNK)
    while path.endswith(tuple(_TRAILING_JUNK)):
        path = path.rstrip(_TRAILING_JUNK)
    return path


def extract_method_path_from_description(description: str) -> tuple[str, str] | None:
    """Extract HTTP method + path from free-form description text."""
    method_match = re.search(r"\b(GET|POST|PUT|PATCH|DELETE)\b", description, re.IGNORECASE)
    path_match = re.search(r"(?:^|\s)(\/[\w/\-?=&]+)", description)
    if method_match and path_match:
        return method_match.group(1).upper(), _clean_path(path_match.group(1))
    if path_match:
        path = _clean_path(path_match.group(1))
        return ("GET" if "get" in description.lower()[:60] else "POST"), path
    ws = re.search(r"\b(?:websocket|ws://|stream)\S*(\/[\w/\-?=&]+)", description, re.IGNORECASE)
    if ws:
        return "WS", _clean_path(ws.group(1))
    return None


def heuristic_fallback(state: RowState) -> RowState:
    name = state["function_name"].lower()
    desc_raw = state["description"]
    desc = desc_raw.lower()
    payload = state["tool_payload"]
    method = "UNKNOWN"
    path = "/"
    base = API_BASE_URL

    if "ml service" in desc or "port 8001" in desc:
        base = ML_BASE_URL

    extracted = extract_method_path_from_description(desc_raw)
    if extracted:
        method, path = extracted
        if path.startswith("/health/models") or path.startswith("/models"):
            base = ML_BASE_URL
    elif "health/models" in desc or "model_health" in name or "models_health" in name:
        method, path = "GET", "/health/models"
        base = ML_BASE_URL
    elif "ml_service_health" in name:
        method, path = "GET", "/health"
        base = ML_BASE_URL
    elif "health" in name or "health" in desc.split(".")[0]:
        method, path = "GET", "/health"
    elif "register" in name and "verify" not in name and "finalize" not in name:
        method, path = "POST", "/api/login/register"
    elif "verify_dl" in name or "verify_driving" in name:
        method, path = "POST", "/api/login/verify-dl"
    elif "finalize_dl" in name or "finalize_driving" in name:
        method, path = "POST", "/api/login/finalize-dl"
    elif "login" in name or "authenticate_driver" in name:
        method, path = "POST", "/api/login/"
    elif "start_session" in name or "start session" in desc:
        method, path = "POST", "/api/sessions/start"
    elif "end_session" in name or "end session" in desc:
        method, path = "POST", "/api/sessions/end"
    elif "safety_score" in name and ("get" in name or "fetch" in name or "query" in name or "retrieve" in name):
        method, path = "GET", "/api/safety-score"
    elif "safety_score" in name or "compute_safety" in name:
        method, path = "POST", "/api/safety-score/compute"
    elif "recalibrate" in name or "recalibrat" in desc:
        method, path = "GET", "/api/recalibrate"
    elif "demo_reset" in name or "reset_driver" in name or "reset-driver" in desc:
        method, path = "POST", "/api/demo/reset-driver"
    elif "stream" in name or "ws_" in name or "websocket" in desc:
        method, path = "WS", "/api/stream"

    sanitized, had_placeholders = sanitize_placeholders(payload)
    state.update(
        {
            "method": method,
            "base_url": base,
            "path": path,
            "request_payload": sanitized,
            "should_execute": True,
            "reason": "placeholder_sanitized" if had_placeholders else "fallback_mapping",
        }
    )
    return state


def normalize_node(state: RowState) -> RowState:
    if state.get("status") == "FAIL":
        return state
    if normalizer_chain is None:
        return heuristic_fallback(state)
    try:
        llm_text = normalizer_chain.invoke(
            {
                "documentation_context": documentation_context,
                "description": state["description"],
                "function_name": state["function_name"],
                "payload_json": json.dumps(state["tool_payload"], ensure_ascii=True),
            }
        )
        normalized = safe_json_loads(llm_text)
        method = str(normalized.get("method", "UNKNOWN")).upper()
        base_hint = str(normalized.get("base_hint", "api")).lower()
        path = str(normalized.get("path", "/")).strip()
        payload = normalized.get("payload", state["tool_payload"])
        reason = str(normalized.get("reason", "llm_mapping"))
        if not path.startswith("/"):
            path = "/" + path
        path = _clean_path(path)
        base_url = ML_BASE_URL if base_hint == "ml" else API_BASE_URL
        if not isinstance(payload, dict):
            payload = state["tool_payload"]
        placeholder_fields = _extract_placeholder_fields(payload)
        if placeholder_fields:
            dynamic_missing = [("body", field) for field in placeholder_fields]
            enriched = enrich_payload_dynamic(state, payload, dynamic_missing)
            if enriched is not None:
                dynamic_body, _ = enriched
                payload = dynamic_body
        sanitized, _ = sanitize_placeholders(payload)
        state.update(
            {
                "method": method,
                "base_url": base_url,
                "path": path,
                "request_payload": sanitized,
                "should_execute": True,
                "reason": reason,
            }
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Prompt_ID=%s LLM normalize failed: %s", state["prompt_id"], exc)
        state = heuristic_fallback(state)
    return state


CLIENT_ERROR_EQUIVALENTS = {400, 422}
AUTH_ERROR_EQUIVALENTS = {401, 403}


def _status_matches_expected(actual: int, expected: int) -> bool:
    if expected == 0:
        return 200 <= actual < 300
    if actual == expected:
        return True
    if expected in CLIENT_ERROR_EQUIVALENTS and actual in CLIENT_ERROR_EQUIVALENTS:
        return True
    if expected in AUTH_ERROR_EQUIVALENTS and actual in AUTH_ERROR_EQUIVALENTS:
        return True
    return False


def _send(
    method: str,
    url: str,
    payload: dict[str, Any],
    query: dict[str, Any] | None = None,
) -> requests.Response:
    """Send a single HTTP request with appropriate body/query handling."""
    if method in {"GET", "DELETE"}:
        params = {k: v for k, v in payload.items()
                  if not (isinstance(v, str) and "<" in v and ">" in v)}
        if query:
            params.update(query)
        return session.request(method, url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    return session.request(
        method,
        url,
        json=payload,
        params=query or None,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def execute_node(state: RowState) -> RowState:
    if state.get("status") == "FAIL":
        return state

    method = state.get("method", "UNKNOWN")
    base_url = state.get("base_url", "")
    path = state.get("path", "")
    url = f"{base_url}{path}"
    payload = state.get("request_payload", {})
    expected = state.get("expected_status", 0)

    if method in {"UNKNOWN", ""}:
        state.update({"status": "FAIL", "reason": "unsupported_function_mapping", "http_status": ""})
        return state

    if method == "WS":
        # Real WS testing requires a WS client; record as skipped so it doesn't
        # pollute HTTP failure stats. Count as PASS when we can't negatively
        # assert anything about it beyond "endpoint declared".
        state.update({
            "status": "SKIP",
            "reason": "websocket_skipped_requires_ws_client",
            "http_status": "",
            "response_snippet": "",
        })
        return state

    try:
        resp = _send(method, url, payload)

        # Auto-retry once on 422: parse FastAPI detail, fill missing fields.
        if resp.status_code == 422 and method not in {"GET", "DELETE"}:
            missing = parse_422_missing_fields(resp.text)
            if missing:
                enriched = enrich_payload_dynamic(state, payload, missing, resp.text)
                if enriched is not None:
                    new_payload, new_query = enriched
                    try:
                        resp2 = _send(method, url, new_payload, new_query)
                        if resp2.status_code != 422 or _status_matches_expected(
                            resp2.status_code, expected
                        ):
                            resp = resp2
                            payload = new_payload
                    except requests.RequestException:
                        pass

        matches = _status_matches_expected(resp.status_code, expected)
        if matches:
            reason = "ok" if expected == 0 else f"expected_{expected}_matched"
        else:
            reason = f"expected_{expected}_got_{resp.status_code}" if expected else "non_2xx_response"

        state.update({
            "status": "PASS" if matches else "FAIL",
            "reason": reason,
            "http_status": str(resp.status_code),
            "response_snippet": resp.text[:240],
            "request_payload": payload,
        })
    except requests.RequestException as exc:
        state.update({
            "status": "FAIL",
            "reason": f"request_error: {exc.__class__.__name__}",
            "http_status": "",
            "response_snippet": "",
        })
    return state


# ---------------------------------------------------------------------------
# GRAPH
# ---------------------------------------------------------------------------

def build_row_graph():
    graph = StateGraph(RowState)
    graph.add_node("parse", parse_node)
    graph.add_node("normalize", normalize_node)
    graph.add_node("execute", execute_node)
    graph.set_entry_point("parse")
    graph.add_edge("parse", "normalize")
    graph.add_edge("normalize", "execute")
    graph.add_edge("execute", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------

REPORT_HEADERS = [
    "prompt_id", "test_type", "status", "reason", "expected_status",
    "function_name", "method", "url", "http_status", "response_snippet",
]


def to_result(state: RowState) -> dict[str, Any]:
    return {
        "prompt_id": state.get("prompt_id", 0),
        "test_type": state.get("test_type", ""),
        "status": state.get("status", "FAIL"),
        "reason": state.get("reason", "unknown"),
        "expected_status": state.get("expected_status", 0),
        "function_name": state.get("function_name", ""),
        "method": state.get("method", ""),
        "url": f"{state.get('base_url', '')}{state.get('path', '')}" if state.get("path") else "",
        "http_status": state.get("http_status", ""),
        "response_snippet": state.get("response_snippet", ""),
    }


def write_reports(
    results: list[dict[str, Any]],
    report_path: Path,
    summary_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_HEADERS)
        writer.writeheader()
        writer.writerows(results)

    working = sum(1 for r in results if r["status"] == "PASS")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    not_working = sum(1 for r in results if r["status"] == "FAIL")
    reason_counts: dict[str, int] = {}
    for r in results:
        reason_counts[r["reason"]] = reason_counts.get(r["reason"], 0) + 1

    evaluated = max(len(results) - skipped, 1)
    summary = {
        "total_rows": len(results),
        "working": working,
        "not_working": not_working,
        "skipped": skipped,
        "pass_rate_pct": round(working / max(len(results), 1) * 100, 1),
        "pass_rate_excl_skipped_pct": round(working / evaluated * 100, 1),
        "breakdown_by_reason": dict(sorted(reason_counts.items(), key=lambda x: -x[1])),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info(
        "Summary: total=%d working=%d not_working=%d skipped=%d (%.1f%% pass, %.1f%% excl skipped)",
        len(results), working, not_working, skipped,
        summary["pass_rate_pct"], summary["pass_rate_excl_skipped_pct"],
    )
    log.info("Detailed report: %s", report_path)
    log.info("Summary report:  %s", summary_path)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    csv_file = resolve_csv_path(CSV_PATH)
    report_path, summary_path = resolve_report_paths(csv_file)

    global normalizer_chain, value_synth_chain, documentation_context  # noqa: PLW0603
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for dynamic normalization/value generation. "
            "Set it in .env or via docker --env-file."
        )

    doc_path = resolve_documentation_path(csv_file)
    if doc_path:
        try:
            documentation_context = _load_document(doc_path)
            log.info("Loaded documentation context from %s (%d chars)", doc_path, len(documentation_context))
        except Exception as exc:  # noqa: BLE001
            documentation_context = ""
            log.warning("Failed to load documentation context from %s: %s", doc_path, exc)
    else:
        documentation_context = ""
        log.warning("No documentation file found near CSV. LLM will use row-only context.")

    if USE_LLM_NORMALIZER and api_key:
        normalizer_chain = build_normalizer_chain()
        log.info("LLM normalizer enabled with model=%s", OPENAI_MODEL)
    else:
        normalizer_chain = None
        log.info("LLM normalizer disabled via USE_LLM_NORMALIZER=false.")

    if USE_LLM_VALUE_SYNTHESIS and api_key:
        value_synth_chain = build_value_synth_chain()
        log.info("LLM value synthesis enabled with model=%s", OPENAI_MODEL)
    else:
        value_synth_chain = None
        log.info("LLM value synthesis disabled via USE_LLM_VALUE_SYNTHESIS=false.")

    row_graph = build_row_graph()
    results: list[dict[str, Any]] = []
    total = skipped = errors = 0
    log.info("Starting LangGraph execution on %s", csv_file)

    with csv_file.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total += 1
            if ALLOWED_TEST_TYPES and row.get("Test_Type") not in ALLOWED_TEST_TYPES:
                skipped += 1
                continue
            try:
                state: RowState = {"row": row}
                final_state = row_graph.invoke(state)
                result = to_result(final_state)
                results.append(result)
                if result["status"] == "PASS":
                    log.info("Prompt_ID=%s PASS %s %s (status=%s)",
                             result["prompt_id"], result["method"], result["url"], result["http_status"])
                else:
                    log.warning("Prompt_ID=%s FAIL reason=%s (status=%s)",
                                result["prompt_id"], result["reason"], result["http_status"])
            except Exception as exc:  # noqa: BLE001
                errors += 1
                log.error("Prompt_ID=%s graph error: %s", row.get("Prompt_ID"), exc)
                if STOP_ON_ERROR:
                    raise

    log.info("Done total=%d skipped=%d errors=%d", total, skipped, errors)
    write_reports(results, report_path, summary_path)


if __name__ == "__main__":
    main()
