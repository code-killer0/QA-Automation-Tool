import csv
import io
import json
import logging
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME            = "gpt-4o"
TEMPERATURE           = 0.3
MAX_DOC_CHARS         = 80_000   # truncate docs that would blow the context
MAX_RETRIES           = 3
RETRY_DELAY_SECONDS   = 4
SUPPORTED_EXTENSIONS  = {".txt", ".pdf", ".docx", ".md"}
OUTPUT_FILE           = "output.csv"
TARGET_PER_CATEGORY   = 50       # aim for this many unique prompts per category

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Document loaders ─────────────────────────────────────────────────────────
LOADERS = {
    ".txt":  lambda p: TextLoader(str(p), encoding="utf-8"),
    ".pdf":  lambda p: PyPDFLoader(str(p)),
    ".docx": lambda p: Docx2txtLoader(str(p)),
    ".md":   lambda p: UnstructuredMarkdownLoader(str(p)),
}

# ── Test categories (order is preserved in the output) ───────────────────────
TEST_CATEGORIES = [
    "End-to-End Testing",
    "Integration Testing",
    "Unit Testing",
    "Model Accuracy & Regression Testing",
    "Hallucination & Fabrication Testing",
    "Bias & Fairness Testing",
    "Adversarial & Security Testing",
    "Edge Case & Boundary Testing",
    "Data Quality & Pipeline Testing",
    "Performance & Scalability Testing",
    "Drift & Monitoring Testing",
    "Explainability & Auditability Testing",
]

# ── Prompt templates ──────────────────────────────────────────────────────────

LOOPHOLE_PROMPT = PromptTemplate(
    template="""
You are a senior QA engineer specialising in AI/ML systems.  You have deep
expertise in machine-learning pipelines, model serving, data quality, bias
& fairness, adversarial attacks, and production monitoring.

Below is the technical documentation for an AI/ML-powered system, followed
by its primary API endpoint.

=== DOCUMENTATION START ===
{doc_content}
=== DOCUMENTATION END ===

Primary endpoint: {end_point}

Analyse this system and produce a detailed, numbered list of **at least 15
loopholes and failure modes** organised under the following headings.  For
every item explain *why* it is a risk and give a concrete failure scenario.

1. **Model Accuracy & Reliability** — incorrect predictions, confidence
   calibration errors, silent degradation.
2. **Hallucination & Fabrication** — model generating plausible but false
   outputs, unsupported claims, invented data.
3. **Data Quality & Pipeline Integrity** — corrupt inputs, schema drift,
   missing values, encoding issues, stale training data.
4. **Bias & Fairness** — demographic bias, under-represented groups,
   disparate impact across protected classes.
5. **Adversarial Robustness** — prompt injection, input perturbation,
   model inversion, data poisoning.
6. **Edge Cases & Boundary Conditions** — out-of-distribution inputs,
   extreme values, empty/null inputs, concurrent requests, timeouts.
7. **Security & Privacy** — PII leakage, model theft, insecure endpoints,
   missing auth, injection attacks.
8. **Operational & Scalability** — latency spikes under load, memory
   leaks, GPU exhaustion, cold-start delays.
9. **Explainability & Auditability** — lack of explanations for
   decisions, missing audit logs, non-reproducible results.
10. **Monitoring & Drift** — no data-drift detection, no model-performance
    dashboards, silent concept drift.

Be thorough.  Do NOT assume anything that is not in the documentation.
""",
    input_variables=["doc_content", "end_point"],
)


# Per-category prompt — generates prompts for ONE category at a time.
QA_CATEGORY_PROMPT = PromptTemplate(
    template="""\
You are an expert QA test-case designer for AI/ML applications.

=== CONTEXT ===
Technical documentation:
{doc_content}

Known loopholes & failure modes:
{loopholes}

Primary endpoint: {end_point}
=== END CONTEXT ===

Generate exactly {target} unique, non-repetitive QA test prompts for the
following category ONLY:

  Category: {category}

Each prompt must be a **detailed, self-contained instruction** that tells
a tester exactly:
  - What setup/preconditions are needed before running the test.
  - The exact input payload or user action to perform.
  - Which endpoint or component is being targeted.
  - What the expected correct behaviour, response body, status code, or
    output is.
  - What constitutes a PASS and what constitutes a FAIL.

Every prompt must reference concrete endpoints, inputs or features from the
documentation — do NOT hallucinate endpoints or features that are not
documented.

For EVERY test prompt you MUST also produce a Tool_Call JSON block that
represents the assistant-side function call a tester or automated harness
would issue to exercise this test.  The Tool_Call must:
  - Use the real endpoint path and HTTP method implied by the documentation.
  - Include all relevant request arguments (headers, query params, body
    fields) as the "arguments" JSON string.
  - Use a descriptive function name that matches the action being tested
    (e.g. "call_prediction_endpoint", "submit_batch_inference_job",
    "fetch_audit_log", "trigger_drift_alert").
  - Follow this exact JSON structure (no extra keys):

{{
  "role": "assistant",
  "tool_calls": [
    {{
      "id": "call_<PROMPT_ID>",
      "type": "function",
      "function": {{
        "name": "<descriptive_snake_case_function_name>",
        "arguments": "<escaped JSON string with all request params>"
      }}
    }}
  ]
}}

Output format — **strict CSV** with FOUR columns and NO extra text before
or after the CSV block:

Prompt_ID,Test_Type,Description,Tool_Call
1,{category},"<full detailed prompt>","<Tool_Call JSON — inner double-quotes escaped as \"\">"
2,{category},"<full detailed prompt>","<Tool_Call JSON — inner double-quotes escaped as \"\">"
...

Rules:
- Quote BOTH the Description AND Tool_Call fields with double-quotes.
- Escape all inner double-quotes inside a field by doubling them ("").
- Prompt_ID must be a sequential integer starting at 1 (re-numbering
  happens after merging, so local IDs only need to be unique within this
  response).
- Test_Type must be exactly: {category}
- Description must be at least four sentences covering: preconditions,
  exact input, expected output/behaviour, and pass/fail criteria.
- Tool_Call must be a single-line JSON string (no newlines inside the
  CSV cell) with inner quotes escaped as "".
- Do NOT include markdown fences, commentary, or section headers — only
  the raw CSV rows with the header line.
- Every prompt must be meaningfully different — do NOT repeat the same
  scenario with trivial word changes.
- If you genuinely cannot produce {target} distinct prompts for this
  category given the documentation, produce as many as possible.
""",
    input_variables=["doc_content", "loopholes", "end_point", "category", "target"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def validate_url(url: str) -> str:
    """Return the URL if it looks syntactically valid, else raise."""
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "https://" + url
        parsed = urlparse(url)
    if not parsed.netloc:
        raise ValueError(f"Invalid URL: {url!r}")
    return url


def load_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    loader = LOADERS[ext](path)
    docs   = loader.load()
    content = "\n".join(d.page_content for d in docs)
    if len(content) > MAX_DOC_CHARS:
        log.warning(
            "Document is %d chars — truncating to %d to stay within context limits.",
            len(content), MAX_DOC_CHARS,
        )
        content = content[:MAX_DOC_CHARS] + "\n\n[… document truncated …]"
    return content


def invoke_with_retry(chain, inputs: dict, label: str = "chain") -> str:
    """Invoke a LangChain chain with exponential-backoff retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return chain.invoke(inputs)
        except Exception as exc:
            wait = RETRY_DELAY_SECONDS * attempt
            log.warning(
                "%s attempt %d/%d failed (%s). Retrying in %ds…",
                label, attempt, MAX_RETRIES, exc, wait,
            )
            if attempt == MAX_RETRIES:
                raise
            time.sleep(wait)


def _try_parse_tool_call(raw: str) -> str:
    """
    Validate that a Tool_Call cell contains parseable JSON.
    Returns the normalised single-line JSON string, or the raw string on failure.
    """
    try:
        obj = json.loads(raw)
        return json.dumps(obj, separators=(",", ":"))
    except (json.JSONDecodeError, TypeError):
        return raw.strip()


def parse_csv_output(raw: str, expected_category: str) -> list[list[str]]:
    """
    Parse the LLM's CSV text into rows of
    [Prompt_ID, Test_Type, Description, Tool_Call].

    • Strips markdown fences.
    • Skips the header row and any row whose ID is not a plain integer.
    • Handles extra commas inside Description by re-joining with last col
      treated as Tool_Call.
    • Normalises the Test_Type to the expected category name.
    """
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$",          "", text)
    text = text.strip()

    rows: list[list[str]] = []
    reader = csv.reader(io.StringIO(text))

    for parts in reader:
        parts = [p.strip() for p in parts]

        if len(parts) < 3:
            continue

        prompt_id = parts[0]
        test_type = parts[1]

        if len(parts) >= 4:
            description   = ", ".join(parts[2:-1])
            tool_call_raw = parts[-1]
        else:
            description   = parts[2]
            tool_call_raw = ""

        # Skip header
        if prompt_id.lower() == "prompt_id":
            continue

        # Skip non-integer IDs
        if not re.match(r"^\d+$", prompt_id):
            continue

        # Normalise test type to the expected category (model may vary wording)
        if expected_category.lower() not in test_type.lower():
            test_type = expected_category

        tool_call = _try_parse_tool_call(tool_call_raw) if tool_call_raw else ""

        rows.append([prompt_id, test_type, description, tool_call])

    return rows


def deduplicate(
    rows: list[list[str]],
    seen: set[str],
) -> tuple[list[list[str]], int]:
    """
    Remove rows whose description (lowercased, stripped) is already in *seen*.
    Updates *seen* in-place with accepted descriptions.
    Returns (accepted_rows, dropped_count).
    """
    accepted, dropped = [], 0
    for row in rows:
        key = row[2].lower().strip()
        if key in seen:
            dropped += 1
        else:
            seen.add(key)
            accepted.append(row)
    return accepted, dropped


def write_csv_header(output_path: str) -> None:
    """Write (or overwrite) the CSV with only the header row."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["Prompt_ID", "Test_Type", "Description", "Tool_Call"])


def append_csv_rows(rows: list[list[str]], output_path: str) -> None:
    """Append rows to an existing CSV file."""
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerows(rows)


def renumber_csv(output_path: str) -> int:
    """
    Re-read the CSV, assign sequential Prompt_IDs starting at 1, write back.
    Returns the total number of data rows.
    """
    with open(output_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows   = list(reader)

    for idx, row in enumerate(rows, start=1):
        row[0] = str(idx)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        writer.writerows(rows)

    return len(rows)


def quality_report(output_path: str) -> None:
    """Print a per-category breakdown so the user can spot gaps."""
    counts: dict[str, int] = {}
    missing_tool_calls = 0

    with open(output_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            test_type = row[1] if len(row) > 1 else "Unknown"
            counts[test_type] = counts.get(test_type, 0) + 1
            if len(row) < 4 or not row[3].strip():
                missing_tool_calls += 1

    total = sum(counts.values())
    log.info("── Quality report ──────────────────────────────")
    for cat in TEST_CATEGORIES:
        n = counts.get(cat, 0)
        flag = "  ⚠ below target" if n < TARGET_PER_CATEGORY else ""
        log.info("  %-44s %3d prompts%s", cat, n, flag)
    log.info("  %-44s %3d prompts", "TOTAL", total)
    if missing_tool_calls:
        log.warning(
            "  %d prompt(s) have no Tool_Call — review raw_llm_output/ for details.",
            missing_tool_calls,
        )
    log.info("────────────────────────────────────────────────")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ---- Collect inputs ----
    file_path = input("Enter the path to your documentation file: ").strip()
    path = Path(file_path)
    if not path.exists():
        log.error("File not found: %s", file_path)
        sys.exit(1)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        log.error(
            "Unsupported file type '%s'. Supported: %s",
            path.suffix, ", ".join(sorted(SUPPORTED_EXTENSIONS)),
        )
        sys.exit(1)

    raw_url = input("Enter the URL of the primary endpoint: ").strip()
    try:
        end_point = validate_url(raw_url)
    except ValueError as exc:
        log.error(str(exc))
        sys.exit(1)

    # ---- Load document ----
    log.info("Loading document: %s", path)
    doc_content = load_document(path)
    log.info("Document loaded — %d characters.", len(doc_content))

    # ---- LLM setup ----
    model  = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    parser = StrOutputParser()

    # ---- Chain 1: loophole analysis (done once, shared across categories) ----
    log.info("Running AI/ML loophole analysis…")
    chain_loopholes = LOOPHOLE_PROMPT | model | parser
    loopholes: str  = invoke_with_retry(
        chain_loopholes,
        {"doc_content": doc_content, "end_point": end_point},
        label="loophole-analysis",
    )
    log.info("Loophole analysis complete (%d chars).", len(loopholes))

    # ---- Prepare output file ----
    write_csv_header(OUTPUT_FILE)
    raw_output_dir = Path("raw_llm_output")
    raw_output_dir.mkdir(exist_ok=True)

    # Global dedup set — prevents the same description appearing twice across
    # different categories.
    seen_descriptions: set[str] = set()

    # ---- Chain 2: per-category QA generation ----
    chain_qa = QA_CATEGORY_PROMPT | model | parser
    category_summary: dict[str, int] = {}

    for cat_index, category in enumerate(TEST_CATEGORIES, start=1):
        log.info(
            "── [%d/%d] Generating prompts for: %s",
            cat_index, len(TEST_CATEGORIES), category,
        )

        # Request prompts for this category
        raw_csv: str = invoke_with_retry(
            chain_qa,
            {
                "doc_content": doc_content,
                "loopholes":   loopholes,
                "end_point":   end_point,
                "category":    category,
                "target":      TARGET_PER_CATEGORY,
            },
            label=f"qa-{category}",
        )

        # Save raw output for post-mortem inspection
        safe_name = re.sub(r"[^\w]+", "_", category).strip("_").lower()
        (raw_output_dir / f"{cat_index:02d}_{safe_name}.txt").write_text(
            raw_csv, encoding="utf-8"
        )

        # Parse and deduplicate
        rows = parse_csv_output(raw_csv, expected_category=category)
        rows, dropped = deduplicate(rows, seen_descriptions)

        if dropped:
            log.info(
                "  Dropped %d duplicate prompt(s) for '%s'.", dropped, category
            )

        if not rows:
            log.warning(
                "  No valid rows parsed for '%s'. "
                "Check raw_llm_output/%02d_%s.txt",
                category, cat_index, safe_name,
            )
            category_summary[category] = 0
            continue

        # Append to CSV immediately (incremental persistence)
        append_csv_rows(rows, OUTPUT_FILE)
        category_summary[category] = len(rows)

        log.info(
            "  Saved %d prompt(s) for '%s' → %s",
            len(rows), category, OUTPUT_FILE,
        )

    # ---- Final renumber & report ----
    total = renumber_csv(OUTPUT_FILE)
    log.info("Re-numbered all Prompt_IDs — %d total rows in %s.", total, OUTPUT_FILE)

    quality_report(OUTPUT_FILE)

    # Human-readable summary
    print(f"\n{'─'*52}")
    print(f"  {'Category':<44}  {'Count':>5}")
    print(f"{'─'*52}")
    for cat in TEST_CATEGORIES:
        n    = category_summary.get(cat, 0)
        flag = "  ⚠" if n < TARGET_PER_CATEGORY else ""
        print(f"  {cat:<44}  {n:>5}{flag}")
    print(f"{'─'*52}")
    print(f"  {'TOTAL':<44}  {total:>5}")
    print(f"{'─'*52}")
    print(f"\nDone — {total} test prompts written to '{OUTPUT_FILE}'")
    print(f"Raw LLM outputs saved to '{raw_output_dir}/' for inspection.")


if __name__ == "__main__":
    main()