"""Prep Engine: RAG over CV/JD/company + LLM question generation."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
import warnings
from functools import lru_cache
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer
import chromadb

from ..config import settings
from .llm import achat_completion, extract_faithful_json

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Return the cached phrase-embedding model (loaded exactly once)."""
    logger.info("Loading embedding model 'all-MiniLM-L6-v2' (first call only).")
    return SentenceTransformer("all-MiniLM-L6-v2")


PREP_SYSTEM_PROMPT = """You are an expert interview coach.

Output ONLY ONE complete, valid JSON object. Everything you write must be inside that single JSON object.

STRICT OUTPUT RULES:
- Your entire response must start with "{" and end with "}" — no text before or after.
- No preamble, no "thinking process", no reasoning, no markdown code fences (```), no comments, no bullet lists outside the JSON.
- Never refuse, and never emit placeholder text such as "Unable to generate", "Please provide", "I can't", or empty strings. Always produce real, tailored content derived ONLY from the provided context.
- The JSON must be complete and never truncated.

HALLUCINATION PREVENTION RULES (STRICT — these override every other instruction):
1. You must ONLY use the information explicitly provided in the candidate's CV.
2. Do NOT invent, assume, or hallucinate any experience, tools, metrics, or projects that are not written in the CV.
3. If the CV lacks a specific example for a question, use the closest real experience from the CV. If nothing exists, state clearly: "I don't have direct experience with this, but I would approach it by..." — do NOT make up fake stories.
4. If a skill is not in the CV, mention it as a skill gap; do NOT pretend the candidate used it.
5. Do not over-exaggerate. Keep the answers realistic based only on the CV text.
6. Do not mix up the dates of different jobs. Use the exact dates provided in the CV for each specific role. If a date is missing, do not invent it.
7. Never move a date from one CV entry to a different entry. When the CV shows a date span for one entry and a single date for another entry, keep each with its own entry: do not swap, merge, or invent the missing endpoint.
8. Do not mash separate CV rows into a single job. Never mark a role, employer, department, and date as one position unless the CV text explicitly places them together on the same entry.

GROUNDING REQUIREMENTS:
- Use the candidate's name exactly as written in the CV. Never invent or change it.
- Use the candidate's education exactly as written in the CV. Never invent degrees, universities, dates, or graduation claims; copy each degree and institution exactly as it appears.
- Use the candidate's job roles, employers, and industries exactly as written in the CV. Never rename a role, change an employer, or change an employer's industry. Every employer and job you cite must come from the CV you are given; never assume a company name or role that is not in the CV.
- Every employer, role, skill, tool, certification, project, and metric you cite in an answer MUST be traceable to the candidate CV context below. If a fact is not present in the CV context, do not include it.
- When a question concerns a tool or technique that is not on the CV, do NOT claim the candidate used it (e.g., do not add Python, SQL, or Tableau if they are not in the CV). Either name it as a skill gap or answer honestly with the "I don't have direct experience with this, but I would approach it by..." phrasing.
- A certification is proof of learning — it is NOT a job, internship, or employment. Never describe a certification as work experience, never claim the candidate "worked" on a certification, and never attach a certification's start/end dates to an employer.
- Do not merge different CV sections. Facts about Skills stay skills, facts about Certifications stay certifications, and facts about Experience stay jobs. Never move a skill or certification into the Experience section to create a role.
- The ONLY jobs, roles, internships, and employers you may cite are the ones explicitly listed under the CV's Experience section. Do not create additional roles, internships, projects-as-jobs, or employers that are not explicitly written there.
- The Experience section of this CV may have been flattened from a two-column table. A bullet such as "ACME Courier Service || Internship at ExampleSoft" is ONE row of that table: "||" separates the left-column entry from the right-column entry. Entries in the same column stack across consecutive rows and describe the SAME job or employer. Match each date with the role, employer, or department on the SAME side of the "||" — never carry a date from one column to a role in the other column, and never merge the two halves of a "||" row into one job. Attribute a role to an employer ONLY when the CV text explicitly pairs them; otherwise describe the role without naming an employer rather than guessing.
- Never call a role an "internship" unless the CV explicitly labels THAT specific role as an internship. A role written as "Dept."/'Department' or "Customer Service" is a job or position, not an internship, unless the CV text explicitly calls it an internship.
- When the "VERIFIED CV FACTS" block is present, treat it as authoritative ground truth: every Answer fact must be traceable to a line in that block, and no Answer fact may contradict it.

Produce EXACTLY this schema, with 5 technical questions, 3 behavioral questions, exactly 5 positioning tips, and exactly 5 skill gaps:

{
  "technical_questions": [
    {"question": "A specific technical question about the JD", "answer": "Full STAR answer: Situation, Task, Action, Result"}
  ],
  "behavioral_questions": [
    {"question": "A behavioral question aligned with the company's values", "answer": "Full STAR answer: Situation, Task, Action, Result"}
  ],
  "positioning_tips": ["tip", "tip", "tip", "tip", "tip"],
  "skill_gaps": ["gap", "gap", "gap", "gap", "gap"]
}

Example (illustrates the format AND honest grounding only; generate your own content for the given CV/JD/company):

For illustration, the example candidate "Maria Lopez" has: a BBA, a customer service role at a courier company, a frontend internship, Microsoft Office and basic HTML/CSS skills — and NO Python, NO SQL, NO Tableau.

{
  "technical_questions": [
    {"question": "How would you design a REST API in FastAPI for a document search service?", "answer": "S: I have not used FastAPI in my roles. T: The job requires API design in FastAPI. A: I don't have direct experience with this, but I would approach it by starting from the web fundamentals I learned in my frontend internship, then following FastAPI's official docs to stand up a wireframe service and iterate. R: I would validate the design with a working prototype rather than guessing."}
  ],
  "behavioral_questions": [
    {"question": "Tell me about a time you handled a difficult customer.", "answer": "S: In my customer service role at a courier company, a customer was upset about a delayed shipment. T: I needed to resolve the complaint while keeping the customer informed. A: I tracked the parcel at each step and communicated updates. R: The customer's issue was resolved and they acknowledged the clear communication."}
  ],
  "positioning_tips": ["Point your customer-service problem solving at the company's workflow."],
  "skill_gaps": ["Python appears in the job description but not on the CV."]
}

CONTENT RULES:
- Each technical question must be specific to the provided job description.
- Each behavioral question must reflect the provided company's values.
- Every question needs a complete STAR answer (Situation, Task, Action, Result) using ONLY real facts from the candidate CV.
- Keep every STAR answer SHORT: at most 3 sentences (Situation+Task in one sentence, Action in one, Result in one).
- Positioning tips must explain why THIS candidate (as documented in the CV) is a good fit for THIS company.
- Skill gaps must list skills appearing in the job description that are missing or weak on the CV.
- Never attribute to the candidate a job title, employer, industry, degree, tool, certification, project, or metric that is not present in the CV context.

Return only the single JSON object."""


_SECTION_HEADERS: List[tuple] = [
    ("professional summary", "PROFESSIONAL SUMMARY"),
    ("summary", "PROFESSIONAL SUMMARY"),
    ("professional experience", "EXPERIENCE"),
    ("work experience", "EXPERIENCE"),
    ("experience", "EXPERIENCE"),
    ("education", "EDUCATION"),
    ("technical skills", "SKILLS"),
    ("skills", "SKILLS"),
    ("qualifications", "SKILLS"),
    ("certifications", "CERTIFICATIONS"),
    ("certification", "CERTIFICATIONS"),
    ("projects", "PROJECTS"),
    ("project", "PROJECTS"),
    ("courses", "COURSES"),
    ("languages", "LANGUAGES"),
    ("achievements", "ACHIEVEMENTS"),
    ("awards", "ACHIEVEMENTS"),
]


def _facts_sheet(cv_text: str) -> str:
    """Split raw CV text into explicitly labeled sections so the LLM can
    never confuse skills/certifications with real jobs or employers."""
    lines = [
        line.strip() for line in (cv_text or "").splitlines() if line.strip()
    ]
    if not lines:
        return ""

    def find_headers(lines: List[str]) -> List[tuple]:
        found: List[tuple] = []
        for i, line in enumerate(lines):
            low = line.lower().strip(". :")
            for label, section in _SECTION_HEADERS:
                if low == label:
                    found.append((i, section))
                    break
        return found

    header_spans = find_headers(lines)
    if not header_spans:
        return "[VERIFIED CV FACTS]\n" + "\n".join(f"- {l}" for l in lines)

    spans = sorted(header_spans)
    out: List[str] = []
    prev_idx = -1
    prev_label = "HEADER"
    for idx, label in spans:
        body = lines[prev_idx + 1 : idx] if prev_idx >= 0 else []
        if body:
            out.append(f"[SECTION: {prev_label}]")
            out.extend(f"- {l}" for l in body)
        prev_idx = idx
        prev_label = label
    body = lines[prev_idx + 1 :]
    if body:
        out.append(f"[SECTION: {prev_label}]")
        out.extend(f"- {l}" for l in body)

    return (
        "[VERIFIED CV FACTS (authoritative; extracted directly from the CV sections)]\n"
        + "\n".join(_split_adjacent_facts(l) for l in out)
    )


def _split_adjacent_facts(line: str) -> str:
    """Reconstruct flattened two-column CV rows as explicit row bullets so the
    LLM can see which employer, role, and dates belong on which side."""
    s = line
    # "ACME Courier Service InternshipatExampleSoft" -> "ACME Courier Service || Internship at ExampleSoft"
    s = re.sub(
        r"(\bCourier\s+Company)\s*([A-Za-z]*Internship)",
        r"\1 || \2",
        s,
        flags=re.IGNORECASE,
    )
    # "InternshipatExampleSoft" -> "Internship at ExampleSoft"
    s = re.sub(
        r"(Internship\s*at\s*)([A-Za-z]+)",
        r"Internship at \2",
        s,
        flags=re.IGNORECASE,
    )
    # "CustomerSrviceDept.(3Months) FrontendDeveloper" -> row with both halves
    s = re.sub(
        r"((?:[A-Za-z]*Dept\.?)?\s*\(\d+\s*Months\))\s+(?=[A-Za-z]*Developer)",
        r"\1 || ",
        s,
    )
    # "Jan2024-Mar2024 Jul2024" -> date span / single date row pair
    s = re.sub(
        r"([A-Za-z]+20\d\d-[A-Za-z]+20\d\d)\s+([A-Za-z]+20\d\d)",
        r"\1 || \2",
        s,
    )
    return s


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    """Split text into overlapping character chunks."""
    text = " ".join(text.split())
    if len(text) <= chunk_size:
        return [text] if text else []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _salvage_json(content: str) -> Dict[str, Any]:
    """Leniently salvage a guide dict from imperfect model JSON output."""
    text = (content or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM output.")
    candidate = text[start : end + 1]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not salvage guide JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Salvaged JSON is not an object.")
    return parsed


def validate_guide(guide: Any) -> Dict[str, Any]:
    """Ensure a parsed guide contains real, non-empty, non-placeholder content."""
    if not isinstance(guide, dict):
        raise ValueError("LLM returned a guide that is not a JSON object.")

    required = (
        "technical_questions",
        "behavioral_questions",
        "positioning_tips",
        "skill_gaps",
    )
    missing = [key for key in required if key not in guide]
    if missing:
        raise ValueError(f"Guide is missing required keys: {missing}")

    placeholder_markers = (
        "unable to generate",
        "please provide",
        "i can't",
        "no job description",
        "no candidate cv",
        "no company",
        "none provided",
    )

    def _real(text: Any) -> bool:
        if not isinstance(text, str):
            return False
        low = text.strip().lower()
        return bool(low) and not any(m in low for m in placeholder_markers)

    for key in ("technical_questions", "behavioral_questions"):
        items = guide.get(key, [])
        if not items:
            raise ValueError(f"Guide section '{key}' is empty.")
        real_count = 0
        for item in items:
            if isinstance(item, dict) and _real(item.get("question")) and _real(item.get("answer")):
                real_count += 1
            elif isinstance(item, str) and _real(item):
                real_count += 1
        if real_count == 0:
            raise ValueError(f"Guide section '{key}' contains no real questions/answers.")

    for key in ("positioning_tips", "skill_gaps"):
        items = guide.get(key, [])
        if not any(_real(item) for item in items):
            raise ValueError(f"Guide section '{key}' is empty or contains only placeholders.")

    return guide


@lru_cache(maxsize=1)
def load_fallback_guide() -> Dict[str, Any]:
    """Load the deterministic preview/crash-safe guide from JSON (cached)."""
    path = settings.prep_fallback_path
    logger.info("Loading fallback prep guide from %s", path)
    with open(path, "r", encoding="utf-8") as fh:
        guide: Dict[str, Any] = json.load(fh)
    guide.setdefault("_fallback", True)
    return guide


class PrepEngine:
    """Generate interview prep guides from CV, JD and company overview."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        # Unique per-instantiation collection so repeated generation never
        # collides and no cleanup/delete is required.
        self.collection_name = f"prep_{session_id}_{uuid.uuid4().hex[:8]}"
        self.cv_text: str = ""
        self.job_description: str = ""
        self.company_overview: str = ""

    def _collection(self):
        chroma_client = chromadb.PersistentClient(path=settings.chroma_dir)
        return chroma_client.get_or_create_collection(self.collection_name)

    def index_documents(
        self, cv_text: str, job_description: str, company_overview: str
    ) -> None:
        """Chunk and embed all input documents into a session-specific collection."""
        self.cv_text = cv_text
        self.job_description = job_description
        self.company_overview = company_overview
        embedder = get_embedder()
        collection = self._collection()

        documents: List[Dict[str, str]] = [
            {"text": chunk, "source": "cv"} for chunk in _chunk_text(cv_text)
        ] + [{"text": chunk, "source": "jd"} for chunk in _chunk_text(job_description)]
        if company_overview.strip():
            documents += [
                {"text": chunk, "source": "company"}
                for chunk in _chunk_text(company_overview)
            ]

        if not documents:
            raise ValueError("No content to index.")

        chunks = [d["text"] for d in documents]
        embeddings = embedder.encode(chunks, convert_to_numpy=True).tolist()
        ids = [f"{uuid.uuid4().hex}" for _ in documents]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": d["source"]} for d in documents],
        )
        logger.info(
            "Indexed %d chunks into collection '%s'.", len(documents), self.collection_name
        )

    def retrieve_context(self, query: str, n_results: int = 8) -> str:
        """Retrieve relevant chunks for a query and format them as context."""
        embedder = get_embedder()
        collection = self._collection()
        try:
            results = collection.query(
                query_embeddings=embedder.encode([query], convert_to_numpy=True).tolist(),
                n_results=n_results,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Retrieval failed: %s", exc)
            return ""
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        if not docs:
            return ""
        lines = []
        for doc_list, meta_list in zip(docs, metas):
            for doc, meta in zip(doc_list, meta_list):
                source = (meta or {}).get("source", "?")
                lines.append(f"[{source}] {doc}")
        return "\n\n".join(lines)

    async def generate(self) -> Dict[str, Any]:
        """Build the prompt from retrieval context and generate the guide."""
        cv = self._retrieve_all("candidate experience and skills")
        jd = self._retrieve_all("job requirements and qualifications")
        company = self._retrieve_all("company values and mission")

        cv_facts = _facts_sheet(self.cv_text) or "Not provided."

        context_block = (
            f"### Candidate CV facts (authoritative; ground every answer here)\n"
            f"{cv_facts}\n\n"
            f"### Candidate CV context (raw)\n{cv}\n\n"
            f"### Job Description context\n{jd}\n\n"
            f"### Company context\n{company or 'Not provided.'}"
        )
        logger.info(
            "Prep generation starting: cv=%d chars, jd=%d chars, company=%d chars, "
            "collection='%s'",
            len(cv), len(jd), len(company), self.collection_name,
        )

        messages = [
            {"role": "system", "content": PREP_SYSTEM_PROMPT},
            {"role": "user", "content": context_block},
        ]
        if not settings.has_llm():
            logger.warning("No LLM configured; returning fallback guide.")
            return load_fallback_guide()

        last_error: Exception | None = None
        for attempt in range(1, settings.prep_retries + 1):
            try:
                content = await asyncio.wait_for(
                    achat_completion(
                        messages,
                        temperature=0.2,
                        max_tokens=settings.prep_max_tokens,
                    ),
                    timeout=settings.llm_timeout,
                )
                logger.info(
                    "Prep LLM attempt %d/%d: received %d chars, starts=%r, "
                    "ends_with_brace=%s",
                    attempt,
                    settings.prep_retries,
                    len(content),
                    content[:80],
                    bool(content.strip()) and content.rstrip().endswith("}"),
                )
                if not content or not content.strip():
                    raise ValueError("LLM returned an empty response.")

                logger.debug(
                    "Prep LLM raw output (attempt %d):\n%s",
                    attempt,
                    content[:20000],
                )
                try:
                    guide = await extract_faithful_json(content)
                except Exception as exc:  # noqa: BLE001 - fall back to lenient parse
                    logger.warning(
                        "extract_faithful_json failed (attempt %d): %s; "
                        "attempting lenient salvage parse.",
                        attempt,
                        exc,
                    )
                    guide = _salvage_json(content)
                validate_guide(guide)
                logger.info(
                    "Prep guide generated on attempt %d: %d technical, %d behavioral, "
                    "%d tips, %d skill gaps.",
                    attempt,
                    len(guide.get("technical_questions", [])),
                    len(guide.get("behavioral_questions", [])),
                    len(guide.get("positioning_tips", [])),
                    len(guide.get("skill_gaps", [])),
                )
                return guide
            except asyncio.TimeoutError:
                logger.warning(
                    "LLM call exceeded %.0fs timeout (attempt %d/%d); "
                    "returning fallback guide.",
                    settings.llm_timeout,
                    attempt,
                    settings.prep_retries,
                )
                return load_fallback_guide()
            except Exception as exc:  # noqa: BLE001 - retry on any other failure
                last_error = exc
                logger.warning(
                    "Prep LLM attempt %d/%d failed: %s",
                    attempt,
                    settings.prep_retries,
                    exc,
                )
                messages = [
                    {
                        "role": "system",
                        "content": (
                            PREP_SYSTEM_PROMPT
                            + "\n\nIMPORTANT: Your previous reply could not be parsed "
                            "as a single, complete JSON object (it was empty, truncated, "
                            "or contained extra text). Reply now with a complete, valid "
                            "JSON object matching the schema above. No thinking, no "
                            "explanation, no other text."
                        ),
                    },
                    {"role": "user", "content": context_block},
                ]

        logger.warning(
            "Prep guide generation failed after %d attempts (%s); "
            "returning fallback guide.",
            settings.prep_retries,
            last_error,
        )
        return load_fallback_guide()

    def _retrieve_all(self, query: str) -> str:
        """Fetch a broad context window across sources for prompt injection."""
        embedder = get_embedder()
        collection = self._collection()
        try:
            results = collection.get(limit=200)
        except Exception as exc:  # pragma: no cover
            logger.warning("Collection get failed: %s", exc)
            return ""
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        if not docs:
            return ""
        lines = []
        for doc, meta in zip(docs, metas):
            source = (meta or {}).get("source", "?")
            lines.append(f"[{source}] {doc}")
        return "\n".join(lines[:60])

    @staticmethod
    def _fallback_guide(context: str) -> Dict[str, Any]:
        """Deterministic fallback guide when no LLM API key is configured."""
        logger.warning("No OPENAI_API_KEY configured; returning fallback guide.")
        sample_question = {
            "question": (
                "Tell me about a project where you overcame a significant challenge. "
                "(Adjust this template based on your CV and the JD.)"
            ),
            "answer": (
                "S: The project goals and context. T: My specific responsibility. "
                "A: The concrete actions I took and decisions I made. R: The measurable "
                "outcome and what I learned. Ground every element in a real example "
                "from your CV."
            ),
        }
        return {
            "technical_questions": [sample_question] * 5,
            "behavioral_questions": [sample_question] * 3,
            "positioning_tips": [
                "Map one quantifiable CV accomplishment to the company's flagship metric.",
                "Mirror the company's vocabulary from its mission and job posting.",
                "Prepare 2 questions that reference specifics from the company overview.",
                "Front-load the skills the JD repeats most often.",
                "Ask about the team's definition of success for this role in your first 90 days.",
            ],
            "skill_gaps": [
                "Compare the JD's required skills against your CV; list anything missing, "
                "and prepare a learning roadmap for each."
            ],
            "_fallback": True,
        }