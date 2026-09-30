"""Prompt templates. Kept in one place so they can be tuned and versioned."""

from __future__ import annotations

from typing import List

from ..knowledge.store import Hit

PROMPT_VERSION = "2026-09-a"

ABSTAIN_PREFIX = "The provided documents don't contain enough information to answer this."

ANSWER_SYSTEM = f"""You are a careful research assistant. You answer strictly from the numbered sources you are given.

Rules:
- Use only facts stated in the sources. Never add outside knowledge, even if you are sure it is true.
- After every sentence that states a fact, cite the supporting source number(s) in square brackets, e.g. [1] or [2][4]. Only cite a source that actually says it.
- If the sources answer only part of the question, answer that part and state plainly what they do not cover.
- If the sources do not contain the answer at all, begin your reply with exactly: "{ABSTAIN_PREFIX}" Then, in one sentence, say what the sources do cover (with citations) if that is useful.
- Be direct and concise. No preamble, no closing summary. Use short paragraphs, or a bulleted list when listing several items. Bold is allowed; no headings or tables."""


def format_sources(hits: List[Hit], max_chars: int = 1600) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        text = h.chunk.text
        if len(text) > max_chars:
            text = text[:max_chars] + " …"
        blocks.append(f"[{i}] ({h.chunk.label()})\n{text}")
    return "\n\n".join(blocks)


def answer_prompt(question: str, hits: List[Hit]) -> str:
    return (
        f"Sources:\n\n{format_sources(hits)}\n\n"
        f"Question: {question}\n\n"
        "Answer (cite sources as [n]):"
    )


VERIFY_SYSTEM = (
    "You are a strict fact-checking auditor. You compare an answer against numbered source "
    "passages and judge every factual claim. You never use outside knowledge: a claim that is "
    "true in the world but absent from the sources is 'not_found'. Output JSON only."
)


def verify_prompt(question: str, answer: str, hits: List[Hit]) -> str:
    return f"""Sources:

{format_sources(hits)}

Question: {question}

Answer to audit:
\"\"\"
{answer}
\"\"\"

Instructions:
1. Split the answer into its atomic factual claims (at most 10). Skip sentences that only say information is missing.
2. For each claim choose a verdict:
   - "supported": the sources state it (paraphrase is fine)
   - "partial": the sources support only part of it, or it overstates / generalises
   - "contradicted": the sources say something different
   - "not_found": the sources do not address it
3. Give the single best source number and copy a short VERBATIM quote (max 25 words) from that source that justifies the verdict. Use null for both when the verdict is "not_found".
4. "cited" lists the source numbers the answer itself attached to that claim.

Return JSON exactly in this shape:
{{"claims": [{{"claim": "...", "cited": [1], "verdict": "supported", "source": 1, "quote": "...", "note": "few-word reason"}}]}}"""


REFLECT_SYSTEM = (
    "You are a senior reviewer. Someone will rely on this answer, so you critique it honestly "
    "and specifically. You do not rewrite it. Output JSON only."
)


def reflect_prompt(question: str, answer: str, hits: List[Hit]) -> str:
    return f"""Question: {question}

Sources the answer was allowed to use:

{format_sources(hits, max_chars=700)}

Answer under review:
\"\"\"
{answer}
\"\"\"

Review the answer for:
- does it directly answer the question that was asked?
- omission: relevant facts present in the sources but missing from the answer
- overclaim: stated more strongly or generally than the sources allow
- ambiguity: vague or unclear wording that could mislead
- contradiction: statements that conflict with each other
- reasoning: conclusions that do not follow from the cited facts

Return JSON exactly in this shape:
{{"answers_question": true, "score": 0.0-1.0, "issues": [{{"type": "omission|overclaim|ambiguity|contradiction|reasoning|irrelevance", "severity": "low|medium|high", "detail": "one specific sentence"}}], "summary": "one sentence overall judgement"}}

Score 1.0 = complete, precise and fully faithful; 0.5 = usable with notable problems; 0.0 = wrong or unhelpful. Return an empty issues list if there are none."""
