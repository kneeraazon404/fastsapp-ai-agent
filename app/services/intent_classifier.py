"""
Agentic Feature 2 — Intent Classification & Routing
====================================================
Analyses each incoming message in a single structured API call to extract:

  * intent       — what the user is trying to do (faq, appointment, …)
  * sentiment    — emotional tone of the message
  * sentiment_score — numeric score in [-1.0, 1.0] for threshold comparisons
  * entities     — key data items (name, date, service type, …)

The result drives downstream routing:
  - ``appointment`` intent → entity_extractor takes over response generation
  - ``emergency`` or very-negative sentiment → escalation flag is set
  - ``faq`` / ``other`` → standard RAG-based response path
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

VALID_INTENTS = frozenset(
    {"faq", "appointment", "complaint", "emergency", "greeting", "farewell", "other"}
)
VALID_SENTIMENTS = frozenset({"positive", "neutral", "negative", "urgent"})

_CLASSIFICATION_PROMPT = """\
Analyse the following WhatsApp message and respond with a JSON object only \
(no markdown fences, no extra text).

Message: "{message}"

Return exactly this structure:
{{
  "intent": "<one of: faq, appointment, complaint, emergency, greeting, farewell, other>",
  "sentiment": "<one of: positive, neutral, negative, urgent>",
  "sentiment_score": <float -1.0 to 1.0>,
  "entities": {{
    "name": <string | null>,
    "date": <string | null>,
    "time": <string | null>,
    "service_type": <string | null>,
    "location": <string | null>
  }}
}}"""


@dataclass
class ClassificationResult:
    intent: str = "other"
    sentiment: str = "neutral"
    sentiment_score: float = 0.0
    entities: dict = field(default_factory=dict)
    requires_escalation: bool = False


def classify_message(user_message: str) -> ClassificationResult:
    """
    Classify *user_message* and return a ``ClassificationResult``.

    Falls back to safe defaults if the API call fails or the response
    cannot be parsed — the bot continues operating in degraded mode.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    prompt = _CLASSIFICATION_PROMPT.format(message=user_message.replace('"', '\\"'))

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=200,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        data: dict = json.loads(raw)
    except Exception as exc:
        logger.error("Intent classification failed: %s", exc)
        return ClassificationResult()

    intent = data.get("intent", "other")
    if intent not in VALID_INTENTS:
        intent = "other"

    sentiment = data.get("sentiment", "neutral")
    if sentiment not in VALID_SENTIMENTS:
        sentiment = "neutral"

    try:
        score = float(data.get("sentiment_score", 0.0))
        score = max(-1.0, min(1.0, score))
    except (TypeError, ValueError):
        score = 0.0

    raw_entities: dict = data.get("entities", {})
    entities = {k: v for k, v in raw_entities.items() if v is not None}

    requires_escalation = (
        intent == "emergency"
        or sentiment in ("negative", "urgent")
        or score <= settings.escalation_sentiment_threshold
    )

    result = ClassificationResult(
        intent=intent,
        sentiment=sentiment,
        sentiment_score=score,
        entities=entities,
        requires_escalation=requires_escalation,
    )
    logger.debug(
        "Classified '%s…' → intent=%s sentiment=%s(%.2f) escalate=%s",
        user_message[:40],
        intent,
        sentiment,
        score,
        requires_escalation,
    )
    return result
