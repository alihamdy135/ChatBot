"""
Router: decides how to handle a message once we know its language, sentiment,
and intent route.

DESIGN DECISION (document this in the assessment):
Complaints and negative-sentiment messages are NOT auto-escalated away from RAG.
Instead we prepend an empathetic acknowledgment before the RAG-generated answer,
and additionally set `priority_flag=True` so a real deployment could surface these
in a human-review queue alongside the auto-response. This was chosen over pure
escalation (i.e. never auto-responding to a complaint) because:
  1. Most "complaint" intent messages in the Bitext dataset still have a factual
     answer available in the knowledge base (e.g. a complaint about a late order
     still wants an order-status-style answer).
  2. Fully blocking auto-response on every complaint would leave frustrated
     customers waiting even when an immediate grounded answer is available.
  3. The priority_flag still gives a human reviewer visibility into every
     complaint, satisfying the "flag for priority handling" requirement without
     sacrificing response speed.
"""

SMALL_TALK_RESPONSES = {
    "greet": "Hi there! How can I help you with your order today?",
    "goodbye": "Thanks for reaching out — have a great day!",
    "thank_you": "You're very welcome! Let me know if there's anything else I can help with.",
}

OUT_OF_SCOPE_RESPONSE = (
    "I'm not able to help with that from here, but I can connect you with a "
    "human agent who can. Would you like me to escalate this?"
)


def route_message(intent_result: dict, sentiment_result: dict) -> dict:
    """
    Decides the handling path for a message.
    Returns a dict describing what the caller (main.py) should do next.
    """
    route = intent_result["route"]
    sentiment = sentiment_result["label"]

    priority_flag = (route == "complaint") or (sentiment == "negative")

    if route == "small_talk":
        # No RAG needed -- direct canned response.
        # We don't have the fine intent mapped to a specific greeting bucket here,
        # so default to a generic friendly reply; refine with fine_intent if needed.
        return {
            "handling": "direct_response",
            "response": "Hi there! How can I help you today?",
            "priority_flag": False,
            "use_rag": False,
        }

    if route == "out_of_scope":
        return {
            "handling": "out_of_scope",
            "response": OUT_OF_SCOPE_RESPONSE,
            "priority_flag": False,
            "use_rag": False,
        }

    # order_status, order_management, billing_and_refunds, account_management, complaint
    # all go through RAG. Complaints/negative sentiment get an apology prefix.
    needs_apology_prefix = (route == "complaint") or (sentiment == "negative")

    return {
        "handling": "rag",
        "response": None,  # filled in by RAG call in main.py
        "priority_flag": priority_flag,
        "use_rag": True,
        "needs_apology_prefix": needs_apology_prefix,
    }


APOLOGY_PREFIX = (
    "I'm sorry to hear you're having trouble with this — I understand the "
    "frustration, and I want to help sort it out right away. "
)
