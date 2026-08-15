"""Compatibility imports for the split gap-analysis/follow-up package.

New code should import from ``app.services.chat.gap_and_followup``. This
module remains so persisted-chat and external in-process callers can migrate
without a flag day.
"""

from app.services.chat.gap_and_followup.followup_policy import (
    AnthropicFollowUpPolicy,
    FOLLOWUP_POLICY_PROVIDER,
    FOLLOWUP_POLICY_VERSION,
    FOLLOWUP_PROMPT_VERSION,
    build_clarified_query,
)
from app.services.chat.gap_and_followup.prompts import (
    FOLLOWUP_POLICY_SCHEMA,
    FOLLOWUP_POLICY_SYSTEM,
)
from app.services.chat.gap_and_followup.schemas import (
    ClarificationExchange,
    FollowUpDecision,
    FollowUpPolicy,
    FollowUpPolicyResult,
    FollowUpReasonCode,
)

__all__ = [
    "AnthropicFollowUpPolicy",
    "ClarificationExchange",
    "FOLLOWUP_POLICY_PROVIDER",
    "FOLLOWUP_POLICY_SCHEMA",
    "FOLLOWUP_POLICY_SYSTEM",
    "FOLLOWUP_POLICY_VERSION",
    "FOLLOWUP_PROMPT_VERSION",
    "FollowUpDecision",
    "FollowUpPolicy",
    "FollowUpPolicyResult",
    "FollowUpReasonCode",
    "build_clarified_query",
]
