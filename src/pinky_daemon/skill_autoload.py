"""Automatic skill loading by context (#1288).

A skill can declare, in its ``SKILL.md`` frontmatter under ``metadata.auto_load``, the triggers that
make its instructions relevant::

    metadata:
      auto_load:
        keywords: ["pde", "assignment_id"]      # whole words, case-insensitive
        patterns: ["4fa82044-[0-9a-f-]{27}"]    # regular expressions
        always: false                           # true: wanted in every session

When a message delivered to an agent matches the triggers of one of its EQUIPPED skills, the daemon
prepends the skill's instructions to that message, once per session (``AutoloadTracker``). The
decision is deterministic: no model is asked, and no message content leaves the daemon.

Skills without ``auto_load`` behave exactly as before (only ``load_skill`` loads them).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

MAX_INLINE_CHARS = 6000  # Satoshi 21/10 (B): full text up to ~6000 chars, else summary + load_skill
MAX_SKILLS_PER_MESSAGE = 3


@dataclass(frozen=True)
class Triggers:
    keywords: tuple[str, ...] = ()
    patterns: tuple[re.Pattern, ...] = ()
    always: bool = False

    @property
    def empty(self) -> bool:
        return not (self.keywords or self.patterns or self.always)


def parse_triggers(raw, skill_name: str = "") -> Triggers:
    """``metadata.auto_load`` → ``Triggers``. Anything malformed is ignored (with a log line): a bad
    trigger must never break skill loading or message delivery."""
    if not isinstance(raw, dict):
        return Triggers()
    keywords = tuple(
        k.strip().casefold() for k in raw.get("keywords") or () if isinstance(k, str) and k.strip()
    )
    patterns = []
    for p in raw.get("patterns") or ():
        if not isinstance(p, str) or not p:
            continue
        try:
            patterns.append(re.compile(p, re.IGNORECASE))
        except re.error as err:
            log.warning("skill %s: auto_load pattern %r ignored: %s", skill_name, p, err)
    return Triggers(keywords, tuple(patterns), raw.get("always") is True)


def _keyword_in(keyword: str, text: str) -> bool:
    # a whole word or phrase: not inside another word («pde» does not match «update»)
    return re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text) is not None


def matches(triggers: Triggers, text: str) -> str | None:
    """What made the text match (for the log), or None."""
    if triggers.always:
        return "always"
    folded = (text or "").casefold()
    for k in triggers.keywords:
        if _keyword_in(k, folded):
            return f"keyword:{k}"
    for p in triggers.patterns:
        if p.search(text or ""):
            return f"pattern:{p.pattern}"
    return None


@dataclass(frozen=True)
class SkillText:
    name: str
    description: str
    directive: str
    triggers: Triggers


def render(skill: SkillText, max_chars: int = MAX_INLINE_CHARS) -> str:
    """The block prepended to a message: the full instructions, or, when they are longer than
    ``max_chars``, the description and the instruction to call ``load_skill``."""
    head = f"[skill loaded automatically: {skill.name}]"
    body = (skill.directive or "").strip()
    if body and len(body) <= max_chars:
        return f"{head}\n{body}\n[end of skill {skill.name}]"
    summary = (skill.description or "").strip() or "(no description)"
    return (f"{head}\n{summary}\nThe full instructions are long: call "
            f'load_skill("{skill.name}") before using it.\n[end of skill {skill.name}]')  # fmt: skip


@dataclass
class AutoloadTracker:
    """Which skills were already injected in each session (agent, session key)."""

    injected: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    # skills active in a previous session of the agent, injected again at the first message of
    # the new one (context restart): the new session starts without them in its context
    pending: dict[tuple[str, str], set[str]] = field(default_factory=dict)

    def carry_over(self, agent: str, old_key: str, new_key: str) -> None:
        """A new session replaces ``old_key``: its active skills come back at the first message."""
        was = self.injected.pop((agent, old_key), set()) | self.pending.pop((agent, old_key), set())
        if was:
            self.pending.setdefault((agent, new_key), set()).update(was)

    def prepare(self, agent: str, session_key: str, text: str,
                skills: list[SkillText]) -> tuple[str, list[tuple[str, str]]]:  # fmt: skip
        """The preamble for ``text`` (possibly empty) and the (skill, reason) pairs injected now.
        Each skill at most once per session, at most ``MAX_SKILLS_PER_MESSAGE`` per message, in
        the order of ``skills``; the skills carried over from a previous session come first."""
        done = self.injected.setdefault((agent, session_key), set())
        back = self.pending.pop((agent, session_key), set())
        blocks, injected = [], []
        ordered = [s for s in skills if s.name in back] + [s for s in skills if s.name not in back]
        for s in ordered:
            if len(injected) >= MAX_SKILLS_PER_MESSAGE:
                # what is carried over and did not fit waits for the next message
                rest = {x.name for x in ordered if x.name in back} - {n for n, _ in injected}
                if rest:
                    self.pending.setdefault((agent, session_key), set()).update(rest)
                break
            if s.name in done:
                continue
            if s.name in back:
                why = "carried over"
            elif s.triggers.empty:
                continue
            else:
                why = matches(s.triggers, text)
            if why is None:
                continue
            done.add(s.name)
            blocks.append(render(s))
            injected.append((s.name, why))
        return ("\n\n".join(blocks) + "\n\n") if blocks else "", injected

    def active(self, agent: str, session_key: str) -> frozenset[str]:
        return frozenset(self.injected.get((agent, session_key), ()))

    def forget(self, agent: str, session_key: str) -> None:
        self.injected.pop((agent, session_key), None)
