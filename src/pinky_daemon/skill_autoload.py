"""Automatic skill loading by context (#1288).

The triggers live in ONE daemon config file (``PINKY_SKILL_AUTOLOAD_FILE``, default
``<data>/skill_autoload.json``), not in the skills: no schema migration, third-party skills stay
untouched, and approving the triggers means editing that file::

    {"skills": {"pde2": {"keywords": ["pde", "assignment_id"],
                         "patterns": ["4fa82044-[0-9a-f-]{27}"],
                         "always": false}}}

When a message delivered to an agent matches the triggers of one of its EQUIPPED skills, the daemon
appends the skill's instructions to the turn as ``agent_hint`` (context for the model, not stored
in the conversation history), once per session (``AutoloadTracker``). After a context restart the
skills that were active come back at the first message. The decision is deterministic: no model is
asked, and no message content leaves the daemon.

Off unless ``PINKY_SKILL_AUTOLOAD=1`` and the config file exists; any error leaves the message as it
was (a skill must never break delivery). Known limit: the tracker is in memory, so after a daemon
restart a skill may be injected once more in a resumed conversation (harmless).
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

MAX_INLINE_CHARS = 6000  # Satoshi 21/10 (B): full text up to ~6000 chars, else summary + load_skill
MAX_SKILLS_PER_MESSAGE = 3


@dataclass(frozen=True)
class Triggers:
    keywords: tuple[str, ...] = ()
    patterns: tuple[re.Pattern, ...] = ()
    always: bool = False
    max_chars: int = MAX_INLINE_CHARS  # per skill: up to this the full text, beyond the summary

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
    mx = raw.get("max_chars")
    mx = mx if isinstance(mx, int) and not isinstance(mx, bool) and 0 < mx <= 50_000 else None
    return Triggers(keywords, tuple(patterns), raw.get("always") is True, mx or MAX_INLINE_CHARS)


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


def render(skill: SkillText, max_chars: int | None = None) -> str:
    """The block added to a message: the full instructions, or, when they are longer than
    ``max_chars`` (the skill's own ``max_chars`` from the config file, by default
    ``MAX_INLINE_CHARS``), the description and the instruction to call ``load_skill``."""
    max_chars = max_chars or skill.triggers.max_chars
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


# --- the daemon side: config file, skills of an agent, one tracker -----------------------------
FLAG_ENV = "PINKY_SKILL_AUTOLOAD"
FILE_ENV = "PINKY_SKILL_AUTOLOAD_FILE"


def load_triggers(path: Path) -> dict[str, Triggers]:
    """The config file → skill name → triggers. A missing or broken file: no triggers."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as err:
        log.warning("skill_autoload: %s ignored: %s", path, err)
        return {}
    skills = raw.get("skills") if isinstance(raw, dict) else None
    if not isinstance(skills, dict):
        return {}
    out = {n: parse_triggers(t, n) for n, t in skills.items() if isinstance(n, str)}
    return {n: t for n, t in out.items() if not t.empty}


class Autoload:
    """The daemon's automatic skill loading: the triggers file (re-read when it changes), the
    skills equipped by an agent (``skills_for_agent(agent)`` → dicts with name, description,
    directive), one tracker, and a session generation per (agent, label) bumped by a context
    restart."""

    def __init__(self, skills_for_agent, path: Path) -> None:
        self._skills_for_agent = skills_for_agent
        self.path = Path(path)
        self._firma: tuple[int, int] | None = None  # (mtime_ns, size) of the file read
        self._triggers: dict[str, Triggers] = {}
        self.tracker = AutoloadTracker()
        self._generation: dict[tuple[str, str], int] = {}
        self.counts: Counter = Counter()  # skill → injections (telemetry)

    def _key(self, agent: str, label: str) -> str:
        return f"{label or 'main'}#{self._generation.get((agent, label or 'main'), 0)}"

    def triggers(self) -> dict[str, Triggers]:
        try:
            st = self.path.stat()
        except OSError:
            self._firma, self._triggers = None, {}
            return {}
        firma = (st.st_mtime_ns, st.st_size)
        if firma != self._firma:
            self._firma, self._triggers = firma, load_triggers(self.path)
        return self._triggers

    def skills(self, agent: str) -> list[SkillText]:
        trig = self.triggers()
        if not trig:
            return []
        out = []
        for d in self._skills_for_agent(agent) or ():
            name = d.get("name") if isinstance(d, dict) else None
            if name in trig:
                out.append(SkillText(name, d.get("description") or "", d.get("directive") or "",
                                     trig[name]))  # fmt: skip
        return sorted(out, key=lambda s: s.name)

    def hint(self, agent: str, label: str, prompt: str, agent_hint: str = "") -> str:
        pre, injected = self.tracker.prepare(agent, self._key(agent, label), prompt,
                                             self.skills(agent))  # fmt: skip
        for name, why in injected:
            self.counts[name] += 1
            log.info("skill_autoload: %s/%s ← %s (%s)", agent, label or "main", name, why)
        if not pre:
            return agent_hint
        return f"{agent_hint}\n\n{pre.rstrip()}" if agent_hint else f"\n\n{pre.rstrip()}"

    def restart(self, agent: str, label: str) -> None:
        label = label or "main"
        old = self._key(agent, label)
        self._generation[(agent, label)] = self._generation.get((agent, label), 0) + 1
        self.tracker.carry_over(agent, old, self._key(agent, label))


AUTOLOAD: Autoload | None = None


def configure(skills_for_agent, data_dir) -> Autoload | None:
    """Called once at daemon start. Off (None) unless ``PINKY_SKILL_AUTOLOAD=1``."""
    global AUTOLOAD
    if os.environ.get(FLAG_ENV, "0") != "1":
        AUTOLOAD = None
        return None
    path = os.environ.get(FILE_ENV) or str(Path(data_dir) / "skill_autoload.json")
    AUTOLOAD = Autoload(skills_for_agent, Path(path))
    log.info("skill_autoload: on, triggers from %s", path)
    return AUTOLOAD


def _label(session) -> str:
    cfg = getattr(session, "_config", None) or getattr(session, "config", None)
    return getattr(cfg, "label", "") or "main"


def hint_for(session, prompt: str, agent_hint: str = "") -> str:
    """For a session's ``send``: the ``agent_hint`` with the skills to load now. Never raises."""
    a = AUTOLOAD
    if a is None:
        return agent_hint
    try:
        return a.hint(getattr(session, "agent_name", "") or "", _label(session), prompt or "",
                      agent_hint or "")  # fmt: skip
    except Exception as err:  # noqa: BLE001 - a skill must never break delivery
        log.warning("skill_autoload: skipped for this message: %s", err)
        return agent_hint


def on_context_restart(agent: str, label: str) -> None:
    """A context restart of (agent, label): the active skills come back at the next message."""
    a = AUTOLOAD
    if a is None:
        return
    try:
        a.restart(agent, label)
    except Exception as err:  # noqa: BLE001
        log.warning("skill_autoload: restart of %s/%s not tracked: %s", agent, label, err)
