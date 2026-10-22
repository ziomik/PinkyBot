"""#1288: automatic skill loading by context (pure part: triggers, matching, rendering, tracker)."""

from pinky_daemon import skill_autoload as SA  # noqa: N812


def _skill(name, directive="Do the thing.", description="A skill.", **auto):
    return SA.SkillText(name, description, directive, SA.parse_triggers(auto, name))


# --- triggers -------------------------------------------------------------------------------
def test_no_auto_load_means_no_triggers():
    for raw in (None, {}, "pde", [], {"keywords": []}, {"always": "yes"}):
        assert SA.parse_triggers(raw).empty


def test_a_bad_pattern_is_ignored_and_the_rest_stays(caplog):
    t = SA.parse_triggers({"keywords": [" PDE ", "", 7], "patterns": ["(", "ab+c"]}, "x")
    assert t.keywords == ("pde",) and [p.pattern for p in t.patterns] == ["ab+c"]
    assert "auto_load pattern '(' ignored" in caplog.text


# --- matching -------------------------------------------------------------------------------
def test_a_keyword_matches_a_whole_word_case_insensitively():
    t = SA.parse_triggers({"keywords": ["pde", "assignment id"]})
    assert SA.matches(t, "Report to the PDE now") == "keyword:pde"
    assert SA.matches(t, "the Assignment ID is …") == "keyword:assignment id"
    assert SA.matches(t, "please update the file") is None  # «pde» inside «update»
    assert SA.matches(t, "") is None


def test_a_pattern_matches_anywhere_and_always_matches_everything():
    t = SA.parse_triggers({"patterns": [r"4fa82044-[0-9a-f-]{8}"]})
    assert SA.matches(t, "progetto 4fa82044-2985-4fc4") == r"pattern:4fa82044-[0-9a-f-]{8}"
    assert SA.matches(t, "progetto 1e68101b") is None
    assert SA.matches(SA.parse_triggers({"always": True}), "anything") == "always"


# --- rendering ------------------------------------------------------------------------------
def test_a_short_skill_is_injected_whole_a_long_one_as_summary_and_load_skill():
    short = _skill("pde2", "Step 1.\nStep 2.")
    assert SA.render(short) == ("[skill loaded automatically: pde2]\nStep 1.\nStep 2.\n"
                                "[end of skill pde2]")  # fmt: skip
    long = _skill("big", "x" * (SA.MAX_INLINE_CHARS + 1), description="Big rules.")
    out = SA.render(long)
    assert "x" * 100 not in out and "Big rules." in out and 'load_skill("big")' in out
    assert "y" * SA.MAX_INLINE_CHARS in SA.render(_skill("edge", "y" * SA.MAX_INLINE_CHARS))


# --- tracker --------------------------------------------------------------------------------
def test_a_matching_skill_is_injected_once_per_session():
    tr = SA.AutoloadTracker()
    skills = [_skill("pde2", keywords=["pde"]), _skill("other", keywords=["zzz"])]
    pre, inj = tr.prepare("eng", "s1", "lavoro sul PDE", skills)
    assert inj == [("pde2", "keyword:pde")] and pre.startswith("[skill loaded automatically: pde2]")
    assert pre.endswith("\n\n")
    assert tr.prepare("eng", "s1", "ancora il PDE", skills) == ("", [])  # già nella sessione
    assert tr.prepare("eng", "s2", "PDE", skills)[1] == [("pde2", "keyword:pde")]  # altra sessione
    assert tr.prepare("sat", "s1", "PDE", skills)[1] == [("pde2", "keyword:pde")]  # altro agente
    assert tr.active("eng", "s1") == {"pde2"}


def test_skills_without_triggers_or_not_matching_are_never_injected():
    tr = SA.AutoloadTracker()
    assert tr.prepare("eng", "s", "pde pde", [_skill("plain"), _skill("x", keywords=["y"])]) == (
        "", [])  # fmt: skip


def test_several_skills_in_order_and_at_most_the_cap_per_message():
    tr = SA.AutoloadTracker()
    skills = [_skill(f"s{i}", keywords=["go"]) for i in range(SA.MAX_SKILLS_PER_MESSAGE + 1)]
    _, inj = tr.prepare("eng", "s", "go", skills)
    assert [n for n, _ in inj] == [f"s{i}" for i in range(SA.MAX_SKILLS_PER_MESSAGE)]
    _, inj = tr.prepare("eng", "s", "go again", skills)  # quella rimasta fuori, al messaggio dopo
    assert [n for n, _ in inj] == [f"s{SA.MAX_SKILLS_PER_MESSAGE}"]


def test_after_a_restart_the_active_skills_come_back_first_without_a_trigger():
    tr = SA.AutoloadTracker()
    skills = [_skill("a", keywords=["aaa"]), _skill("pde2", keywords=["pde"])]
    tr.prepare("eng", "old", "PDE", skills)
    tr.carry_over("eng", "old", "new")
    assert tr.active("eng", "old") == frozenset()
    pre, inj = tr.prepare("eng", "new", "aaa, nessun PDE qui", skills)
    assert inj == [("pde2", "carried over"), ("a", "keyword:aaa")]
    assert pre.index("pde2") < pre.index("skill loaded automatically: a]")
    assert tr.prepare("eng", "new", "PDE", skills) == ("", [])


def test_a_carried_over_skill_that_does_not_fit_waits_for_the_next_message():
    tr = SA.AutoloadTracker()
    n = SA.MAX_SKILLS_PER_MESSAGE
    skills = [_skill(f"s{i}", keywords=[f"k{i}"]) for i in range(n + 1)]
    for i in range(n + 1):
        tr.prepare("eng", "old", f"k{i}", skills)
    tr.carry_over("eng", "old", "new")
    assert len(tr.prepare("eng", "new", "", skills)[1]) == n
    assert tr.prepare("eng", "new", "", skills)[1] == [(f"s{n}", "carried over")]
    assert tr.prepare("eng", "new", "", skills) == ("", [])


def test_a_carried_over_skill_no_longer_equipped_is_dropped():
    tr = SA.AutoloadTracker()
    tr.prepare("eng", "old", "pde", [_skill("pde2", keywords=["pde"])])
    tr.carry_over("eng", "old", "new")
    assert tr.prepare("eng", "new", "pde", []) == ("", [])


# --- daemon side: config file, equipped skills, sessions ------------------------------------
import json  # noqa: E402
import os  # noqa: E402

import pytest  # noqa: E402


def _equipped(*names):
    return lambda agent: [{"name": n, "description": f"{n} desc", "directive": f"{n} body"}
                          for n in names]  # fmt: skip


_TICK = [10**9]


def _write(path, skills):
    path.write_text(json.dumps({"skills": skills}))
    _TICK[0] += 10**9  # a new mtime even within the same second
    os.utime(path, ns=(_TICK[0], _TICK[0]))


def test_triggers_come_from_the_config_file_and_reload_when_it_changes(tmp_path):
    f = tmp_path / "skill_autoload.json"
    a = SA.Autoload(_equipped("pde2", "other"), f)
    assert a.skills("eng") == []  # no file: nothing
    _write(f, {"pde2": {"keywords": ["pde"]}, "ghost": {"keywords": ["x"]}, "empty": {}})
    assert [s.name for s in a.skills("eng")] == ["pde2"]  # only equipped skills with triggers
    _write(f, {"other": {"always": True}})
    assert [s.name for s in a.skills("eng")] == ["other"]
    f.write_text("{not json")
    _TICK[0] += 10**9
    os.utime(f, ns=(_TICK[0], _TICK[0]))
    assert a.skills("eng") == []


class _Sess:
    def __init__(self, agent="eng", label="main"):
        self.agent_name = agent
        self._config = type("C", (), {"label": label})()


def test_the_hint_appends_the_skill_once_and_comes_back_after_a_restart(tmp_path, monkeypatch):
    f = tmp_path / "skill_autoload.json"
    _write(f, {"pde2": {"keywords": ["pde"]}})
    monkeypatch.setenv(SA.FLAG_ENV, "1")
    monkeypatch.delenv(SA.FILE_ENV, raising=False)
    a = SA.configure(_equipped("pde2"), tmp_path)
    assert a is SA.AUTOLOAD and a.path == f
    s = _Sess()
    h = SA.hint_for(s, "report to the PDE", " [reply on telegram]")
    assert h.startswith(" [reply on telegram]\n\n[skill loaded automatically: pde2]\npde2 body")
    assert SA.hint_for(s, "PDE again", "") == ""
    assert SA.hint_for(_Sess(label="side"), "PDE", "").startswith(
        "\n\n[skill loaded"
    )  # other label
    SA.on_context_restart("eng", "main")
    assert "pde2 body" in SA.hint_for(s, "hello, nothing here", "")  # carried over
    assert a.counts["pde2"] == 3


def test_off_by_default_and_never_raising(tmp_path, monkeypatch):
    monkeypatch.delenv(SA.FLAG_ENV, raising=False)
    assert SA.configure(_equipped("pde2"), tmp_path) is None
    assert SA.hint_for(_Sess(), "PDE", "keep") == "keep"
    SA.on_context_restart("eng", "main")  # no-op
    monkeypatch.setenv(SA.FLAG_ENV, "1")

    def boom(agent):
        raise RuntimeError("store down")

    _write(tmp_path / "skill_autoload.json", {"pde2": {"keywords": ["pde"]}})
    SA.configure(boom, tmp_path)
    assert SA.hint_for(_Sess(), "PDE", "keep") == "keep"


@pytest.fixture(autouse=True)
def _reset_autoload():
    yield
    SA.AUTOLOAD = None


def test_a_skill_can_have_its_own_inline_limit_in_the_config():
    big = "z" * 9000
    assert "z" * 9000 not in SA.render(_skill("pde2", big, keywords=["pde"]))
    assert "z" * 9000 in SA.render(_skill("pde2", big, keywords=["pde"], max_chars=20000))
    for bad in (0, -1, True, "20000", 60000):
        assert (
            SA.parse_triggers({"keywords": ["x"], "max_chars": bad}).max_chars
            == SA.MAX_INLINE_CHARS
        )
