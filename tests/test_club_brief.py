"""The club brief must never print a missing value."""
import importlib.util
import re

import pytest

from abs_judgment.config import PROCESSED, ROOT

LABEL = "mlb_2026"
_spec = importlib.util.spec_from_file_location("club_brief", ROOT / "scripts" / "11_club_brief.py")
brief = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(brief)

NEEDED = [f"pitch_table_{LABEL}.parquet", f"judgment_{LABEL}.csv", f"token_value_{LABEL}.csv", f"policy_simulation_{LABEL}.csv",
          f"policy_seeds_{LABEL}.csv", f"policy_sigma_{LABEL}.csv", f"policy_belief_{LABEL}.json"]


def test_check_rejects_nan_outside_scripts_only():
    with pytest.raises(AssertionError):
        brief.check_rendered("<p>correlation +nan across regulars</p>")
    with pytest.raises(AssertionError):
        brief.check_rendered("<td>None</td>")
    brief.check_rendered("<p>Fernando, finance</p><script>const x = undefined; NaN;</script>")


@pytest.mark.skipif(not all((PROCESSED / f).exists() for f in NEEDED), reason="processed tables not built")
def test_brief_renders_without_nan():
    html = brief.render(brief.build_data(LABEL), LABEL)
    text = re.sub(r"<script\b.*?</script>", "", html, flags=re.S)
    assert not re.search(r"\b(?:nan|NaN|None|undefined)\b", text)
    data = re.search(r"const DATA = (.*?);\n", html).group(1)
    assert "NaN" not in data and "Infinity" not in data
