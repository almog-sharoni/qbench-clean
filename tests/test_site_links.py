"""Documentation checks stay portable and do not import the QBench runtime."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "qbench_site_checker", Path(__file__).resolve().parents[1] / "scripts" / "check_site.py",
)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def test_project_prefix_resolves_404_assets_and_anchors(tmp_path):
    (tmp_path / "index.html").write_text(
        '<link rel="canonical" href="https://example.github.io/qbench-clean/">'
        '<h1 id="intro">Intro</h1>'
    )
    (tmp_path / "style.css").write_text("body {}")
    (tmp_path / "404.html").write_text(
        '<link href="/qbench-clean/style.css"><a href="/qbench-clean/#intro">Home</a>'
    )
    checker.check(tmp_path)


@pytest.mark.parametrize("link", ["/qbench-clean/missing.css", "/qbench-clean/#missing", "/different/", "/qbench-clean/../outside.css"])
def test_project_prefix_does_not_hide_invalid_links(tmp_path, link):
    (tmp_path / "index.html").write_text(
        '<link rel="canonical" href="https://example.github.io/qbench-clean/">'
        f'<a href="{link}">Broken</a>'
    )
    with pytest.raises(SystemExit):
        checker.check(tmp_path)


def test_root_hosting_without_canonical_url(tmp_path):
    (tmp_path / "index.html").write_text('<a href="/page.html#target">Page</a>')
    (tmp_path / "page.html").write_text('<h1 id="target">Target</h1>')
    checker.check(tmp_path)
