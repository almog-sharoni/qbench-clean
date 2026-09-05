"""Check local links and anchors in an already-built MkDocs site (no network)."""
import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
        self.canonical = None

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "link" and attrs.get("rel") == "canonical":
            self.canonical = attrs.get("href")
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "name" in attrs:
            self.ids.add(attrs["name"])
        if "href" in attrs:
            self.links.append(attrs["href"])
        if "src" in attrs:
            self.links.append(attrs["src"])


def check(directory):
    root = Path(directory).resolve()
    pages = {}
    for path in root.rglob("*.html"):
        parsed = Links()
        parsed.feed(path.read_text())
        pages[path] = parsed
    if not pages:
        raise SystemExit("No built HTML pages found")
    # MkDocs uses root-relative URLs in 404.html when site_url contains a
    # project prefix. Resolve those against the published site, not site/site/.
    index = pages.get(root / "index.html")
    canonical = index.canonical if index else None
    base_path = unquote(urlsplit(canonical).path) if canonical else "/"
    base_path = "/" + base_path.strip("/") + "/" if base_path.strip("/") else "/"
    failures = []
    checked = 0
    for path, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = unquote(url.path)
            if target.startswith("/"):
                if not target.startswith(base_path):
                    failures.append(f"{path.relative_to(root)}: outside site prefix {link}")
                    continue
                target_path = root / target[len(base_path):]
            elif target:
                target_path = path.parent / target
            else:
                target_path = path
            target_path = target_path.resolve()
            if target_path.is_dir():
                target_path /= "index.html"
            checked += 1
            if not target_path.is_relative_to(root) or not target_path.is_file():
                failures.append(f"{path.relative_to(root)}: missing {link}")
            elif url.fragment and target_path in pages and unquote(url.fragment) not in pages[target_path].ids:
                failures.append(f"{path.relative_to(root)}: missing anchor {link}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Checked {len(pages)} HTML pages and {checked} local links/anchors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default="site")
    check(parser.parse_args().directory)
