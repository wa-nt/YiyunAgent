import re
from html.parser import HTMLParser
from pathlib import Path

import httpx
import pymupdf


class _HtmlTextExtractor(HTMLParser):
    """把 HTML 拆成标题与正文纯文本，块级标签转成段落换行。"""

    # 不含 head：缺 </head> 的页面会让跳过计数永不归零，正文全丢；title 已单独提取
    _SKIP = {"script", "style", "noscript", "template"}
    _BLOCK = {
        "p",
        "div",
        "li",
        "tr",
        "td",
        "dd",
        "dt",
        "section",
        "article",
        "header",
        "footer",
        "nav",
        "aside",
        "main",
        "blockquote",
        "pre",
        "figure",
        "figcaption",
        "ul",
        "ol",
        "table",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.parts: list[str] = []
        self._skip = 0
        self._in_title = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title += 1
        elif tag in self._BLOCK:
            self.parts.append("\n\n")
        elif tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        elif tag == "title" and self._in_title:
            self._in_title -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        elif not self._skip:
            self.parts.append(data)


def _normalize(raw: str) -> str:
    """折叠行尾空白与连续空行，但保留行首缩进（代码笔记的缩进是内容）。"""
    raw = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    raw = re.sub(r"[ \t]+(?=\n)", "", raw)
    return re.sub(r"\n{3,}", "\n\n", raw).strip("\n")


def extract_html(html: str, fallback_title: str = "") -> tuple[str, str]:
    parser = _HtmlTextExtractor()
    parser.feed(html)
    parser.close()
    title = " ".join("".join(parser.title_parts).split())
    return title or fallback_title, _normalize("".join(parser.parts))


def load_markdown(path: str | Path) -> tuple[str, str]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or path.stem, text
    return path.stem, text


def load_pdf(path: str | Path) -> tuple[str, str]:
    path = Path(path)
    with pymupdf.open(path) as doc:
        title = (doc.metadata or {}).get("title") or ""
        text = "\n\n".join(page.get_text() for page in doc)
    return title.strip() or path.stem, _normalize(text)


def load_url(url: str) -> tuple[str, str]:
    resp = httpx.get(url, follow_redirects=True, timeout=30.0)
    resp.raise_for_status()
    return extract_html(resp.text, fallback_title=url)


_LOADERS = {".md": load_markdown, ".markdown": load_markdown, ".pdf": load_pdf}


def load(source: str | Path) -> tuple[str, str]:
    source = str(source)
    if source.startswith(("http://", "https://")):
        return load_url(source)
    loader = _LOADERS.get(Path(source).suffix.lower())
    if loader is None:
        raise ValueError(f"不支持的来源类型：{source}")
    return loader(source)