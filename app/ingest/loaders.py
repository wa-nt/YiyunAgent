import ipaddress
import re
import socket
from html.parser import HTMLParser
from pathlib import Path

import httpx
import pymupdf

# 单页抓取上限：无上限时一个超大页面（或无限流）就能把内存吃满
MAX_URL_BYTES = 20 * 1024 * 1024
# 重定向跳数上限，防止 A→B→A 这类循环把抓取卡死
MAX_URL_REDIRECTS = 5


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


def _assert_public_url(url: str) -> None:
    """只放行公网目标，堵住 SSRF（回环/私网/链路本地一律拒）。

    host 是 IP 字面量就直接判定；域名用 getaddrinfo 解析出的**全部**地址逐个判定——
    只看第一个等于放过「同一域名解析出多个地址、其中含内网」的情况。
    """
    host = httpx.URL(url).host
    if not host:
        raise ValueError(f"URL 缺少主机名：{url}")
    try:
        ips = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror as exc:
            raise ValueError(f"域名解析失败：{host}") from exc
        # 带 scope 的 IPv6（fe80::1%eth0）要先去掉 % 后缀才能解析
        ips = [ipaddress.ip_address(info[4][0].split("%")[0]) for info in infos]
    if not all(ip.is_global for ip in ips):
        raise ValueError(f"只允许抓取公网地址，已拒绝：{host}")


# ponytail: 域名解析与实际建连之间有 TOCTOU（rebinding）窗口，理论上可被换成内网地址；
# 个人知识库的量级下接受，要堵死只能自定义 transport 在校验过的 IP 上建连。
def load_url(url: str) -> tuple[str, str]:
    """流式抓取网页正文；手动逐跳跟随重定向，每一跳都重新做公网校验（防 302 打到内网）。"""
    with httpx.Client(follow_redirects=False, timeout=30.0) as client:
        for _ in range(MAX_URL_REDIRECTS + 1):
            _assert_public_url(url)
            with client.stream("GET", url) as resp:
                # follow_redirects=False 时 httpx 仍会给出下一跳请求，据此手动跟随
                if resp.next_request is not None:
                    url = str(resp.next_request.url)
                    continue
                resp.raise_for_status()
                pieces: list[bytes] = []
                total = 0
                for piece in resp.iter_bytes():
                    total += len(piece)
                    if total > MAX_URL_BYTES:
                        raise ValueError(
                            f"页面超过 {MAX_URL_BYTES // 1024 // 1024}MB 上限：{url}"
                        )
                    pieces.append(piece)
                # 网页自己声明的编码优先，缺失或不可识别时按 utf-8 且坏字节替换，别让单个字节炸掉整页
                html = b"".join(pieces).decode(resp.encoding or "utf-8", errors="replace")
                return extract_html(html, fallback_title=url)
    raise ValueError(f"重定向次数过多（上限 {MAX_URL_REDIRECTS} 跳）：{url}")


# .txt 走 markdown 读取器：都是 UTF-8 纯文本，标题同样取首个「# 」行、没有就取文件名
_LOADERS = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_markdown,
    ".pdf": load_pdf,
}


def load(source: str | Path) -> tuple[str, str]:
    source = str(source)
    if source.startswith(("http://", "https://")):
        return load_url(source)
    loader = _LOADERS.get(Path(source).suffix.lower())
    if loader is None:
        raise ValueError(f"不支持的来源类型：{source}")
    return loader(source)