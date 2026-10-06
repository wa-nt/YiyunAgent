# Desktop sidebar navigation icon research

- **Date:** 2026-10-06
- **Scope:** Claude Code, ChatGPT, and Codex desktop sidebars; recommendations for this repository's navigation UI.
- **Why this filename:** `docs/` contains dated research/review notes, but `docs/product-review-2026-10-04.md` is an existing, broad product-audit report rather than a source-research note. This focused note therefore uses the requested fallback filename.

## Evidence and limits

| Product | Primary-source evidence | Finding that can be stated safely | Limitation |
|---|---|---|---|
| Claude Code | [Anthropic Claude Code documentation](https://code.claude.com/docs/en/desktop) | Anthropic publishes a first-party desktop documentation entry, so it is the correct canonical location to consult for product behavior and screenshots when available. | The page was inaccessible from this research environment (HTTP 403). The public [Anthropic Claude Code repository](https://github.com/anthropics/claude-code) is primarily the CLI/plugin repository and does not expose a documented desktop sidebar icon specification. No icon names, pixel sizes, stroke widths, or spacing are asserted here. |
| ChatGPT | [OpenAI Help: ChatGPT Windows app](https://help.openai.com/en/articles/9982051-using-the-chatgpt-windows-app) | OpenAI's first-party help center is the appropriate source for the desktop app's navigation behavior. | The page was inaccessible from this environment (HTTP 403), and no publicly accessible first-party design-token/spec source was found. Therefore this note does **not** claim that any particular glyph, size, stroke, or spacing is the ChatGPT implementation. |
| Codex | [OpenAI Codex app documentation](https://developers.openai.com/codex/app/) and the [official OpenAI Codex repository](https://github.com/openai/codex) | OpenAI's developer documentation and repository are the primary sources for Codex product/API and open-source implementation material. | The app page was inaccessible from this environment (HTTP 403). The repository exposes Codex source and docs, but no verified desktop sidebar icon design spec was located. Do not treat a CLI/TUI symbol or repository asset as proof of the desktop app's icon treatment. |

**Research boundary:** Product screenshots and live UI are visual evidence, not a published design contract. Because the three product-specific pages could not be fetched here, the recommendations below are implementation guidance, not claims that these vendors use identical tokens.

## Cross-product recommendation for this app

Use a consistent, neutral **outline** icon family rather than copying a vendor mark. Recommended Lucide mappings:

| Navigation meaning | Recommended icon | Rationale / source |
|---|---|---|
| New conversation | [`square-pen`](https://lucide.dev/icons/square-pen) | A pen-in-square convention communicates creating/editing a conversation without imitating a brand logo. |
| Conversations/history | [`message-square`](https://lucide.dev/icons/message-square) or [`messages-square`](https://lucide.dev/icons/messages-square) | The speech-bubble metaphor is semantically explicit; choose one and use it consistently. |
| Knowledge/documents | [`library`](https://lucide.dev/icons/library) or [`file-text`](https://lucide.dev/icons/file-text) | Use `library` for the knowledge area and `file-text` for a document-specific destination. |
| Work/code mode | [`code-2`](https://lucide.dev/icons/code-2) | The code glyph is recognizable while remaining a generic utility icon. |
| Search | [`search`](https://lucide.dev/icons/search) | Conventional magnifier; avoid using a logo-shaped mark for search. |
| Scheduled tasks | [`calendar-clock`](https://lucide.dev/icons/calendar-clock) | Communicates both schedule and time; use `calendar` if the label is already explicit. |
| Settings | [`settings-2`](https://lucide.dev/icons/settings-2) | Familiar controls affordance with the same outline language. |
| Collapse sidebar | [`panel-left-close`](https://lucide.dev/icons/panel-left-close) / [`panel-left-open`](https://lucide.dev/icons/panel-left-open) | State-specific icons make the action discoverable and avoid ambiguous chevrons. |

The [Lucide icon catalog](https://lucide.dev/icons/) is the source for the icon names and SVG assets above. Lucide describes itself as an open-source vector icon library in its [official repository README](https://github.com/lucide-icons/lucide#readme); use its license and package terms when adding it as a dependency or copying individual SVGs.

## Size, stroke, and spacing guidance

These are **recommended implementation tokens**, not claims about Claude, ChatGPT, or Codex internals:

- Render sidebar glyphs at **20 × 20 CSS px** inside a **32 × 32 px** hit area; use **24 × 24** for prominent toolbar controls. This preserves a compact desktop rail while meeting a comfortable pointer target.
- Keep the icon family at a single nominal size within one navigation level; do not mix 16/18/20 px glyphs merely to compensate for different artwork bounds.
- Start with **`stroke-width="2"`**, `stroke-linecap="round"`, and `stroke-linejoin="round"` for Lucide SVGs. Individual Lucide SVG source files expose the common `viewBox="0 0 24 24"` and stroke-based structure; verify the exact current asset before vendoring (example: [`message-square.svg`](https://raw.githubusercontent.com/lucide-icons/lucide/main/icons/message-square.svg)).
- Use **12 px icon-to-label gap** in a labelled row, **8 px horizontal inset** from the row edge, and **4–8 px vertical gap** between adjacent rows. Treat these as local tokens and adjust after testing the longest label.
- Give each row a minimum **32 px visual height** and a larger **40–44 px interactive height** when the row is a primary action; preserve visible focus and selected states independently of the icon.
- Keep icon color secondary to text (for example, muted foreground at rest and the same high-contrast foreground as the label when selected). Do not encode mode meaning by color alone.

For implementation, prefer the official Lucide package/API over hand-redrawing: [Lucide packages](https://lucide.dev/guide/packages/lucide) and [Lucide GitHub repository](https://github.com/lucide-icons/lucide) are the primary references. If the project does not want a dependency, copy only the needed SVGs with their license notice and retain accessible labels.

## Accessibility and product-fit checks

- Every icon-only control needs an accessible name (`aria-label` or an equivalent visible tooltip); labelled sidebar rows should expose the text label as the accessible name.
- Selected navigation must have a non-color cue (background, weight, indicator, or `aria-current="page"`).
- Keep the icon visually subordinate to the label: the icon identifies the destination; the label carries the meaning.
- Re-check the final result against the app's existing light/dark themes and compact/mobile breakpoints. The unavailable vendor pages mean no claim can be made here about their dark-mode or responsive icon tokens.

## Sources consulted

1. [Anthropic — Claude Code desktop documentation](https://code.claude.com/docs/en/desktop) (official page; inaccessible here, HTTP 403).
2. [Anthropic — Claude Code GitHub repository](https://github.com/anthropics/claude-code) (official repository; no verified desktop icon spec located).
3. [OpenAI Help — ChatGPT Windows app](https://help.openai.com/en/articles/9982051-using-the-chatgpt-windows-app) (official page; inaccessible here, HTTP 403).
4. [OpenAI — Codex app documentation](https://developers.openai.com/codex/app/) (official page; inaccessible here, HTTP 403).
5. [OpenAI — Codex GitHub repository](https://github.com/openai/codex) (official repository; no verified desktop icon spec located).
6. [Lucide icon catalog](https://lucide.dev/icons/) (official icon names/assets).
7. [Lucide package guide](https://lucide.dev/guide/packages/lucide) and [Lucide repository README](https://github.com/lucide-icons/lucide#readme) (official implementation/library references).
8. [Lucide `message-square` SVG](https://raw.githubusercontent.com/lucide-icons/lucide/main/icons/message-square.svg) (official source asset demonstrating the 24-unit stroke SVG structure).
