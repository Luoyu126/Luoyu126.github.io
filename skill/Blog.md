---
name: blog-md-to-polished-post
description: Turn raw Markdown blog drafts into this repo's polished al-folio style with responsive images, consistent heading hierarchy, left TOC behavior, and cleaned metadata. Use when the user provides a new blog markdown, asks to beautify post layout, or wants the same visual style applied to other posts.
---

# Blog Markdown To Polished Post

## Goal

Convert a user-provided Markdown draft into a publish-ready post that matches this repository's existing blog style and interaction pattern.

## Target Files

- Post content: `_posts/*.md`
- Post list behavior: `_pages/blog.md`
- Post layout: `_layouts/post.liquid`
- TOC/layout shell: `_layouts/default.liquid`
- Main style rules: `_sass/_blog.scss`, `_sass/_utilities.scss`
- Post interactions (TOC toggle): `assets/js/common.js`

## Workflow

1. Place post in `_posts/` with `YYYY-MM-DD-slug.md`.
2. Normalize frontmatter.
3. Normalize heading hierarchy.
4. Normalize images and paths.
5. Apply shared blog style rules (do not inline CSS in markdown).
6. Validate rendering consistency and metadata.

## Frontmatter Baseline

Use this as the default unless user asks otherwise:

```yaml
---
layout: post
title: "..."
date: YYYY-MM-DD HH:MM:SS +0800
description: "..."
tags: [tag1, tag2]
giscus_comments: true
stars: 0
---
```

Notes:
- Prefer tags; avoid categories unless explicitly requested.
- Keep `giscus_comments: true` for per-post reactions/comments.

## Heading Hierarchy Rules

- Major sections: `#`
- Numbered section blocks at the same depth must use the same heading level.
- Do not mix `### 1.1` with `## 2.1` in one outline level.
- Use `##` for chapter subsections like `1.1`, `1.2`, `2.1`.
- Use `###` only for deeper children under a `##` section.

## Image Rules

- Move post images to `assets/img/blog/<post-slug>/`.
- Use clean lowercase kebab-case filenames.
- Replace non-standard links with standard markdown:
  - `![alt](/assets/img/blog/<post-slug>/image-name.jpg)`
- Do not hardcode width/height in markdown unless user asks.
- Let shared CSS handle responsive sizing and shadows.

## Style Contract (Match Existing Visual Design)

When applying/reusing style, keep these behaviors:

- `#markdown-content` rendered as a card-like reading container.
- First paragraph emphasis with larger first line and drop cap.
- `h2` with theme-colored left border.
- `h3` with dashed bottom separator.
- Responsive images: centered, max width constrained, rounded corners, shadow.
- Regular blockquotes (`>`) visually lighter and smaller than body-heavy callouts.
- Code blocks and tables use bordered, scroll-safe presentation.
- Left sidebar TOC with hide/show toggle on desktop, hidden on small screens.

## TOC Rules

- Posts should render with left sidebar TOC via defaults (no per-post duplication unless needed).
- TOC toggle must both:
  - collapse/expand TOC content
  - adjust layout width so main content expands when TOC is hidden

## Blog List Rules

- Keep blog list in English controls (`Sort by`, `Time`, `Hot (Stars)`).
- Keep read-time text removed unless user explicitly asks for it back.
- Hot sort uses frontmatter `stars`.

## Validation Checklist

- [ ] No broken image paths.
- [ ] Heading levels are consistent and TOC nesting looks correct.
- [ ] No oversized images on desktop/mobile.
- [ ] Blockquotes are not oversized.
- [ ] Post still supports comments/reactions and repo star CTA.
- [ ] Lints clean for edited files.

## Output Style For User Updates

When reporting completion:
- Briefly list what changed.
- Call out any content normalization (headings, image filenames, metadata).
- Mention whether push/deploy was done or still pending.
