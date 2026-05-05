# Gemini Handoff: Blog Like Performance + Star Behavior

## 1) Context

- Stack: Jekyll + al-folio (GitHub Pages)
- Main blog list page: `_pages/blog.md`
- Single post page: `_layouts/post.liquid`
- Interaction styles: `_sass/_blog.scss`
- Current post: `_posts/2026-04-29-gil-cpu-parallelism.md`
- Repo: `https://github.com/Luoyu126/Luoyu126.github.io`

## 2) Current Problems (User-visible)

1. On `/blog`, like counts are not instant; they appear after some delay.
2. User wants repo star to affect the real GitHub repo star count.
3. Previous requirement added undo behavior for both Like and Star.

## 3) Current Implementation (Important)

### A) Blog card like count loading (list page)

File: `_pages/blog.md`

- Each post card has:
  - `data-like-key` computed from `post.id` (fallback `post.url`)
  - visible count in `.post-like-count`
- On page load:
  - JS runs `syncLikeCounts()`
  - loops all cards
  - `fetch("https://api.countapi.xyz/get/...")` per card
  - updates count text and `data-stars` used for hot sorting
  - then triggers sorting

Potential bottleneck:
- No request timeout / abort.
- One network call per post card.
- If `countapi` is slow from user network region, UI count refresh is delayed.

### B) Like button (single post)

File: `_layouts/post.liquid`

- Like uses `countapi` namespace: `chenyy-homepage-post-likes`
- Toggle behavior implemented:
  - click -> `+1`
  - click again -> `-1`
- Uses localStorage for:
  - clicked state flag
  - local counter fallback
- Remote API:
  - read: `GET /get/{namespace}/{key}`
  - update: `GET /update/{namespace}/{key}?amount={delta}`

### C) Star Repo button (single post)

File: `_layouts/post.liquid`

- Current state is **local counter simulation**, not real GitHub star.
- Uses `countapi` namespace: `chenyy-homepage-repo-stars`
- Same toggle logic as Like (+1 / -1 with undo).

Reason this does not change actual GitHub stars:
- No authenticated GitHub API call is made.
- No OAuth/session backend exists.
- GitHub real star/unstar requires authenticated user context.

### D) Comment

File: `_layouts/post.liquid`

- Local in-page comment form.
- Comments saved in localStorage key `comments:{like_key}`.
- Not shared across users/devices (no backend DB).

## 4) Key Code Locations

- Blog list count fetch/sort: `_pages/blog.md` (script near bottom)
- Like/Star toggle + comment logic: `_layouts/post.liquid` (script in post footer area)
- Button/comment visual styles: `_sass/_blog.scss`

## 5) What We Need Gemini To Solve

Please propose concrete code-level solutions for:

1. **Fast like count rendering on `/blog`**
   - Make counts feel instant (prefer <200ms perceived delay).
   - Keep hot sorting correct.
   - Handle slow/unreachable `countapi` gracefully.
   - Prefer minimal architecture change first.

2. **Real GitHub Star/Unstar support**
   - User requirement: click in page should affect actual repo star.
   - Also wants undo (unstar).
   - Need practical architecture for static-site context:
     - Is it possible without redirect?
     - If not, what is minimal secure OAuth/backend approach?
     - Alternative UX if full in-page auth is not feasible.

3. **If needed: migration plan**
   - Keep current local fallback behavior while introducing a real backend path.
   - Avoid breaking current hot sort and UI.

## 6) Constraints

- Site is static-first (GitHub Pages deployment).
- Prefer small incremental changes.
- Keep current visual style and existing file structure.
- Do not remove current blog content/custom styling.

## 7) Nice-to-have from Gemini

- Provide 2-3 implementation options ranked by effort vs reliability.
- For chosen option, include exact patch guidance for:
  - `_pages/blog.md`
  - `_layouts/post.liquid`
  - optional new backend/API files (if required)
