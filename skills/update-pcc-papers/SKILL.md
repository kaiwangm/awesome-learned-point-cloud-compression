---
name: update-pcc-papers
description: >
  Fetch the latest learned point cloud compression (PCC) papers from arXiv, IEEE,
  CVF Open Access and other sources, deduplicate them against the existing README,
  and append new entries to the correct year section. Use when the user asks to
  "update papers", "pull latest literature", "find new PCC papers", "refresh the
  awesome list", or any time the awesome-learned-point-cloud-compression repo
  needs its literature list brought up to date.
---

# Update Learned Point Cloud Compression Papers

This skill keeps the `awesome-learned-point-cloud-compression` README up to date
with the newest learned / neural point cloud compression literature.

## Target file

`README.md` at the repository root.

## Entry format (MUST follow exactly)

Each paper is a single bullet line under a `### <year>` heading inside the
`## Papers` section:

```markdown
- [[Venue](link)] Paper Title. [[Code](repo-url)]
```

Rules:
- `Venue` = short venue tag (see [reference.md](reference.md) for the full list).
- If a preprint has no venue yet, use `[[arxiv](arxiv-url)]`.
- `[[Code](repo-url)]` is optional — only include when an official / author
  GitHub repo exists.
- One blank line between consecutive entries.
- Entries inside a year are ordered newest-first (most recent publication date on top).

## Workflow

### Step 1 — Read the current state

1. Read `README.md`.
2. Collect every existing paper title (lower-cased) into a set for duplicate
   checking.

### Step 2 — Search for new papers

Run **parallel** web searches using the query templates below. Use the current
year and the previous year. Add more years if the user requests a wider window.

Search query templates (replace `{YEAR}`):

| # | Query |
|---|-------|
| 1 | `learned point cloud compression {YEAR}` |
| 2 | `neural point cloud compression {YEAR}` |
| 3 | `point cloud geometry compression deep learning {YEAR}` |
| 4 | `point cloud attribute compression deep learning {YEAR}` |
| 5 | `LiDAR point cloud compression neural {YEAR}` |
| 6 | `3D point cloud compression transformer {YEAR}` |
| 7 | `point cloud entropy model compression {YEAR}` |

Additionally, fetch the arXiv listing pages for `cs.CV` / `eess.IV` filtered by
the keyword `point cloud compression`:
`https://arxiv.org/list/cs.CV/{YEAR}{MM}` for recent months (only the last 6
months are needed for an update).

### Step 3 — Filter and deduplicate

For each candidate paper found:

1. **Relevance check** — the paper must be primarily about *learned / neural /
   deep-learning-based* point cloud compression (geometry, attribute, or joint).
    - Exclude pure MPEG G-PCC / V-PCC traditional codec papers unless they
      contain a learned component.
    - Exclude point cloud *quality assessment* or *enhancement* papers that do
      not propose a compression method (unless they are already represented in
      the repo — keep parity with existing style).
2. **Duplicate check** — compare the lower-cased title against the existing set
   (Step 1). Skip if already present. Use fuzzy match: if >85 % of significant
   words overlap, treat as duplicate.
3. **Venue resolution** — try to identify the official venue:
    - Check CVF Open Access, IEEE Xplore, ACM DL, openreview, the paper PDF.
    - If only an arXiv version exists, tag as `arxiv`.
4. **Code link** — search for an author GitHub repo (check the paper PDF
   "Code / Project page" footnote, and search GitHub by paper title). Only add
   `[[Code](...)]` when a repo is found.

### Step 4 — Format entries

Build each new entry string following the format in the "Entry format" section
above.

### Step 5 — Insert into README

1. Determine the publication year of each new paper.
2. Find or create the corresponding `### {year}` subsection under `## Papers`.
    - If the year section does not exist yet, create it **immediately above**
      the next-most-recent year section (years are sorted descending).
3. Insert new entries at the **top** of that year section (newest first).
4. Preserve all existing entries and formatting.

### Step 6 — Summarise for the user

After editing, print a short summary table:

```
| Year | Venue | Title | Code |
|------|-------|-------|------|
```

Also list any candidates that were skipped and the reason (duplicate / not
learned-based / venue not found).

## Key venue short-tags

Common venues already used in this repo:

`CVPR, ICCV, ECCV, NeurIPS/NIPS, AAAI, ICML, IJCAI, ACM MM, ACM TOMM,
TPAMI, TIP, TMM, TCSVT, TVCG, TII, RA-L, ICASSP, ICME, ICIP, DCC, VCIP,
PCS, MMVE, MM Asia, ICRA, IROS, ICMR, APCCPA, CMM, CAS, EG, IET, arxiv`

For the full venue list and search-URL cheat-sheet, see
[reference.md](reference.md).

## Important rules

- **Never invent links.** Every URL must be verified by actually fetching the
  page or seeing it in search results.
- **Never fabricate paper titles.** Only add papers that were found in search
  results or arXiv listings.
- **Keep alphabetical / chronological parity** with the existing style.
- When unsure whether a paper qualifies as "learned", default to excluding it
  and mention it in the skipped summary so the user can decide.
- After editing, do **not** run `git commit` unless the user explicitly asks.
