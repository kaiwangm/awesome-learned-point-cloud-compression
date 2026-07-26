# Reference: Venues, Search URLs, and Formatting Details

This file provides supplementary detail to [SKILL.md](SKILL.md). Read it when
you need venue short-tag mappings, direct search URLs, or formatting examples.

---

## 1. Venue short-tag map

Use the short tag in the `[[Venue](url)]` portion of each entry. When a paper
appears in multiple venues (e.g. arXiv preprint later accepted to CVPR), use
the **final published venue** and link the official version.

### Conferences

| Full name                                | Short tag       |
|------------------------------------------|-----------------|
| IEEE/CVF Conference on Computer Vision and Pattern Recognition | CVPR |
| International Conference on Computer Vision | ICCV |
| European Conference on Computer Vision    | ECCV            |
| Conference on Neural Information Processing Systems | NeurIPS (older: NIPS) |
| AAAI Conference on Artificial Intelligence | AAAI            |
| International Conference on Machine Learning | ICML            |
| International Joint Conference on Artificial Intelligence | IJCAI |
| ACM Multimedia                           | ACM MM          |
| ACM Multimedia Systems                   | MMVE            |
| ACM Multimedia Asia                      | MM Asia         |
| IEEE International Conference on Acoustics, Speech and Signal Processing | ICASSP |
| IEEE International Conference on Multimedia & Expo | ICME |
| IEEE International Conference on Image Processing | ICIP |
| Data Compression Conference              | DCC             |
| Visual Communications and Image Processing | VCIP            |
| Picture Coding Symposium                  | PCS             |
| IEEE International Conference on Robotics and Automation | ICRA |
| IEEE/RSJ International Conference on Intelligent Robots and Systems | IROS |
| ACM International Conference on Multimedia Retrieval | ICMR |
| ACM Workshop on Point Cloud Processing, Applications, and Acquisition | APCCPA |
| Computer Graphics International           | CMM             |
| Circuits and Systems Conference           | CAS             |
| Eurographics                              | EG              |

### Journals

| Full name                                                | Short tag |
|----------------------------------------------------------|-----------|
| IEEE Transactions on Pattern Analysis and Machine Intelligence | TPAMI |
| IEEE Transactions on Image Processing                    | TIP       |
| IEEE Transactions on Multimedia                          | TMM       |
| IEEE Transactions on Circuits and Systems for Video Technology | TCSVT |
| IEEE Transactions on Visualization and Computer Graphics | TVCG      |
| IEEE Transactions on Industrial Informatics             | TII       |
| IEEE Robotics and Automation Letters                     | RA-L      |
| IEEE/CAA Journal of Automatica Sinica                    | JAS       |
| ACM Transactions on Multimedia Computing, Communications, and Applications | ACM TOMM |
| Computational Visual Media                               | CVM       |
| IET Electronics Letters                                  | IET       |

### Preprint

| Source | Short tag |
|--------|-----------|
| arXiv  | arxiv     |

---

## 2. Search-URL cheat-sheet

### arXiv

- Listing by month: `https://arxiv.org/list/cs.CV/{YEAR}{MM}` (MM = 01–12).
  Scan titles for "point cloud compression", "PCC", "LiDAR compression".
- Search API: `https://arxiv.org/search/?query=point+cloud+compression&start=0`
- Tip: filter results by checking the "Subjects" line for `cs.CV` or `eess.IV`.

### CVF Open Access

- CVPR: `https://openaccess.thecvf.com/CVPR{YEAR}?view=all`
- ICCV: `https://openaccess.thecvf.com/ICCV{YEAR}?view=all`
- ECCV: `https://openaccess.thecvf.com/ECCV{YEAR}?view=all`
- Winter Conference on Applications of Computer Vision (WACV):
  `https://openaccess.thecvf.com/WACV{YEAR}?view=all`

Search within a CVF page with Ctrl-F for "point cloud compression".

### IEEE Xplore

- Advanced search URL pattern:
  `https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=point%20cloud%20compression&highlight=true&returnType=SEARCH&matchPubs=true&sortType=newest`
- Filter by Year and Publication Type.

### ACM Digital Library

- Search: `https://dl.acm.org/doi/doSearch?AllField=point+cloud+compression`
- Filter by year and publication venue.

### openreview.net

- Used for ICLR / NeurIPS / ICML submissions.
- Search: `https://openreview.net/search?term=point+cloud+compression`

### Google Scholar

- `https://scholar.google.com/scholar?q=learned+point+cloud+compression&as_ylo={YEAR}`
- Useful for catching papers across all venues.

### GitHub code search

- `https://github.com/search?q=point+cloud+compression&type=repositories`
- Useful for finding the official code repo of a paper.

---

## 3. Formatting examples

### Accepted conference paper with code

```markdown
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/You_RENO_Real-Time_Neural_Compression_for_3D_LiDAR_Point_Clouds_CVPR_2025_paper.pdf)] RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds. [[Code](https://github.com/NJUVISION/RENO)]
```

### Journal paper without code

```markdown
- [[TCSVT](https://ieeexplore.ieee.org/abstract/document/10938715)] GAEM: Graph-Driven Attention-Based Entropy Model for LiDAR Point Cloud Compression.
```

### arXiv preprint with code

```markdown
- [[arxiv](https://arxiv.org/abs/2404.06936)] Efficient and Generic Point Model for Lossless Point Cloud Attribute Compression. [[Code](https://github.com/I2-Multimedia-Lab/PoLoPCAC)]
```

### arXiv preprint without code

```markdown
- [[arxiv](https://arxiv.org/abs/2404.07698)] Point Cloud Geometry Scalable Coding with a Quality-Conditioned Latents Probability Estimator.
```

---

## 4. Duplicate-detection tips

- Lower-case both the candidate title and every existing title.
- Tokenise on whitespace, drop stopwords (the, a, an, of, for, with, via,
  using, to, in, on, and).
- Compute word-overlap ratio = |intersection| / |union| of significant tokens.
- If ratio > 0.85 → treat as duplicate.
- Special case: when the same title appears with a subtitle difference
  (e.g. "Part I: Geometry" vs "Part II: Attribute"), compare the full title
  string including the subtitle.

---

## 5. Quality checklist before finishing

- [ ] Every new entry URL was fetched/seen in a search result — none invented.
- [ ] Every venue short-tag matches the table in §1.
- [ ] `[[Code](url)]` is present **only** when a repo was actually found.
- [ ] New entries are placed under the correct `### {year}` heading.
- [ ] Year sections remain sorted descending (newest year on top).
- [ ] No existing entries were modified or removed.
- [ ] A summary table was printed for the user.
