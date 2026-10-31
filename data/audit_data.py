#!/usr/bin/env python3
"""
audit_data.py - sanity-check the frozen Cardiology-Chat data artifacts (D1-D6).

Usage (Windows):
    python audit_data.py --root "D:\\Coding\\Projects\\Academic\\Cardiology Chat\\data"

Writes audit_report.txt into the root folder. Field names are guessed from
common alternatives, so if a section prints "could not find field ...",
paste the report back and the script can be adjusted. Needs only the stdlib.
"""
import argparse, collections, difflib, hashlib, json, math, random, re, statistics as st, traceback
from pathlib import Path

REPORT = []


def say(level, msg):
    line = f"[{level}] {msg}"
    print(line)
    REPORT.append(line)


def head(t):
    line = f"\n=== {t} ==="
    print(line)
    REPORT.append(line)


# ---------------------------------------------------------------- helpers
def load(p):
    if not p.exists():
        say("FAIL", f"missing file: {p}")
        return None, 0
    rows, bad = [], 0
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                bad += 1
    return rows, bad


def norm(t):
    return re.sub(r"\W+", " ", str(t).lower()).strip()


def words(t):
    return norm(t).split()


STOP = set("with without associated acute chronic syndrome disease due from type left right and "
           "the of in on by for to a an or secondary primary severe mild".split())


def content_words(t):
    return [w for w in words(t) if len(w) >= 4 and w not in STOP]


def flat(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return " ".join(flat(i) for i in x)
    if isinstance(x, dict):
        return " ".join(flat(v) for v in x.values())
    return str(x)


def pick(rec, names, default=None):
    low = {k.lower(): k for k in rec}
    for n in names:
        k = low.get(n.lower())
        if k is not None and rec[k] not in (None, ""):
            return rec[k]
    return default


def get_stages(rec):
    s = pick(rec, ["stages", "steps", "reveal_stages", "trajectory"])
    if isinstance(s, list):
        return [flat(i) for i in s]
    if isinstance(s, dict):
        return [flat(v) for v in s.values()]
    ks = [k for k in rec if re.search(r"stage|step_?\d|presentation|diagnostic|intervention|outcome", k, re.I)]

    def order(k):
        m = re.search(r"\d+", k)
        return int(m.group()) if m else 99

    ks.sort(key=order)
    return [flat(rec[k]) for k in ks]


GOLD_KEYS = ["gold_primary", "gold_diagnosis", "final_diagnosis", "diagnosis", "gold", "label"]
STRICT = re.compile(
    r"\b(heart|cardi\w*|myocard\w*|coronary|atrial|atrium|ventric\w*|ecg|ekg|aort\w*|mitral|tricuspid|"
    r"valv\w*|angina|arrhythm\w*|troponin|infarct\w*|murmur|pericard\w*|endocard\w*|tachycard\w*|"
    r"bradycard\w*|stemi|qrs|fibrillation|ductus|tetralogy|rheumatic|hypertension|bnp|echocardiogra\w*)\b", re.I)
OCR = re.compile(r"\b(aoic|hea|mial|shormess|pressue|sta)\b", re.I)
VET = re.compile(r"\b(dogs?|canine|felines?|cats?|equine|horses?|bovine|cows?|porcine|pigs?|murine|"
                 r"veterinar\w*)\b", re.I)


def leaks(gold, text):
    cw = content_words(gold)
    if not cw:
        return False
    tw = set(words(text))
    return all(w in tw for w in cw)


def ci(n, p=0.6):
    return 1.96 * math.sqrt(p * (1 - p) / n) * 100 if n else float("nan")


def grams(t, n=13):
    w = words(t)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def near_dups(queries, refs, thr=0.8, df_cap=100):
    """queries/refs: lists of (id, text). Returns [(qid, rid, jaccard)]."""
    rt = [(i, frozenset(words(t))) for i, t in refs]
    df = collections.Counter(w for _, s in rt for w in s)
    index = collections.defaultdict(list)
    for j, (_, s) in enumerate(rt):
        for w in s:
            if df[w] <= df_cap:
                index[w].append(j)
    out = []
    for qid, t in queries:
        qs = frozenset(words(t))
        cnt = collections.Counter(j for w in qs for j in index.get(w, ()))
        for j, _ in cnt.most_common(15):
            rs = rt[j][1]
            jac = len(qs & rs) / max(1, len(qs | rs))
            if jac >= thr:
                out.append((qid, rt[j][0], round(jac, 2)))
                break
    return out


# ---------------------------------------------------------------- sections
def sec_hashes(root):
    head("Manifests vs files")
    hashes = set()
    for m in sorted((root / "manifests").glob("*.txt")):
        hs = {h.lower() for h in re.findall(r"\b[0-9a-fA-F]{64}\b", m.read_text(encoding="utf-8", errors="ignore"))}
        hashes |= hs
        say("INFO", f"{m.name}: {len(hs)} hash(es) listed")
    files = sorted((root / "artifacts").glob("*.jsonl")) + \
        sorted((root / "knowledge base" / "indexed chunks").glob("*.jsonl"))
    for f in files:
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        n = sum(1 for l in f.open(encoding="utf-8") if l.strip())
        ok = h in hashes
        say("OK" if ok else "FAIL",
            f"{f.name}: {n} rows, " + ("hash matches a manifest" if ok else "hash NOT in any manifest (edited after freezing?)"))


def sec_d2(root):
    head("D2 MCQ splits")
    A = root / "artifacts"
    S = {}
    for n in ["train", "val", "test"]:
        rows, bad = load(A / f"D2_{n}_clean.jsonl")
        S[n] = rows or []
        if bad:
            say("FAIL", f"D2 {n}: {bad} unparsable lines")
    for n, rows in S.items():
        if rows:
            say("INFO", f"{n} keys: {sorted(rows[0].keys())}")
        src = collections.Counter(pick(r, ["source"], "?") for r in rows)
        say("INFO", f"{n}: {len(rows)} rows, sources {dict(src)}, answer letters "
                    f"{dict(sorted(collections.Counter(r.get('answer') for r in rows).items()))}")
        bad_o = sum(1 for r in rows if not isinstance(r.get("options"), dict)
                    or len(r["options"]) != 4 or r.get("answer") not in r["options"])
        say("OK" if bad_o == 0 else "FAIL", f"{n}: {bad_o} rows with malformed options/answer")
        ids = [r.get("id") for r in rows]
        say("OK" if len(ids) == len(set(ids)) else "FAIL", f"{n}: {len(ids) - len(set(ids))} duplicate ids")
        c = collections.Counter(norm(r["question"]) for r in rows)
        d = sum(v - 1 for v in c.values() if v > 1)
        say("OK" if d == 0 else "WARN", f"{n}: {d} exact duplicate questions inside split")
        for s in sorted(src):
            sub = [r for r in rows if r.get("source") == s]
            short = sum(len(r["question"].split()) <= 15 for r in sub)
            card = sum(bool(STRICT.search(r["question"])) for r in sub)
            ocr = sum(bool(OCR.search(r["question"] + " " + " ".join(r["options"].values()))) for r in sub)
            say("INFO", f"{n}/{s}: {short}/{len(sub)} short factoid (<=15 words); {card}/{len(sub)} with a cardiology "
                        f"term in stem; {ocr} OCR-damage markers")
    # cross-split leakage
    for a, b in [("test", "train"), ("test", "val"), ("val", "train")]:
        qa = {norm(r["question"]) for r in S[a]}
        qb = {norm(r["question"]) for r in S[b]}
        say("OK" if not (qa & qb) else "FAIL", f"exact question overlap {a}<->{b}: {len(qa & qb)}")
    ref_trval = [(f"train:{r['id']}", r["question"]) for r in S["train"]] + \
                [(f"val:{r['id']}", r["question"]) for r in S["val"]]
    nd = near_dups([(f"test:{r['id']}", r["question"]) for r in S["test"]], ref_trval)
    say("OK" if not nd else "FAIL", f"NEAR-duplicate test items (word-Jaccard>=0.8) vs train/val: {len(nd)}")
    for x in nd[:5]:
        say("INFO", f"   example near-dup: {x}")
    nd2 = near_dups([(f"val:{r['id']}", r["question"]) for r in S["val"]],
                    [(f"train:{r['id']}", r["question"]) for r in S["train"]])
    say("OK" if not nd2 else "WARN", f"near-duplicate val items vs train: {len(nd2)}")
    # label conflicts
    grp = collections.defaultdict(set)
    for n, rows in S.items():
        for r in rows:
            grp[norm(r["question"])].add(norm(r["options"][r["answer"]]))
    conf = sum(1 for v in grp.values() if len(v) > 1)
    say("OK" if conf == 0 else "WARN", f"same question with conflicting answer text across dataset: {conf} groups")
    say("INFO", f"95% CI half-width at n={len(S['test'])}, acc~0.6: +/-{ci(len(S['test'])):.1f} pp")
    return S


def sec_d3(root, d2):
    head("D3 perturbations")
    A = root / "artifacts"
    T = {r["id"]: r for r in d2.get("test", [])}
    suffix = re.compile(r"(__|_|-|#|:)(p\d.*|perm.*|noise.*|jitter.*|strip.*)$", re.I)
    rnd = random.Random(0)
    HR = re.compile(r"(?:pulse|heart rate|HR)[^.\d]{0,25}(\d{2,3})\s*/\s*min", re.I)
    BP = re.compile(r"(\d{2,3})\s*/\s*(\d{2,3})\s*mm\s*Hg", re.I)
    num = re.compile(r"\d+(?:\.\d+)?")

    def cls(x, cuts):
        return sum(x >= c for c in cuts)

    for f in sorted(A.glob("D3_*.jsonl")):
        rows, bad = load(f)
        if rows is None:
            continue
        say("INFO", f"{f.name}: {len(rows)} rows, keys {sorted(rows[0].keys()) if rows else []}")
        pairs = []
        for r in rows:
            b = None
            for k in ["base_id", "orig_id", "original_id", "source_id", "parent_id", "id"]:
                v = r.get(k)
                if v in T:
                    b = v
                    break
                if isinstance(v, str) and suffix.sub("", v) in T:
                    b = suffix.sub("", v)
                    break
            pairs.append((r, T.get(b)))
        linked = sum(1 for _, b in pairs if b)
        say("OK" if linked == len(rows) else "FAIL", f"{f.name}: {linked}/{len(rows)} rows link to a D2 test item")
        pairs = [(r, b) for r, b in pairs if b]
        name = f.name.lower()
        if "p1" in name or "perm" in name:
            same = badkey = badopt = 0
            for r, b in pairs:
                ro, bo = r.get("options"), b["options"]
                if not isinstance(ro, dict) or r.get("answer") not in ro:
                    badkey += 1
                    continue
                if sorted(norm(v) for v in ro.values()) != sorted(norm(v) for v in bo.values()):
                    badopt += 1
                if norm(ro[r["answer"]]) != norm(bo[b["answer"]]):
                    badkey += 1
                if ro == bo:
                    same += 1
            say("OK" if badkey == 0 else "FAIL", f"P1: {badkey} rows where remapped key points at a different option text")
            say("OK" if badopt == 0 else "FAIL", f"P1: {badopt} rows where the option set changed")
            say("OK" if same < 0.1 * max(1, len(pairs)) else "WARN",
                f"P1: {same} rows identical to original (no shuffle). Position of correct answer: "
                f"{dict(sorted(collections.Counter(r.get('answer') for r, _ in pairs).items()))}")
            continue
        qf = ["question", "perturbed_question", "text", "vignette"]
        grow, retain, noop, nonnum, cross, comparable, nchg = [], [], 0, 0, 0, 0, []
        removed_examples = []
        for r, b in pairs:
            pt, bt = flat(pick(r, qf, "")), b["question"]
            if norm(pt) == norm(bt):
                noop += 1
            grow.append(len(pt.split()) - len(bt.split()))
            bw = collections.Counter(words(bt))
            pw = collections.Counter(words(pt))
            retain.append(sum((bw & pw).values()) / max(1, sum(bw.values())))
            if "p4" in name or "jitter" in name:
                if num.sub("#", norm(pt)) != num.sub("#", norm(bt)):
                    nonnum += 1
                h1, h2, p1, p2 = HR.search(bt), HR.search(pt), BP.search(bt), BP.search(pt)
                if h1 and h2 and p1 and p2:
                    comparable += 1
                    if cls(int(h1.group(1)), (60, 101)) != cls(int(h2.group(1)), (60, 101)) or \
                       cls(int(p1.group(1)), (90, 140, 180)) != cls(int(p2.group(1)), (90, 140, 180)) or \
                       cls(int(p1.group(2)), (60, 90, 120)) != cls(int(p2.group(2)), (60, 90, 120)):
                        cross += 1
                nchg.append(sum(a != c for a, c in zip(num.findall(bt), num.findall(pt))))
            if "p5" in name or "strip" in name:
                sm = difflib.SequenceMatcher(None, bt.split(), pt.split())
                rem = " ".join(" ".join(bt.split()[i1:i2]) for tag, i1, i2, _, _ in sm.get_opcodes()
                               if tag in ("delete", "replace"))
                removed_examples.append(rem)
        say("OK" if noop == 0 else "WARN", f"{f.name}: {noop} rows where perturbation changed nothing")
        say("INFO", f"{f.name}: mean words added/removed {st.mean(grow):+.1f}; mean original-word retention "
                    f"{st.mean(retain):.2f}; rows with retention<0.9: {sum(x < 0.9 for x in retain)}")
        if "p4" in name or "jitter" in name:
            say("OK" if nonnum == 0 else "WARN", f"P4: {nonnum} rows changed something other than numbers")
            say("OK" if cross == 0 else "FAIL",
                f"P4: {cross}/{comparable} rows where jitter crosses a clinical band (HR 60/100; SBP 90/140/180; DBP 60/90/120)")
            say("INFO", f"P4: mean numbers changed per row {st.mean(nchg):.1f} (jitter may touch labs/doses too)")
        if "p5" in name or "strip" in name:
            nonempty = [e for e in removed_examples if e.strip()]
            say("INFO", f"P5: {len(nonempty)}/{len(removed_examples)} rows had text removed; "
                        f"median removed words {st.median(len(e.split()) for e in nonempty) if nonempty else 0}")
            for e in rnd.sample(nonempty, min(5, len(nonempty))):
                say("INFO", f"   REVIEW removed text: {e[:160]}")
        if any(k in name for k in ("p3", "noise")):
            ex = rnd.sample(pairs, min(3, len(pairs)))
            for r, b in ex:
                added = [s for s in re.split(r"(?<=[.!?])\s+", flat(pick(r, qf, ""))) if norm(s) not in norm(b["question"])]
                say("INFO", f"   REVIEW injected text: {' '.join(added)[:160]}")


def title_overlap(title, text):
    tcw = set(content_words(title))
    if not tcw:
        return None
    return len(tcw & set(words(text))) / len(tcw)


def sec_d1(root):
    head("D1 cases")
    rows, bad = load(root / "artifacts" / "D1_cases_staged.jsonl")
    if not rows:
        return []
    say("INFO", f"{len(rows)} cases, keys {sorted(rows[0].keys())}")
    ids = collections.Counter(flat(pick(r, ["pmcid", "pmid", "case_id", "doi", "id"], f"row{i}")) for i, r in enumerate(rows))
    say("OK" if max(ids.values()) == 1 else "FAIL", f"duplicate case ids/DOIs: {sum(v - 1 for v in ids.values() if v > 1)}")
    years = collections.Counter()
    for r in rows:
        m = re.search(r"(19|20)\d{2}", flat(pick(r, ["pub_date", "publication_date", "date", "year"], "")))
        years[m.group(0) if m else "missing"] += 1
    say("INFO", f"publication year: {dict(sorted(years.items()))}")
    new = sum(v for k, v in years.items() if k.isdigit() and int(k) >= 2025)
    say("OK" if new >= 0.8 * len(rows) else "WARN", f"{new}/{len(rows)} published 2025 or later (contamination control)")
    lic = collections.Counter(flat(pick(r, ["license", "licence"], "missing")) for r in rows)
    say("WARN" if "missing" in lic else "INFO", f"licenses: {dict(lic)} (needed if you redistribute case text)")
    stg = [get_stages(r) for r in rows]
    nst = collections.Counter(len(s) for s in stg)
    say("INFO", f"stage counts per case: {dict(nst)}")
    gold = [flat(pick(r, GOLD_KEYS, "")) for r in rows]
    n_gold = sum(1 for g in gold if g)
    say("OK" if n_gold == len(rows) else "WARN",
        f"{n_gold}/{len(rows)} cases have a structured gold-diagnosis field "
        "(without one, accuracy/F1 needs an extra label-extraction step)")
    miss_f = [k for k in ("gold_secondary", "synonyms", "category", "split") if k not in rows[0]]
    say("INFO", f"optional fields not present in D1: {miss_f}")
    title = [flat(pick(r, ["title", "case_title"], "")) for r in rows]
    abstract = [flat(pick(r, ["ground_truth_abstract", "ground_truth_reference", "abstract"], "")) for r in rows]
    for i in range(max(nst) if nst else 0):
        txts = [s[i] if len(s) > i else "" for s in stg]
        lens = [len(t.split()) for t in txts]
        tov = [title_overlap(t, x) for t, x in zip(title, txts)]
        tov = [x for x in tov if x is not None]
        copy = sum(1 for a, x in zip(abstract, txts) if grams(a, 8) & grams(x, 8))
        say("INFO", f"stage {i + 1}: median {st.median(lens):.0f} words; <20 words in {sum(l < 20 for l in lens)}/{len(lens)} cases; "
                    f"title-word overlap (leak proxy) mean {st.mean(tov) if tov else float('nan'):.2f}; "
                    f"shares an 8-gram with the abstract in {copy}/{len(rows)}")
    for i in range(max(nst) if nst else 0):
        if n_gold:
            gl = [flat(pick(r, ["pmcid", "id", "case_id"], "?")) for r, t, g in zip(rows, stg, gold)
                  if g and len(t) > i and leaks(g, t[i])]
            say("INFO", f"   stage {i + 1}: gold-diagnosis words all present in {len(gl)}/{len(rows)} cases {gl[:12]}")
    say("INFO", "   leak proxy: stage 1 should be low; high overlap means the diagnosis is likely given away")
    rnd = random.Random(0)
    for i in rnd.sample(range(len(rows)), min(4, len(rows))):
        s = stg[i] + [""] * 3
        say("INFO", f"   REVIEW title: {title[i][:110]} | stage1: {s[0][:120]!r} | stage2: {s[1][:120]!r}")
    vet = sum(1 for s in stg if s and VET.search(s[0]))
    say("OK" if vet == 0 else "WARN", f"{vet} cases with animal/veterinary words in stage 1")
    nc = [i for i, s in enumerate(stg) if s and not STRICT.search(" ".join(s))]
    say("OK" if not nc else "WARN", f"{len(nc)} cases with no cardiology term anywhere (rows {nc[:10]})")
    say("INFO", f"95% CI half-width at n={len(rows)}, acc~0.6: +/-{ci(len(rows)):.1f} pp (paired tests are tighter)")
    return rows


def sec_d4(root, d1, d2):
    head("D4 knowledge base")
    out = {}
    for path in sorted((root / "knowledge base" / "indexed chunks").glob("*.jsonl")):
        kb, _ = load(path)
        if not kb:
            continue
        say("INFO", f"--- {path.name} ---")
        _audit_kb(path.name, kb, d1, d2)
        out[path.name] = kb
    return out


def _audit_kb(name, kb, d1, d2):
    say("INFO", f"{len(kb)} chunks, keys {sorted(kb[0].keys())}")
    texts = [flat(pick(r, ["text", "content", "chunk", "page_content"], "")) for r in kb]
    wl = [len(t.split()) for t in texts]
    say("INFO", f"chunk words: median {st.median(wl):.0f}, p10 {sorted(wl)[len(wl) // 10]}, p90 {sorted(wl)[9 * len(wl) // 10]}, max {max(wl)}")
    say("OK" if sum(w < 30 for w in wl) < 0.05 * len(wl) else "WARN", f"{sum(w < 30 for w in wl)} chunks under 30 words")
    say("OK" if len(set(map(norm, texts))) == len(texts) else "WARN", f"{len(texts) - len(set(map(norm, texts)))} duplicate chunk texts")
    is_hierarchical = "hierarchical" in name.lower()
    if not is_hierarchical:
        say("INFO", "naive fixed-window chunking arm (retained intentionally for naive vs. structured retrieval ablation)")
    else:
        if len(set(wl)) <= 3:
            say("WARN", "chunk lengths are (nearly) constant in hierarchical arm: expected variable length sections")
        if not any(re.search(r"head|section|title", k, re.I) for k in kb[0]):
            say("WARN", "no heading/section metadata found in hierarchical KB chunks")
    for key in ("source_doc", "source", "doc", "guideline", "file", "title"):
        if key in kb[0]:
            say("INFO", f"chunks per {key}: {dict(collections.Counter(flat(r.get(key)) for r in kb).most_common(10))}")
            break
    kg = set().union(*[grams(t) for t in texts])
    for name, items in [("D2 test", d2.get("test", [])), ("D2 val", d2.get("val", []))]:
        hit = sum(1 for r in items if grams(r["question"]) & kg)
        say("OK" if hit == 0 else "FAIL", f"{name}: {hit}/{len(items)} questions share a 13-gram with the KB "
                                         "(blind spot: questions under 13 words cannot match)")
    hit = sum(1 for r in d1 if grams(" ".join(get_stages(r))) & kg)
    say("OK" if hit == 0 else "FAIL", f"D1: {hit}/{len(d1)} cases share a 13-gram with the KB")
    idx = collections.defaultdict(set)
    for i, t in enumerate(texts):
        for w in set(words(t)):
            idx[w].add(i)
    cov = tot = 0
    for r in d1:
        cw = content_words(flat(pick(r, GOLD_KEYS, "")))
        if not cw:
            continue
        tot += 1
        sets = [idx.get(w, set()) for w in cw]
        if all(sets) and set.intersection(*sets):
            cov += 1
    say("INFO", f"crude coverage: {cov}/{tot} D1 gold diagnoses have all their key words inside a single KB chunk")


def sec_d5(root, kb, d2):
    head("D5 retrieval benchmark")
    rows, _ = load(root / "artifacts" / "D5_retrieval_benchmark.jsonl")
    if not rows:
        return
    say("INFO", f"{len(rows)} pairs, keys {sorted(rows[0].keys())}")
    say("INFO", f"Recall@k 95% CI half-width at n={len(rows)}: +/-{ci(len(rows), 0.5):.1f} pp; "
                f"covers {100 * len(rows) / max(1, len(d2.get('test', []))):.0f}% of D2 test size")
    if "domain" in rows[0]:
        say("INFO", f"queries per domain: {dict(collections.Counter(flat(r.get('domain')) for r in rows))}")
    for name, kbrows in (kb or {}).items():
        kbids = {flat(pick(r, ["chunk_id", "id"], i)) for i, r in enumerate(kbrows)}
        ck = [k for k in rows[0] if "chunk" in k.lower() or "gold" in k.lower() or "relevant" in k.lower()]
        for k in ck:
            vals = [v for r in rows for v in (r[k] if isinstance(r.get(k), list) else [r.get(k)])]
            miss = sum(1 for v in vals if str(v) not in kbids)
            say("OK" if miss == 0 else "INFO", f"field '{k}': {len(vals) - miss}/{len(vals)} ids found in {name}")
    q = collections.Counter(norm(flat(pick(r, ["query", "question"], ""))) for r in rows)
    say("OK" if max(q.values()) == 1 else "WARN", f"{sum(v - 1 for v in q.values() if v > 1)} duplicate queries")


def sec_d6(root, d1):
    head("D6 progressive reveal")
    rows, _ = load(root / "artifacts" / "D6_progressive_reveal.jsonl")
    if not rows:
        return
    say("INFO", f"{len(rows)} trajectories, keys {sorted(rows[0].keys())}")
    stg = [get_stages(r) for r in rows]
    if not any(stg):
        say("WARN", "could not find step fields; D6 checks skipped")
        return
    say("INFO", f"steps per trajectory: {dict(collections.Counter(len(s) for s in stg))}; median words per step: "
                f"{[int(st.median(len(s[i].split()) for s in stg if len(s) > i)) for i in range(max(len(s) for s in stg))]}")
    d1map = {flat(pick(r, ['doi'], '')): get_stages(r) for r in d1 if pick(r, ['doi'])}
    jac = []
    for r, s in zip(rows, stg):
        o = d1map.get(flat(pick(r, ['doi'], '')))
        if o:
            a, b = set(words(" ".join(s))), set(words(" ".join(o)))
            jac.append(len(a & b) / max(1, len(a | b)))
    if jac:
        say("INFO", f"linked to D1 by DOI: {len(jac)}/{len(rows)}; mean word-Jaccard of D6 steps vs D1 stages {st.mean(jac):.2f} "
                    "(near 1.0 means D6 is D1 re-labelled)")
    cum = sum(1 for s in stg if all(norm(a)[:60] in norm(b) for a, b in zip(s, s[1:])))
    say("INFO", f"{cum}/{len(rows)} trajectories are cumulative (each step contains the previous one)")
    titles = [flat(pick(r, ['case_title', 'title'], '')) for r in rows]
    for i in range(max(len(s) for s in stg)):
        tov = [title_overlap(t, s[i]) for t, s in zip(titles, stg) if len(s) > i]
        tov = [x for x in tov if x is not None]
        say("INFO", f"step {i + 1}: title-word overlap (leak proxy) mean {st.mean(tov) if tov else float('nan'):.2f}")


def safe(fn, *a):
    try:
        return fn(*a)
    except Exception:
        say("FAIL", f"{fn.__name__} crashed: {traceback.format_exc().splitlines()[-1]}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    root = Path(ap.parse_args().root)
    safe(sec_hashes, root)
    d2 = safe(sec_d2, root) or {}
    safe(sec_d3, root, d2)
    d1 = safe(sec_d1, root) or []
    kb = safe(sec_d4, root, d1, d2)
    safe(sec_d5, root, kb, d2)
    safe(sec_d6, root, d1)
    head("SUMMARY")
    for lvl in ("FAIL", "WARN"):
        say(lvl, f"{sum(1 for l in REPORT if l.startswith('[' + lvl + ']'))} line(s) flagged {lvl}")
    (root / "audit_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    print(f"\nReport written to {root / 'audit_report.txt'}")


if __name__ == "__main__":
    main()
