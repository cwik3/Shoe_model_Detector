# tools/retrieval.py
import numpy as np

def l2_normalize(x, eps=1e-12):
    n = np.linalg.norm(x)
    return x / max(eps, n)

def classify_with_margin(query_emb, faiss_index, id2sku, k=20, tau=0.30, delta=0.06):
    """
    Works with both IndexFlatIP (cosine/inner product) and L2 indexes.
    Converts FAISS outputs to a 'score' where HIGHER is better.
    Gates on:
        - score1 >= tau
        - (score1 - score2) >= delta
    """
    q = query_emb.astype(np.float32)
    # q = l2_normalize(q)  # keep commented if you already normalize upstream

    sims, idxs = faiss_index.search(q[np.newaxis, :], k)
    sims, idxs = sims[0], idxs[0]

    if len(idxs) == 0 or idxs[0] < 0:
        return None, 0.0

    # Detect metric and convert to "higher is better" score
    try:
        import faiss
        metric = getattr(faiss_index, "metric_type", faiss.METRIC_INNER_PRODUCT)
        # METRIC_L2 -> sims are squared L2 distances (lower better). Convert to scores.
        if metric == faiss.METRIC_L2:
            scores = -sims  # higher is better now
        else:
            scores = sims   # already higher is better
    except Exception:
        scores = sims  # fallback

    # Group top-k votes by class
    votes = {}
    for s, i in zip(scores, idxs):
        if i < 0:
            continue
        sku = id2sku[int(i)] if id2sku is not None else str(int(i))
        votes[sku] = votes.get(sku, 0.0) + float(s)

    if not votes:
        return None, 0.0

    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    top1_sku, top1_score = ranked[0]
    top2_score = ranked[1][1] if len(ranked) > 1 else -1e9

    # Raw top-1/2 for margin gate (using converted "scores")
    s1 = float(scores[0])
    s2 = float(scores[1]) if len(scores) > 1 else -1e9

    # Unknown gate
    if (s1 < float(tau)) or ((s1 - s2) < float(delta)):
        return None, 0.0

    # Pseudo-confidence based on separation
    denom = max(1e-6, abs(top1_score) + abs(top2_score))
    conf = float((top1_score - max(0.0, top2_score)) / denom)
    return top1_sku, conf
