# Bug Fix Summary: Experimental Pipeline

## The Bug

The experimental pipeline was using **ALL passages** from the context instead of restricting to seeds + KG-expanded passages, giving it an unfair advantage.

## What Was Fixed

### Before (Buggy):

```python
# Used ALL passages from chunks_index
for ent in chunks_index:
    for seq_str in chunks_index[ent]:
        candidates.append((doc_id, text))  # ❌ ALL passages!
```

### After (Fixed):

```python
# 1. Build seed nodes from multi-view retrieval
seed_nodes = []
for doc_id, score, text in seed_passages:
    node = TextNode(text=text, id_=doc_id)
    seed_nodes.append(NodeWithScore(node=node, score=float(score)))

# 2. Apply KG expansion from seeds
expanded_nodes = expansion_pp._postprocess_nodes(seed_nodes, query_bundle)

# 3. Build candidates ONLY from seeds + expanded
candidates = []
for nws in expanded_nodes:
    candidates.append((doc_id, text))  # ✅ Only seeds + expanded!
```

## Implementation Flow (Fixed)

1. **Multi-view retrieval** → Get 8 seed passages ✅
2. **KG expansion (1-hop)** → Expand from seeds using `KGRetrievePostProcessor` ✅
3. **Candidate pool** → Seeds + expanded only (~8-30 passages) ✅
4. **Knapsack selection** → Select from restricted pool ✅

## Expected Results After Fix

**Before Fix (with bug):**

- Experimental: 76.0%
- Original: 62.0%
- Difference: +14.0% (inflated)

**After Fix (fair comparison):**

- Experimental: ~65-70% (estimated)
- Original: 62.0%
- Difference: +3-8% (realistic)

## Next Steps

1. **Re-run experimental pipeline** with fix
2. **Compare results** fairly
3. **Analyze real improvements** (if any)

## Verification

The fix ensures:

- ✅ Candidate pool restricted to seeds + KG-expanded
- ✅ Fair comparison with original pipeline
- ✅ Implementation matches project proposal
- ✅ No unfair advantage from accessing all passages
