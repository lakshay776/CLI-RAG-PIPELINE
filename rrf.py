def reciprocal_rank_fusion(result_lists, k=60):

    fused_scores = {}

    for results in result_lists:

        for rank, item in enumerate(results):

            key = (item["text"], item["source"])

            if key not in fused_scores:
                fused_scores[key] = {
                    "text": item["text"],
                    "source": item["source"],
                    "score": 0
                }

            fused_scores[key]["score"] += 1 / (k + rank + 1)

    # sort by fused score
    sorted_results = sorted(
        fused_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return sorted_results