from typing import List

class Solution:
    def maximumSaleItems(self, items: List[List[int]], budget: int) -> int:
        n = len(items)
        cnt = [0] * (n + 1)
        for f, _ in items:
            cnt[f] += 1
        multiples = [0] * (n + 1)
        for f in range(1, n + 1):
            multiples[f] = sum(cnt[f::f])
        pmin = min(p for _, p in items)
        slots = sorted((p, multiples[f] - 1) for f, p in items)
        k = 0
        left = budget
        for p, m in slots:
            if p > 2 * pmin:
                break
            if m == 0:
                continue
            t = min(m, left // p)
            k += t
            left -= t * p
            if t < m:
                break
        return 2 * k + left // pmin