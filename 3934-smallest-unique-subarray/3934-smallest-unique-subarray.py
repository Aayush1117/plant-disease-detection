import random
from typing import List

class Solution:
    def smallestUniqueSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        mod = (1 << 61) - 1
        base = random.randrange(10**6, mod - 1)
        pw = [1] * (n + 1)
        pre = [0] * (n + 1)
        for i, x in enumerate(nums):
            pw[i + 1] = pw[i] * base % mod
            pre[i + 1] = (pre[i] * base + x) % mod

        def has_unique(L: int) -> bool:
            p = pw[L]
            seen = {}
            for i in range(n - L + 1):
                h = (pre[i + L] - pre[i] * p) % mod
                seen[h] = seen.get(h, 0) + 1
            return any(c == 1 for c in seen.values())

        lo, hi = 1, n
        while lo < hi:
            mid = (lo + hi) // 2
            if has_unique(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo