from bisect import bisect_left
from typing import List

class Solution:
    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        potions.sort()
        m = len(potions)
        res = []
        for s in spells:
            need = (success + s - 1) // s  # ceil(success / s)
            idx = bisect_left(potions, need)
            res.append(m - idx)
        return res