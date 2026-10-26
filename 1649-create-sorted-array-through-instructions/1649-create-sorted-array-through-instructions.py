from typing import List

class Solution:
    def createSortedArray(self, instructions: List[int]) -> int:
        MOD = 10**9 + 7
        size = max(instructions)
        tree = [0] * (size + 1)

        def update(i):
            while i <= size:
                tree[i] += 1
                i += i & -i

        def query(i):
            s = 0
            while i > 0:
                s += tree[i]
                i -= i & -i
            return s

        total = 0
        for i, x in enumerate(instructions):
            less = query(x - 1)
            greater = i - query(x)
            total += min(less, greater)
            update(x)

        return total % MOD