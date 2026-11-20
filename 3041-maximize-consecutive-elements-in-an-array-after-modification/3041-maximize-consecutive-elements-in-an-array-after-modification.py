from collections import defaultdict
from typing import List

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = defaultdict(int)
        for x in nums:
            dp[x + 1] = dp[x] + 1
            dp[x] = dp[x - 1] + 1
        return max(dp.values())