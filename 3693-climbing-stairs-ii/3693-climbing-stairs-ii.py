from typing import List

class Solution:
    def climbStairs(self, n: int, costs: List[int]) -> int:
        INF = float("inf")
        dp = [INF] * (n + 1)
        dp[0] = 0
        for j in range(1, n + 1):
            for k in (1, 2, 3):
                if j - k >= 0:
                    dp[j] = min(dp[j], dp[j - k] + k * k + costs[j - 1])
        return dp[n]
        