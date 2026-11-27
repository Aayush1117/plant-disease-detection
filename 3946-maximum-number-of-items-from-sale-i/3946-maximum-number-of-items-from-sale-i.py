from typing import List

class Solution:
    def maximumSaleItems(self, items: List[List[int]], budget: int) -> int:
        dp = [0] * (budget + 1)
        min_price = min(price for _, price in items)

        for factor, price in items:
            # counts the item itself plus its free items
            gain = sum(1 for other, _ in items if other % factor == 0)

            # go backwards so each first copy is used at most once
            for b in range(budget, price - 1, -1):
                if dp[b - price] + gain > dp[b]:
                    dp[b] = dp[b - price] + gain

        # spend the leftover budget on the cheapest item
        return max(dp[b] + (budget - b) // min_price for b in range(budget + 1))