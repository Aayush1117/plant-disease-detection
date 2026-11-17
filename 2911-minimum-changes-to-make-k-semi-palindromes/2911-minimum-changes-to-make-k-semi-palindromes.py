class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        INF = float("inf")

        divisors = [[] for _ in range(n + 1)]
        for length in range(2, n + 1):
            for d in range(1, length):
                if length % d == 0:
                    divisors[length].append(d)

        cost = [[INF] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                length = j - i + 1
                best = INF
                for d in divisors[length]:
                    m = length // d
                    changes = 0
                    for r in range(d):
                        for t in range(m // 2):
                            if s[i + r + t * d] != s[i + r + (m - 1 - t) * d]:
                                changes += 1
                    best = min(best, changes)
                cost[i][j] = best

        dp = [[INF] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        for i in range(1, n + 1):
            for j in range(1, min(k, i // 2) + 1):
                for l in range(2 * (j - 1), i - 1):
                    if dp[l][j - 1] < INF:
                        dp[i][j] = min(dp[i][j], dp[l][j - 1] + cost[l][i - 1])
        return dp[n][k]