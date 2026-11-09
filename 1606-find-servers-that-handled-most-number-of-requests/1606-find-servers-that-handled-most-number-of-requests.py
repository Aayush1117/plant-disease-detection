

import heapq
from typing import List

class Solution:
    def busiestServers(self, k: int, arrival: List[int], load: List[int]) -> List[int]:
        free = list(range(k))          # virtual indices
        heapq.heapify(free)
        busy = []                      # (end_time, server_id)
        cnt = [0] * k

        for i, (a, l) in enumerate(zip(arrival, load)):
            # release servers that finished
            while busy and busy[0][0] <= a:
                _, s = heapq.heappop(busy)
                heapq.heappush(free, i + ((s - i) % k))

            # lift stale virtual indices to >= i
            while free and free[0] < i:
                x = heapq.heappop(free)
                x += ((i - x + k - 1) // k) * k
                heapq.heappush(free, x)

            if not free:
                continue                # dropped
            j = heapq.heappop(free)
            s = j % k
            cnt[s] += 1
            heapq.heappush(busy, (a + l, s))

        m = max(cnt)
        return [i for i, c in enumerate(cnt) if c == m]