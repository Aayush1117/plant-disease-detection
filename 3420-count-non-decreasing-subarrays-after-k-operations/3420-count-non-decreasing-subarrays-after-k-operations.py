from collections import deque
from itertools import accumulate
from typing import List


class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        P = [0] + list(accumulate(nums))   # prefix sums
        dq = deque()                       # starts of prefix-max segments in window [l, r]
        r = n - 1
        sum_max = 0                        # sum of prefix maxima over window
        ans = 0
        for l in range(n - 1, -1, -1):
            while dq and nums[dq[0]] <= nums[l]:
                p = dq.popleft()
                nxt = dq[0] if dq else r + 1
                sum_max -= nums[p] * (nxt - p)
            end = dq[0] if dq else r + 1
            sum_max += nums[l] * (end - l)
            dq.appendleft(l)
            while sum_max - (P[r + 1] - P[l]) > k:
                b = dq[-1]
                sum_max -= nums[b]
                if b == r:
                    dq.pop()
                r -= 1
            ans += r - l + 1
        return ans