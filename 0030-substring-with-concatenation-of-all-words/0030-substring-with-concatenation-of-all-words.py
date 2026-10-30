from collections import Counter
from typing import List

class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        w = len(words[0])
        n = len(words)
        total = w * n
        if len(s) < total:
            return []

        need = Counter(words)
        res = []

        for o in range(w):
            left = o
            count = 0
            seen = Counter()
            for right in range(o, len(s) - w + 1, w):
                word = s[right:right + w]
                if word in need:
                    seen[word] += 1
                    count += 1
                    while seen[word] > need[word]:
                        lw = s[left:left + w]
                        seen[lw] -= 1
                        count -= 1
                        left += w
                    if count == n:
                        res.append(left)
                        lw = s[left:left + w]
                        seen[lw] -= 1
                        count -= 1
                        left += w
                else:
                    seen.clear()
                    count = 0
                    left = right + w
        return res