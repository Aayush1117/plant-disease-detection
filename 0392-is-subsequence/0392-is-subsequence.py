class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        i = 0
        n = len(s)
        for ch in t:
            if i == n:
                break
            if ch == s[i]:
                i += 1
        return i == n