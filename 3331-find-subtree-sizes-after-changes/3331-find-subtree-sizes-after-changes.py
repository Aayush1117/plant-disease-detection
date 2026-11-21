from typing import List

class Solution:
    def findSubtreeSizes(self, parent: List[int], s: str) -> List[int]:
        n = len(parent)
        children = [[] for _ in range(n)]
        for i in range(1, n):
            children[parent[i]].append(i)

        new_par = parent[:]
        stacks = [[] for _ in range(26)]
        order = []

        todo = [(0, False)]
        while todo:
            node, leaving = todo.pop()
            c = ord(s[node]) - 97
            if leaving:
                stacks[c].pop()
                continue
            order.append(node)
            if node != 0 and stacks[c]:
                new_par[node] = stacks[c][-1]
            stacks[c].append(node)
            todo.append((node, True))
            for ch in children[node]:
                todo.append((ch, False))

        size = [1] * n
        for node in reversed(order):
            if node != 0:
                size[new_par[node]] += size[node]
        return size