from typing import Optional


class Solution:
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        def leaves(node: Optional[TreeNode], out: list) -> list:
            if not node:
                return out
            if not node.left and not node.right:
                out.append(node.val)
            leaves(node.left, out)
            leaves(node.right, out)
            return out

        return leaves(root1, []) == leaves(root2, [])