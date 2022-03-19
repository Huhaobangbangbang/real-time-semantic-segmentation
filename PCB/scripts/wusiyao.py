"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2022/3/17 11:25 PM
"""
class ListNode():
    def __init__(self, val):
        if isinstance(val, int):
            self.val = val
            self.next = None

        elif isinstance(val, list):
            self.val = val[0]
            self.next = None
            cur = self
            for i in val[1:]:
                cur.next = ListNode(i)
                cur = cur.next

    def gatherAttrs(self):
        return ", ".join("{}: {}".format(k, getattr(self, k)) for k in self.__dict__.keys())

    def __str__(self):
        return self.__class__.__name__ + " {" + "{}".format(self.gatherAttrs()) + "}"


class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        if not head:
            return head
        evenHead = head.next
        odd, even = head, evenHead
        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next
        odd.next = evenHead


        # 输出排序好的数字，以数组形式保存

        huhao_wusiyao = []
        while head.next:
            huhao_wusiyao.append(head.val)
            head = head.next
        huhao_wusiyao.append(head.val)

        return huhao_wusiyao

if __name__ == "__main__":
    a = Solution()
    ori_listNode = ListNode([2, 1, 3, 5, 6, 4, 7])
    end_list = a.oddEvenList(ori_listNode)
    print(end_list)
