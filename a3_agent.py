"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a State class for Task 1

@author: B20 (100528137 and 100531086 and 100331080)
@date: 11/10/2025

"""
# genet ic algorithm
class Agent:
    def __init__(self):
        self.name = "B20"
        self.size = "m,n"
        self.modle = ["minimax", "alphabeta"]
    def __str__(self):
        return self.name +self.size
    def move(state, model):
        # model一般默认为minimax
        # 在当前函数中应该获取链表中每一个图形的桥的数量

        pass