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
        """
        在minimax模式中
        max是所有可能的情况（包含安全和危险路径）
        min应该计算到达这个情况的最小化的步骤
        即一个搜索算法为一个策略
        通过evaluate函数推测最佳路径
        并移动
        同理适用于alphabeta算法
        """
        pass
    def evaluate(self,state):

        pass
    def MiniMax(self):
        """
        max所有可能的情况
        min应该计算到达这个情况的最小化的步骤
        即一个搜索算法为一个策略
        """
        pass
    def AlphaBeta(self):
        """
        剪枝算法应该经过evaluate函数进行评估
        寻找最优路径
        对于minmax进行剪枝
        """
        pass