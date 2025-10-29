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
        self.model = ["minimax", "alphabeta"]
    def __str__(self):
        return self.name +self.size
    def move(state, model):
        # model一般默认为minimax
        # 在当前函数中应该获取链表中每一个图形的桥的数量
        """
        在minimax模式中
        建出来的树其中应该有两个孩子一个是桥一个是非桥
        在桥的情况下应该判断是否为dangerous还是safe
        并以此进行评估
        评估函数应该包含桥的数量以及桥的类型（区域中的桥）
        在非桥的情况下应该评估当前区域的奇偶数还有安全路径的数量
        """
        pass
    def evaluate(self,state):
        """
        在判断出在一个区域下真桥的数量为1的时候视为safe
        反之应该视为dangerous
        在非桥的情况下应该判断当前区域的安全路径下的奇偶数 其中边缘应该>内部
        优先级应该为一个区域下真桥的数量为1(即可能产生新的区域)>非桥区域的奇偶数>safe>dangerous
        """
        pass
    def MiniMax(self):
        """
        建出来的树其中应该有两个孩子一个是桥一个是非桥
        在桥的情况下应该判断是否为may_bridge还是bridge
        并以此进行评估
        评估函数应该包含桥的数量以及桥的类型（区域中的桥）
        在非桥的情况下应该评估当前区域的奇偶数还有安全路径的数量
        """
        pass
    def AlphaBeta(self):
        """
        剪枝算法应该经过evaluate函数进行评估
        寻找最优路径
        对于minmax进行剪枝
        """
        pass
    def MCTS(self):
        pass