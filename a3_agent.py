"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a State class for Task 1

@author: B20 (100528137 and 100531086 and 100331080)
@date: 11/10/2025

"""
import collections # 需要导入collections模块用于BFS
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
        在非桥的情况下应该评估当前区域的奇偶数还有可能产生may_bridge
        """
        pass
    def evaluate(self,state):
        """
        在判断出在一个区域下bridge的时候视为safe
        反之如果为may_bridge应该视为dangerous
        在非桥的情况下应该判断当前区域的安全路径下的奇偶数 其中边缘应该>内部
        优先级应该为一个区域下bridge>安全路径的数量>可能产生桥的数量>may_bridge
        """
        pass
    def MiniMax(self):
        """
        建出来的树其中应该有两个孩子一个是桥一个是非桥
        在桥的情况下应该判断是否为may_bridge还是bridge
        并以此进行评估
        评估函数应该包含桥的数量以及桥的类型（区域中的桥）
        在非桥的情况下应该评估当前区域安全路径的数量已经可能产生桥的路径
        """
        pass
    def AlphaBeta(self):
        """
        剪枝算法应该经过evaluate函数进行评估
        寻找最优路径
        对于minmax进行剪枝
        """
        pass

    def May_Hinger(self, node):
        """
        检查指定节点的桥梁状态
        1.增加周围非零点四联通性检查
        2.增加仅对计数器数值为1的单元格进行桥梁判断
        仅返回当前节点桥梁坐标信息
        """
        # 获取节点的边界信息
        min_x, max_x = node.get_min_x(), node.get_max_x()
        min_y, max_y = node.get_min_y(), node.get_max_y()

        # 获取节点的二维数组数据（局部坐标）
        node_grid = node.get_grid().data

        # 计算节点网格的尺寸（包含周围一圈空白）
        node_rows = len(node_grid)
        node_cols = len(node_grid[0]) if node_rows > 0 else 0

        # 创建临时列表存储当前节点的桥梁全局坐标
        node_tinger_global_coords = []

        # 遍历节点二维数组中的所有单元格（局部坐标）
        for i_local in range(node_rows):
            for j_local in range(node_cols):
                # 只检查非零单元格（活跃区域）
                if node_grid[i_local][j_local] == 0:
                    continue

                # 将局部坐标转换为全局坐标
                # 节点网格的(0,0)对应全局的(min_x-1, min_y-1)
                # 因为节点网格包含周围一圈空白
                i_global = min_x - 1 + i_local
                j_global = min_y - 1 + j_local

                # 确保转换后的全局坐标在有效范围内
                if not (0 <= i_global < self.m and 0 <= j_global < self.n):
                    continue

                # 新增检查：只对有计数器值为1的单元格才进行桥梁判断
                if self.result[i_global][j_global] != 1:
                    continue  # 跳过计数器值不为1的单元格

                # 检查直接相邻的左右和上下方向（使用全局坐标）
                directions = [
                    [(0, -1), (0, 1)],  # 左右方向（同一行）
                    [(-1, 0), (1, 0)]  # 上下方向（同一列）
                ]

                may_hinger = False
                for dir_pair in directions:
                    zero_count = 0
                    # 检查该方向上的两个相邻格子（使用全局坐标）
                    for dr, dc in dir_pair:
                        r_adj_global = i_global + dr
                        c_adj_global = j_global + dc
                        # 确保坐标在全局网格边界内
                        if 0 <= r_adj_global < self.m and 0 <= c_adj_global < self.n:
                            if self.result[r_adj_global][c_adj_global] == 0:
                                zero_count += 1

                    # 如果同一行/列有两个相邻零格子，则当前点初步判定为桥梁
                    if zero_count >= 2:
                        may_hinger = True
                        break

            # 额外检查1：如果周围邻居数少于等于1，则不是桥梁
            # 额外检查2：如果被标记为桥梁，检查周围非零点是否全部连通
            if may_hinger:
                # 收集周围八个方向的非零点（摩尔邻居）
                neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue  # 跳过自身
                        r_adj = i_global + dr
                        c_adj = j_global + dc
                        if 0 <= r_adj < self.m and 0 <= c_adj < self.n:
                            if self.result[r_adj][c_adj] > 0:
                                neighbors.append((r_adj, c_adj))

                # 检查1：如果邻居数少于等于1，则不是桥梁
                if len(neighbors) <= 1:
                    may_hinger = False
                # 检查2：如果周围非零邻居数大于1，检查它们是否全部连通（四连通）
                elif len(neighbors) > 1:
                    visited = set()
                    queue = collections.deque()
                    start = neighbors[0]
                    queue.append(start)
                    visited.add(start)

                    # 使用BFS检查连通性
                    while queue:
                        r, c = queue.popleft()
                        # 检查四方向（上下左右）
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            neighbor_pos = (nr, nc)
                            if (neighbor_pos in neighbors and
                                    neighbor_pos not in visited):
                                visited.add(neighbor_pos)
                                queue.append(neighbor_pos)

                    # 如果所有邻居点都连通，则不是桥梁
                    if len(visited) == len(neighbors):
                        may_hinger = False

            if may_hinger:
                # 添加到临时数组，使用全局坐标(i, j)同体系的二维数组坐标(i_global, j_global)
                node_tinger_global_coords.append((i_global, j_global))

        # 返回当前节点的桥梁坐标数组
        return node_tinger_global_coords
    def MCTS(self):
        pass