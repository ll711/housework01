#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a State class for Task 1

@author: B20 (100528137 and 100531086 and 100331080)
@date: 11/10/2025

"""
import tkinter as tk
from MyList import MyList  # Keep dependency
import collections # 需要导入collections模块用于BFS
# 保持对 MyList 的依赖

class State:
    # Core state class that holds the grid data and graph logic
    # 保存网格数据与图相关逻辑的核心类
    def __init__(self, data):
        self.m = len(data)
        self.n = len(data[0]) if self.m > 0 else 0
        self.result = [row[:] for row in data] # 全局网格数据

        from MyList import MyList
        self.mylist = MyList() # 储存活跃区域的链表
        self.node = self.mylist.append(
            0, 0, 0, 0,
            [[0] * (self.n + 2) for _ in range(self.m + 2)],
            bridge_num=0,
            graph_num=1
        )
        self.hinger_coords = []  # 桥梁坐标列表
        self.hinger_count = 0  # 桥梁数量
        self.first_check = True  # 是否是第一次检查
        self.affected_nodes = set()  # 受影响的节点集合
        # 新增：用于记录鼠标事件（像素与网格坐标）
        self.mouse_events = []  # [{'x': int, 'y': int, 'row': int|None, 'col': int|None}, ...]

    def record_mouse(self, x: int, y: int, row: int | None = None, col: int | None = None) -> None:
        """
        记录一次鼠标事件
        :param x: 鼠标点击的像素坐标 x
        :param y: 鼠标点击的像素坐标 y
        :param row: 可选的网格坐标行号
        :param col: 可选的网格坐标列号
        """
        # 记录鼠标事件信息
        self.mouse_events.append({'x': x, 'y': y, 'row': row, 'col': col})

        # 如果提供了行列信息，找到对应的节点并标记为受影响
        if row is not None and col is not None:
            node = self.Search_Node(row, col)
            if node is not None:
                self.affected_nodes.add(node)
                # 同时修改数据
                self.Change_Data(row, col)

    def Change_Data(self, row, col):
        """
        修改指定坐标位置的数字，减一操作
        :param row: 行坐标
        :param col: 列坐标
        如果已经是零，则保持不变
        :return: 修改成功返回True，否则返回False
        只修改鼠标点击位置所在的区域节点内的数据
        """
        # 找到包含该坐标的节点
        node = self.Search_Node(row, col)
        if node is None:
            # 如果找不到对应节点，可能是点击了非活跃区域
            print(f"警告: 坐标({row}, {col})不在任何活跃区域内")
            return False

        # 获取节点的网格数据
        node_grid = node.get_grid().data

        # 将节点局部坐标转换为全局坐标
        node_min_x, node_min_y = node.get_min_x(), node.get_min_y()
        local_row = row - node_min_x
        local_col = col - node_min_y

        # 确保局部坐标在节点网格范围内
        if (0 <= local_row < len(node_grid) and
                0 <= local_col < len(node_grid[0])):

            # 获取当前值
            current_value = node_grid[local_row][local_col]

            # 如果当前值大于零，则减一
            if current_value > 0:
                # 更新节点网格数据
                node_grid[local_row][local_col] = current_value - 1

                # 同时更新全局网格数据
                self.result[row][col] = current_value - 1

                print(f"坐标({row}, {col})的值从{current_value}减少到{current_value - 1}")
                return True
            else:
                print(f"坐标({row}, {col})的值已经是零，保持不变")
                return False
        else:
            print(f"错误: 局部坐标({local_row}, {local_col})超出节点网格范围")
            return False

    def Get_Graph(self):
        graph = self.result
        grid = [[0] * self.n for _ in range(self.m)]
        visited = [[False] * self.n for _ in range(self.m)]

        dirs8 = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

        used_head = False

        for i in range(self.m):
            for j in range(self.n):
                if graph[i][j] == 0 or visited[i][j]:
                    continue

                # 队列与起点
                arr_ones = [(i, j)]
                head = 0

                # 边界与采样
                minrow = maxrow = i
                mincol = maxcol = j
                comp_cells = []  # (r, c, v)

                # 起点入采样，写入临时 grid（不清零 graph）
                v0 = graph[i][j]
                grid[i][j] = v0
                visited[i][j] = True
                comp_cells.append((i, j, v0))

                # BFS
                while head < len(arr_ones):
                    r, c = arr_ones[head]
                    head += 1

                    # 更新边界
                    if r < minrow: minrow = r
                    if r > maxrow: maxrow = r
                    if c < mincol: mincol = c
                    if c > maxcol: maxcol = c

                    # 扩展八邻域
                    for dr, dc in dirs8:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.m and 0 <= nc < self.n:
                            if not visited[nr][nc] and graph[nr][nc] != 0:
                                val = graph[nr][nc]
                                arr_ones.append((nr, nc))
                                grid[nr][nc] = val
                                visited[nr][nc] = True
                                comp_cells.append((nr, nc, val))

                                # 同步更新边界
                                if nr < minrow: minrow = nr
                                if nr > maxrow: maxrow = nr
                                if nc < mincol: mincol = nc
                                if nc > maxcol: maxcol = nc

                # 子网络裁剪：最小行/列对齐到 (1,1)，尺寸为(最大行+1, 最大列+1)
                rows_rel_max = (maxrow - minrow) + 1
                cols_rel_max = (maxcol - mincol) + 1
                subgrid = [[0] * (cols_rel_max + 1) for _ in range(rows_rel_max + 1)]
                for r, c, v in comp_cells:
                    rr = (r - minrow) + 1
                    cc = (c - mincol) + 1
                    subgrid[rr][cc] = v

                # 写入链表：头节点直写，其后尾插
                if not used_head and self.mylist.head is not None and self.mylist.head == self.mylist.tail:
                    head_node = self.mylist.head
                    head_node.set_min_x(mincol)
                    head_node.set_max_x(maxcol)
                    head_node.set_min_y(minrow)
                    head_node.set_max_y(maxrow)
                    head_node.set_grid_data(subgrid)
                    head_node.set_graph_num(1)
                    used_head = True
                else:
                    last_graph_num = self.mylist.tail.get_graph_num() if self.mylist.tail else 0
                    self.mylist.append(
                        mincol, maxcol, minrow, maxrow,
                        subgrid, bridge_num=0, graph_num=last_graph_num + 1
                    )

                # 清零本地缓冲 grid（不影响 self.result）
                for rr in range(self.m):
                    for cc in range(self.n):
                        grid[rr][cc] = 0

    def Search_Node(self, row, col):
        """
        根据坐标找到对应的链表节点
        :param row: 全局网格的行坐标
        :param col: 全局网格的列坐标
        :return: 对应的链表节点或 None
        每个节点代表一个活跃区域，有特定的坐标范围
        """
        current_node = self.mylist.head
        while current_node is not None:
            # 检查坐标是否在节点的活跃坐标集合中
            if hasattr(current_node, 'get_active_coords_set') and (row, col) in current_node.get_active_coords_set():
                return current_node
            # 备用方法：如果节点没有活跃坐标集合，使用边界检查
            elif hasattr(current_node, 'get_min_x'):
                min_x, max_x = current_node.get_min_x(), current_node.get_max_x()
                min_y, max_y = current_node.get_min_y(), current_node.get_max_y()
                if min_x <= row <= max_x and min_y <= col <= max_y:
                    # 进一步检查坐标是否确实活跃
                    if self.result[row][col] > 0:
                        return current_node
            current_node = current_node.next
        return None

    def IS_Hinger(self, node):
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

                #新增检查：只对有计数器值为1的单元格才进行桥梁判断
                if self.result[i_global][j_global] != 1:
                    continue #跳过计数器值不为1的单元格

                # 检查直接相邻的左右和上下方向（使用全局坐标）
                directions = [
                    [(0, -1), (0, 1)],  # 左右方向（同一行）
                    [(-1, 0), (1, 0)]  # 上下方向（同一列）
                ]

                is_hinger = False
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
                        is_hinger = True
                        break

            # 额外检查1：如果周围邻居数少于等于1，则不是桥梁
            # 额外检查2：如果被标记为桥梁，检查周围非零点是否全部连通
            if is_hinger:
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
                    is_hinger = False
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
                        is_hinger = False

            if is_hinger:
                # 添加到临时数组，使用全局坐标(i, j)同体系的二维数组坐标(i_global, j_global)
                node_tinger_global_coords.append((i_global, j_global))

        #返回当前节点的桥梁坐标数组
        return node_tinger_global_coords

    def numHingers(self):

        num = 0
        current_node = self.mylist.head
        while current_node is not None:
            num += len(self.IS_Hinger(current_node))
            current_node = current_node.next
        return num

    def numRegions(self):
        # Placeholder for returning number of regions
        # 返回活跃区域数量的占位函数
        return len(self.mylist)

    def moves(self):
        # Placeholder for returning possible moves
        # 返回可能移动的占位函数
        pass

def tester():
    import tkinter as tk

    m = int(input("enter rows m: "))
    n = int(input("enter columns n: "))

    cell_size = min(60, 800 // max(m, n)) if m and n else 60
    pad = cell_size // 2
    canvas_width = n * cell_size + 2 * pad
    canvas_height = m * cell_size + 2 * pad

    root = tk.Tk()
    root.title("game")
    root.geometry(f"{canvas_width}x{canvas_height}")

    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
    canvas.pack()

    for i in range(m + 1):
        y = pad + i * cell_size
        canvas.create_line(pad, y, pad + n * cell_size, y)
    for j in range(n + 1):
        x = pad + j * cell_size
        canvas.create_line(x, pad, x, pad + m * cell_size)

    entries = []
    for i in range(m):
        row_entries = []
        for j in range(n):
            entry = tk.Entry(root, width=5, justify='center')
            x = pad + j * cell_size
            y = pad + i * cell_size
            entry.place(x=x + 2, y=y + 2, width=cell_size - 4, height=cell_size - 4)
            row_entries.append(entry)
        entries.append(row_entries)

    data = [[0] * n for _ in range(m)]
    game_started = {'value': False}
    state_holder = {'state': None}

    def on_enter(event=None):
        from a1_state import State
        for i in range(m):
            for j in range(n):
                val = entries[i][j].get()
                try:
                    num = int(val)
                except ValueError:
                    num = 0
                data[i][j] = num
                entries[i][j].destroy()
                cx = pad + j * cell_size + cell_size / 2
                cy = pad + i * cell_size + cell_size / 2
                canvas.create_text(
                    cx, cy, text=str(num), tags=f"cell_{i}_{j}",
                    font=("Arial", int(cell_size // 2))
                )
        game_started['value'] = True
        state_holder['state'] = State(data)
        state_holder['state'].Get_Graph()

    def on_click(event):
        if not game_started['value']:
            return
        gx = event.x - pad
        gy = event.y - pad
        if gx < 0 or gy < 0:
            return
        col = int(gx // cell_size)
        row = int(gy // cell_size)
        if 0 <= row < m and 0 <= col < n:
            # 新增：将鼠标像素坐标与网格坐标传递给 State
            if state_holder['state'] is not None:
                state_holder['state'].record_mouse(event.x, event.y, row=row, col=col)

            if data[row][col] > 0:
                data[row][col] -= 1
                canvas.delete(f"cell_{row}_{col}")
                cx = pad + col * cell_size + cell_size / 2
                cy = pad + row * cell_size + cell_size / 2
                canvas.create_text(
                    cx, cy, text=str(data[row][col]), tags=f"cell_{row}_{col}",
                    font=("Arial", int(cell_size // 2))
                )

    root.bind('<Return>', on_enter)
    canvas.bind('<Button-1>', on_click)
    root.mainloop()

if __name__ == "__main__":
    # Program entry point
    # 程序入口
    tester()


