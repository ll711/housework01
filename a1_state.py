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

    def Moves(self, row: int, col: int) -> bool:
        """
        修改指定鼠标点击坐标位置的数字，减一操作
        不对零进行操作，仅判断，由a4的win函数判断
        :param row: 全局网格地行坐标
        :param col: 全局网格的列坐标
        如果已经是零，则保持不变
        :return: 修改成功返回True，否则返回False
        只修改鼠标点击位置所在的区域节点内的数据（现交由Change_Date修改）
        """

        # 检查当前格内计数器数字是否为零
        if self.result[row][col] == 0:
            print(f"坐标({row}, {col})的值已经是零，不进行操作")
            return False

        # 通过鼠标坐标找到包含该坐标的节点
        node = self.Search_Node(row, col)
        if node is None:
            # 如果找不到对应节点，可能是点击了非活跃区域
            print(f"警告: 坐标({row}, {col})不在任何活跃区域内")
            return False

    def Change_Data(self, row: int, col: int, node) -> bool:
        """
        修改指定坐标位置的数字，减一操作
        :param row: 行坐标（全局）
        :param col: 列坐标（全局）
        :param node: 包含该坐标的节点
        :return: 修改成功返回True，否则返回False
        只修改鼠标点击位置所在的区域节点内的数据
        """
        # 获取节点的边界信息
        min_x, max_x = node.get_min_x(), node.get_max_x()  # 列的范围
        min_y, max_y = node.get_min_y(), node.get_max_y()  # 行的范围

        # 获取节点的网格数据
        node_grid = node.get_grid().data

        # 计算节点网格的尺寸（包含周围一圈空白）
        node_rows = len(node_grid)
        node_cols = len(node_grid[0]) if node_rows > 0 else 0

        # 将全局坐标转换为节点局部坐标
        # 节点网格的(0,0)对应全局的(min_y-1, min_x-1)
        local_row = row - (min_y - 1)
        local_col = col - (min_x - 1)

        # 确保局部坐标在节点网格范围内
        if (0 <= local_row < node_rows and
                0 <= local_col < node_cols):

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
            print(f"节点边界: min_y={min_y}, max_y={max_y}, min_x={min_x}, max_x={max_x}")
            print(f"节点网格尺寸: {node_rows} x {node_cols}")
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
                    # 检查是否存在相同图形（比较边界与二维数组内容）
                    is_dup = False
                    for node in self.mylist:
                        if (node.get_min_x() == mincol and node.get_max_x() == maxcol and
                                node.get_min_y() == minrow and node.get_max_y() == maxrow and
                                node.get_grid().data == subgrid):
                            is_dup = True
                            break

                    # 不重复则尾插，重复则跳过
                    if not is_dup:
                        last_graph_num = self.mylist.tail.get_graph_num() if self.mylist.tail else 0
                        self.mylist.append(
                            mincol, maxcol, minrow, maxrow,
                            subgrid, bridge_num=0, graph_num=last_graph_num + 1
                        )
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


