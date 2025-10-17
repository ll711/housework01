"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a State class for Task 1

@author: B20 (100528137 and 100531086 and 100331080)
@date: 11/10/2025

"""
#在这个搜索文件中应该包含BFS,DFS,A*等算法
# 在其中这个文件中的所有数据应该先预处理 寻找出变化的大体范围在采用对应的算法

from collections import deque
from inspect import stack
from typing import List, Tuple, Optional
from a1_state import State

Coord=Tuple[int, int]

def _region_cells(state: State, min_x: int, max_x: int, min_y: int, max_y: int) -> List[Coord]:
    """收集某个区域范围内的所有“安全格子”(值>0) 坐标"""
    cells: List[Coord] = []
    for r in range(min_x, max_x + 1):
        for c in range(min_y, max_y + 1):
            if 0 <= r < state.m and 0 <= c < state.n and state.result[r][c] > 0:
                cells.append((r, c))
    return cells

def path_BFS (state: State) -> Optional[List[Coord]]:
    # 这个项目的BFS通过list将每一个图形放到每一个节点上
    #其中list的max,min是标注每个图形的大致范围
    #在这个算法中应该填入每个链表中安全路径的变化即list1,list2,list3中的第一个并以此类推
        node_ranges = []
        node = state.mylist.head
        while node is not None:
            node_ranges.append({
                "graph_num": node.get_graph_num(),
                "min_x": node.get_min_x(),
                "max_x": node.get_max_x(),
                "min_y": node.get_min_y(),
                "max_y": node.get_max_y(),
            })
            node = node.next

        # ====== 兜底：只有一个区域时，在同一区域里找一条路径 ======
        def bfs_between_cells(starts: List[Coord], goal: Coord) -> Optional[List[Coord]]:
            from collections import deque
            q = deque()
            visited = [[False] * state.n for _ in range(state.m)]
            parent: dict[Coord, Optional[Coord]] = {}
            for s in starts:
                q.append(s)
                visited[s[0]][s[1]] = True
                parent[s] = None

            # 用 8 邻域，和 Get_Graph 一致
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]

            while q:
                r, c = q.popleft()
                if (r, c) == goal:
                    # 回溯
                    path: List[Coord] = []
                    cur: Optional[Coord] = (r, c)
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    return path

                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < state.m and 0 <= nc < state.n:
                        if not visited[nr][nc] and state.result[nr][nc] > 0:
                            visited[nr][nc] = True
                            parent[(nr, nc)] = (r, c)
                            q.append((nr, nc))
            return None

        if len(node_ranges) < 2:
            # 单一区域：选“第一个非零格子”作为起点，“右下角最后一个非零格子”作为终点
            start_cell: Optional[Coord] = None
            for i in range(state.m):
                for j in range(state.n):
                    if state.result[i][j] > 0:
                        start_cell = (i, j)
                        break
                if start_cell:
                    break

            goal_cell: Optional[Coord] = None
            for i in range(state.m - 1, -1, -1):
                for j in range(state.n - 1, -1, -1):
                    if state.result[i][j] > 0:
                        goal_cell = (i, j)
                        break
                if goal_cell:
                    break

            if not start_cell or not goal_cell:
                return None

            return bfs_between_cells([start_cell], goal_cell)

        # ====== 正常：区域≥2，按“list1 -> listN”找跨区域路径 ======
        start_rng = node_ranges[0]
        goal_rng = node_ranges[-1]

        starts: List[Coord] = _region_cells(
            state,
            start_rng["min_x"], start_rng["max_x"],
            start_rng["min_y"], start_rng["max_y"]
        )
        goals_set = set(_region_cells(
            state,
            goal_rng["min_x"], goal_rng["max_x"],
            goal_rng["min_y"], goal_rng["max_y"]
        ))

        if not starts or not goals_set:
            return None

        # BFS 主体（与之前一致）
        from collections import deque
        q = deque()
        visited = [[False] * state.n for _ in range(state.m)]
        parent: dict[Coord, Optional[Coord]] = {}

        for s in starts:
            q.append(s)
            visited[s[0]][s[1]] = True
            parent[s] = None

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]

        reach: Optional[Coord] = None
        while q:
            r, c = q.popleft()
            if (r, c) in goals_set:
                reach = (r, c)
                break

            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < state.m and 0 <= nc < state.n:
                    if not visited[nr][nc] and state.result[nr][nc] > 0:
                        visited[nr][nc] = True
                        parent[(nr, nc)] = (r, c)
                        q.append((nr, nc))

        if reach is None:
            return None

        # 回溯
        path: List[Coord] = []
        cur: Optional[Coord] = reach
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

def path_DFS(state: State) -> Optional[List[Coord]]:
    """
    深度优先搜索（DFS）
    - 若只有 1 个区域：在同一区域内从第一个非零格子到最后一个非零格子；
    - 若 ≥ 2 个区域：从 list1 的任意安全格子出发，寻找能到达 listN 任意安全格子的路径；
    - 使用 8 邻域，与 Get_Graph 一致；返回任意一条可行路径（不保证最短）。
    """
    # 1) 收集区域边界
    node_ranges = []
    node = state.mylist.head
    while node is not None:
        node_ranges.append({
            "graph_num": node.get_graph_num(),
            "min_x": node.get_min_x(),
            "max_x": node.get_max_x(),
            "min_y": node.get_min_y(),
            "max_y": node.get_max_y(),
        })
        node = node.next

    # --- 工具：在网格上从 starts（取第一个）深挖到单一 goal ---
    def dfs_between_cells(starts: List[Coord], goal: Coord) -> Optional[List[Coord]]:
        rows, cols = state.m, state.n
        visited = [[False] * cols for _ in range(rows)]
        parent: dict[Coord, Optional[Coord]] = {}
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
        if not starts:
            return None
        start_cell = starts[0]
        stack: List[Coord] = [start_cell]
        parent[start_cell] = None

        while stack:
            r, c = stack.pop()
            if visited[r][c]:
                continue
            visited[r][c] = True

            if (r, c) == goal:
                # 回溯
                path: List[Coord] = []
                cur: Optional[Coord] = (r, c)
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path

            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and state.result[nr][nc] > 0:
                    if not visited[nr][nc]:
                        if (nr, nc) not in parent:
                            parent[(nr, nc)] = (r, c)
                        stack.append((nr, nc))
        return None

    # ===== 情况 A：只有 1 个区域（在同一区域里找一条路径）=====
    if len(node_ranges) < 2:
        start_cell: Optional[Coord] = None
        for i in range(state.m):
            for j in range(state.n):
                if state.result[i][j] > 0:
                    start_cell = (i, j)
                    break
            if start_cell:
                break

        goal_cell: Optional[Coord] = None
        for i in range(state.m - 1, -1, -1):
            for j in range(state.n - 1, -1, -1):
                if state.result[i][j] > 0:
                    goal_cell = (i, j)
                    break
            if goal_cell:
                break

        if not start_cell or not goal_cell:
            return None
        return dfs_between_cells([start_cell], goal_cell)

    # ===== 情况 B：≥2 区域（从 list1 深度优先找到 listN）=====
    start_rng = node_ranges[0]
    goal_rng = node_ranges[-1]

    starts: List[Coord] = _region_cells(
        state,
        start_rng["min_x"], start_rng["max_x"],
        start_rng["min_y"], start_rng["max_y"]
    )
    goals_set = set(_region_cells(
        state,
        goal_rng["min_x"], goal_rng["max_x"],
        goal_rng["min_y"], goal_rng["max_y"]
    ))

    if not starts or not goals_set:
        return None

    rows, cols = state.m, state.n
    visited = [[False] * cols for _ in range(rows)]
    parent: dict[Coord, Optional[Coord]] = {}
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    stack: List[Coord] = [starts[0]]
    parent[starts[0]] = None
    reach: Optional[Coord] = None

    while stack:
        r, c = stack.pop()
        if visited[r][c]:
            continue
        visited[r][c] = True

        if (r, c) in goals_set:
            reach = (r, c)
            break

        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and state.result[nr][nc] > 0:
                if not visited[nr][nc]:
                    if (nr, nc) not in parent:
                        parent[(nr, nc)] = (r, c)
                    stack.append((nr, nc))

    if reach is None:
        return None

    # 回溯
    path: List[Coord] = []
    cur: Optional[Coord] = reach
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path




def path_IDDFS(start, end):
    # 这个项目的BFS通过list将每一个图形放到每一个节点上
    # 其中list的max,min是标注每个图形的大致范围
    #在这个算法中应该采用先进先出+先进后出的思想完成即先广度后深度
    #例如list1,list2,list3中先搜索list1的第一个安全路径变化然后搜索list2的第一个安全路径变化
    #然后搜索list3的第一个安全路径变化然后再回到list1搜索list1的第二个安全路径变化
    return None
def path_astar(start,end):
    # 这个项目的A*通过list将每一个图形放到每一个节点上
    # 其中list的max,min是标注每个图形的大致范围
    # A* 算法是 f(n) = g(n) + h(n)
    # g(n) 是从起点到当前节点的所有安全路径(即路径长度)
    # h(n) 是从当前节点到目标节点的没有完成的路径的估计代价(启发式函数)
    # 在这个算法中应该填入每个链表中安全路径的变化即list1,list2,list3中的第一个并以此类推
    return None
def compare():
    # 比较以上四种算法的优缺点
    # 通过计算每个算法的时间长度进行比较
    return None
def min_safe(start,end):
    # 计算从start到end的最短安全路径
    #推测最短应该为A*算法或者IDDFS算法
    return None

def tester():
    # 测试以上所有函数
    state = State.from_file("test_map.txt")
    start = (0,0)
    end = (5,5)
    print("BFS:",path_BFS(start,end))
    print("DFS:",path_DFS(start,end))
    print("IDDFS:",path_IDDFS(start,end))
    print("A*:",path_astar(start,end))
    print("Min Safe Path:",min_safe(start,end))
    compare()