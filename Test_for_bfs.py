"""
Test_search_compare.py
同时测试 & 对比 BFS 与 DFS：
- 是否找到路径
- 路径长度
- 多次运行耗时与平均耗时
- 网格可视化（路径用 9 标记）
"""

from typing import List, Tuple, Optional
import time

from a1_state import State
from a2_path import path_BFS, path_DFS

Coord = Tuple[int, int]


def print_grid_with_path(state: State, path: Optional[List[Coord]]) -> None:
    """在 state.result 上用 9 覆盖路径并打印"""
    grid = [row[:] for row in state.result]
    if path:
        for r, c in path:
            grid[r][c] = 9
    for row in grid:
        print(" ".join(f"{x:2d}" for x in row))


def time_one(func, state: State, runs: int = 7):
    """
    对某个搜索函数进行多次计时，返回：
    (是否找到, 路径长度, 单次耗时列表, 平均耗时)
    备注：BFS/DFS 不修改 state，本函数可安全复用同一个 state。
    """
    durations = []
    found = False
    path_len = 0

    # 预热一次，避免首次调用波动
    _ = func(state)

    for _ in range(runs):
        t0 = time.perf_counter()
        path = func(state)
        t1 = time.perf_counter()
        durations.append(t1 - t0)
        if path:
            found = True
            path_len = len(path)

    avg = sum(durations) / len(durations) if durations else 0.0
    return found, path_len, durations, avg


def run_case(name: str, grid_data: List[List[int]]):
    print("=" * 66)
    print(f"Case: {name}")

    # 1) 初始化 State 并构建区域链表
    s = State(grid_data)
    s.Get_Graph()

    # 2) 打印区域（list1、list2 …）信息，确认划分是否正确
    node = s.mylist.head
    if node is None:
        print("⚠️ 未检测到任何区域节点，请检查输入数据。")
    else:
        while node is not None:
            print(f"list{node.get_graph_num()}: "
                  f"x[{node.get_min_x()},{node.get_max_x()}], "
                  f"y[{node.get_min_y()},{node.get_max_y()}], "
                  f"graph_num={node.get_graph_num()}")
            node = node.next
    print(f"regions = {len(s.mylist)}")

    # 3) 路径与可视化：BFS
    bfs_path = path_BFS(s)
    print("\n🔹 BFS 结果：", end=" ")
    if bfs_path:
        print(f"找到路径（长度 {len(bfs_path)}）")
    else:
        print("未找到路径。")
    print("BFS 覆盖路径后的网格（9 表示路径）：")
    print_grid_with_path(s, bfs_path)

    # 4) 路径与可视化：DFS
    dfs_path = path_DFS(s)
    print("\n🔸 DFS 结果：", end=" ")
    if dfs_path:
        print(f"找到路径（长度 {len(dfs_path)}）")
    else:
        print("未找到路径。")
    print("DFS 覆盖路径后的网格（9 表示路径）：")
    print_grid_with_path(s, dfs_path)

    # 5) 多次计时对比
    bfs_found, bfs_len, bfs_runs, bfs_avg = time_one(path_BFS, s, runs=7)
    dfs_found, dfs_len, dfs_runs, dfs_avg = time_one(path_DFS, s, runs=7)

    print("\n⏱️ 速度对比（单位：秒）")
    print("BFS: 单次耗时 =", ", ".join(f"{d:.6f}" for d in bfs_runs),
          f"| 平均 = {bfs_avg:.6f}s | 找到={bfs_found} | 长度={bfs_len}")
    print("DFS: 单次耗时 =", ", ".join(f"{d:.6f}" for d in dfs_runs),
          f"| 平均 = {dfs_avg:.6f}s | 找到={dfs_found} | 长度={dfs_len}")

    faster = "BFS" if bfs_avg < dfs_avg else "DFS"
    print(f"\n✅ 本用例更快：{faster}（按平均耗时）\n")


def main():
    # 说明：1=可走（安全格子），0=障碍
    # 下面三组样例与我们之前讨论一致：
    case1 = [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1],
    ]

    case2 = [
        [1, 0, 1, 0, 1],
        [1, 1, 1, 0, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]

    case3 = [
        [1, 1, 1, 0, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1],
    ]

    run_case("CASE-1 (8邻域更友好)", case1)
    run_case("CASE-2 (稍微复杂)", case2)
    run_case("CASE-3 (4邻域也可通)", case3)


if __name__ == "__main__":
    main()
