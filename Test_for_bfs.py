from a1_state import State
from a2_path import path_BFS
from typing import List, Tuple, Optional

Coord = Tuple[int, int]


def print_grid_with_path(grid: List[List[int]], path: Optional[List[Coord]]) -> None:
    """
    将路径覆盖到网格上打印：
    - 原网格数字保持不变
    - 路径上的格子用 9 标记（方便看）
    """
    g = [row[:] for row in grid]
    if path:
        for r, c in path:
            g[r][c] = 9
    for row in g:
        print(" ".join(f"{v:2d}" for v in row))


def run_case(name: str, data: List[List[int]]) -> None:
    print("=" * 60)
    print(f"Case: {name}")
    s = State(data)
    s.Get_Graph()  # 一定要先构建区域链表

    # 打印区域信息，帮助确认 list1/listN
    node = s.mylist.head
    idx = 1
    while node:
        print(f"list{idx}: x[{node.get_min_x()},{node.get_max_x()}], "
              f"y[{node.get_min_y()},{node.get_max_y()}], "
              f"graph_num={node.get_graph_num()}")
        idx += 1
        node = node.next
    print(f"regions = {len(s.mylist)}")

    path = path_BFS(s)

    print("\nBFS 路径结果：")
    if path:
        print(f"找到路径，步数: {len(path)}")
        print(path)
    else:
        print("未找到路径。")

    print("\n覆盖路径后的网格（9 表示路径）：")
    print_grid_with_path(data, path)
    print()


def main():
    # 说明：
    # 1=可走（安全格子），0=障碍
    # Case1/Case2 设计为可在 8 邻域下较容易找到跨区域路径
    # Case3 为 4 邻域也能通的典型“走廊”式地图

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

    # 典型 4 邻域“走廊”通路（若你把 BFS 改回 4 邻域，依然能通）
    case3 = [
        [1, 1, 1, 0, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1],
    ]

    run_case("CASE-1 (8邻域更友好)", case1)
    run_case("CASE-2 (8邻域更友好)", case2)
    run_case("CASE-3 (4邻域也可通)", case3)


if __name__ == "__main__":
    main()
