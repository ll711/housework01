"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a State class for Task 1

@author: B20 (100528137 and 100531086 and 100331080)
@date: 11/10/2025

"""
from __future__ import annotations
#main game loop
# load a1, a2, a3 files and give a GUI graphical interface to choose which moode to
# run(matichine VS machine, human VS machine)
#in the GUI, show a star path, and import the path from a3 file
# show the path on the GUI and give who need to active ahead
"""
这个环节就是搭积木的环节
把所有做好的模块全部放入其中
并最终完成交付
"""
from typing import List, Tuple, Optional
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

from a1_state import State               # 读取 true_hinger_global_coords、IS_Hinger、Search_Node
import a2_path as alg                    # 调算法：path_astar/path_BFS/path_DFS/path_IDDFS

Coord = Tuple[int, int]

CELL = 32
PAD  = 14

ALGO_NAMES = ["A*", "BFS", "DFS", "IDDFS"]


# ---------- 小工具：用 a2_path 跑路径 ----------
def run_algo(name: str, start: Coord, end: Coord) -> Optional[List[Coord]]:
    if name == "A*":
        return alg.path_astar(start, end)
    if name == "BFS":
        return alg.path_BFS(start, end)
    if name == "DFS":
        return alg.path_DFS(start, end)
    if name == "IDDFS":
        return alg.path_IDDFS(start, end)
    return None


class GameUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hinger A4 – 简洁版")

        # ---- 先询问 m, n ----
        rows = simpledialog.askinteger("Board", "Rows (m ≥ 2):", minvalue=2, parent=self.root)
        cols = simpledialog.askinteger("Board", "Cols (n ≥ 2):", minvalue=2, parent=self.root)
        if not rows or not cols:
            rows, cols = 10, 15

        self.rows, self.cols = rows, cols

        # 交互状态
        self.grid: List[List[int]] = [[0]*self.cols for _ in range(self.rows)]
        self.state: Optional[State] = None
        self.start: Coord = (0, 0)
        self.end:   Coord = (self.rows-1, self.cols-1)

        self.turn_name = "HumanA"
        self.game_over = False
        self.ever_hinger = False

        # 结束后展示用的快照
        self._hinger_snapshot: Optional[List[List[int]]] = None
        self._hinger_snapshot_coords: Optional[List[Coord]] = None

        # ------- UI 结构 -------
        w = self.cols * CELL + 2 * PAD
        h = self.rows * CELL + 2 * PAD
        self.canvas = tk.Canvas(self.root, width=w, height=h, bg="white")
        self.canvas.grid(row=0, column=0, rowspan=30)
        self.canvas.bind("<Button-1>", self.on_click)

        ctrl = tk.Frame(self.root)
        ctrl.grid(row=0, column=1, sticky="n")

        # 先输入棋盘（格内 Entry）
        self.btn_confirm = tk.Button(ctrl, text="Confirm Board", command=self.on_confirm_board)
        self.btn_confirm.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        ttk.Label(ctrl, text="Mode").grid(row=1, column=0, sticky="w")
        self.mode_var = tk.StringVar(value="Human vs AI")
        ttk.Combobox(ctrl, textvariable=self.mode_var,
                     values=["Human vs AI", "Human vs Human"],
                     state="readonly", width=14).grid(row=1, column=1, padx=4, pady=2, sticky="e")

        ttk.Label(ctrl, text="AI Algorithm").grid(row=2, column=0, sticky="w")
        self.algo_var = tk.StringVar(value="A*")
        ttk.Combobox(ctrl, textvariable=self.algo_var,
                     values=ALGO_NAMES, state="readonly", width=14).grid(row=2, column=1, padx=4, pady=2, sticky="e")

        self.btn_show = tk.Button(ctrl, text="Show Path", command=self.on_show_path, state="disabled")
        self.btn_show.grid(row=3, column=0, columnspan=2, sticky="ew", pady=2)

        self.btn_next = tk.Button(ctrl, text="Next Turn", command=self.on_next_turn, state="disabled")
        self.btn_next.grid(row=4, column=0, columnspan=2, sticky="ew", pady=2)

        self.btn_reset = tk.Button(ctrl, text="Reset (Re-enter)", command=self.on_reset)
        self.btn_reset.grid(row=5, column=0, columnspan=2, sticky="ew", pady=6)

        # 用于输入棋盘的 Entry
        self.entries: List[List[tk.Entry]] = []
        self._make_entries()
        self.draw()

    # ---------- A1 hinger 列表：刷新 + 查询 ----------
    def _refresh_hingers(self, touched_rc: Optional[Coord] = None):
        """
        刷新 a1_state.State.true_hinger_global_coords
        - 初次/重置后：全量 (full_scan=True)
        - 针对某一步：增量 node=self.state.Search_Node(r,c)；如无则全量兜底
        """
        if self.state is None:
            return
        try:
            if touched_rc is None:
                self.state.IS_Hinger(full_scan=True)
            else:
                r, c = touched_rc
                node = self.state.Search_Node(r, c) if hasattr(self.state, "Search_Node") else None
                if node is not None:
                    self.state.IS_Hinger(node=node)
                else:
                    self.state.IS_Hinger(full_scan=True)
        except TypeError:
            # 兼容无关键字版本
            try:
                self.state.IS_Hinger()
            except Exception:
                pass

    def _is_hinger_from_a1(self, rc: Coord) -> bool:
        coords = getattr(self.state, "true_hinger_global_coords", None)
        return bool(coords) and (rc in coords)

    # ---------- 绘制 ----------
    def draw(self):
        self.canvas.delete("all")

        # 网格
        for r in range(self.rows):
            for c in range(self.cols):
                x1 = PAD + c*CELL
                y1 = PAD + r*CELL
                x2 = x1 + CELL
                y2 = y1 + CELL
                v = self.grid[r][c]
                fill = "#f2f2f2" if v == 0 else "white"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline="#aaa")
                if v != 0 and not self.game_over:
                    # 对局中显示数字
                    self.canvas.create_text((x1+x2)//2, (y1+y2)//2,
                                            text=str(v), fill="#222",
                                            font=("Arial", max(10, CELL//2)))

        # 起止位置轻微标
        self._paint_cell(self.start, "#cfe8ff")
        self._paint_cell(self.end,   "#ffd5e1")

        # 结束后：按快照坐标统一标红 hinger
        if self.game_over and self._hinger_snapshot_coords:
            for (r, c) in self._hinger_snapshot_coords:
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    x1 = PAD + c*CELL
                    y1 = PAD + r*CELL
                    x2 = x1 + CELL
                    y2 = y1 + CELL
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="#ff8080", outline="#999")
                    if self._hinger_snapshot:
                        v = self._hinger_snapshot[r][c]
                        if v > 0:
                            self.canvas.create_text((x1+x2)//2, (y1+y2)//2,
                                                    text=str(v), fill="#111",
                                                    font=("Arial", max(10, CELL//2), "bold"))

        # 标题
        self.root.title(f"Hinger A4 – Turn: {self.turn_name} – Mode: {self.mode_var.get()} – AI: {self.algo_var.get()}")

    def _paint_cell(self, rc: Coord, color: str):
        r, c = rc
        x1 = PAD + c*CELL + 2
        y1 = PAD + r*CELL + 2
        x2 = x1 + CELL - 4
        y2 = y1 + CELL - 4
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    # ---------- 棋盘输入 ----------
    def _make_entries(self):
        """在棋盘上摆 Entry，让用户逐格输入数字。"""
        for r in range(self.rows):
            row_e: List[tk.Entry] = []
            for c in range(self.cols):
                x = PAD + c * CELL + 2
                y = PAD + r * CELL + 2
                e = tk.Entry(self.canvas, width=3, justify="center")
                self.canvas.create_window(x, y, window=e, anchor="nw", width=CELL-4, height=CELL-4)
                row_e.append(e)
            self.entries.append(row_e)

    def on_confirm_board(self):
        """读入 Entry，生成棋盘；创建 State；刷新 hinger 列表；启用按钮。"""
        for r in range(self.rows):
            for c in range(self.cols):
                txt = self.entries[r][c].get().strip()
                try:
                    v = int(txt)
                except:
                    v = 0
                self.grid[r][c] = v

        # 隐藏所有 Entry
        for row in self.entries:
            for e in row:
                e.destroy()
        self.entries.clear()

        # 创建 State + 初次全量刷新 hinger 列表
        self.state = State([row[:] for row in self.grid])
        if hasattr(self.state, "Get_Graph"):
            try:
                self.state.Get_Graph()
            except Exception:
                pass
        self._refresh_hingers(None)

        # 启用对局按钮
        self.btn_show.config(state="normal")
        self.btn_next.config(state="normal")
        self.btn_confirm.config(state="disabled")

        self.turn_name = "HumanA"
        self.game_over = False
        self.ever_hinger = False
        self._hinger_snapshot = None
        self._hinger_snapshot_coords = None

        # 同步给 a2_path
        alg.grid = self.grid
        self.draw()

    # ---------- 事件 ----------
    def on_click(self, ev):
        if self.game_over or self.state is None:
            return

        r = (ev.y - PAD) // CELL
        c = (ev.x - PAD) // CELL
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return
        if self.mode_var.get() == "Human vs AI" and self.turn_name != "HumanA":
            return

        # 非法落子：越界/为0
        if self.grid[r][c] <= 0:
            messagebox.showinfo("Game Over", f"非法落子，{self._opponent_name()} 获胜")
            self.game_over = True
            self.draw()
            return

        rc = (r, c)

        # 刷新 + 判断 hinger（A1 列表）
        self._refresh_hingers(rc)
        if self._is_hinger_from_a1(rc):
            self._hinger_snapshot = [row[:] for row in self.grid]
            self._hinger_snapshot_coords = list(getattr(self.state, "true_hinger_global_coords", []))
            self._apply_minus_one(rc)
            self.ever_hinger = True
            self.game_over = True
            messagebox.showinfo("Win", f"{self.turn_name} 在 hinger 上落子，获胜！")
            self.draw()
            return

        # 正常落子（-1）
        self._apply_minus_one(rc)

        # 全盘清空但未触发 hinger → 平局
        if self._all_cleared() and not self.ever_hinger:
            self.game_over = True
            messagebox.showinfo("Draw", "所有格子清空且未触发 hinger，本局平局。")
            self.draw()
            return

        # 轮到对方
        self._swap_turn()
        self.draw()

        # 人机模式：AI 立即落子
        if self.mode_var.get() == "Human vs AI" and not self.game_over and self.turn_name == "AI":
            self._ai_move()

    def on_show_path(self):
        if self.state is None:
            return
        alg.grid = self.grid
        p = run_algo(self.algo_var.get(), self.start, self.end)
        if p:
            messagebox.showinfo("Path", f"{self.algo_var.get()} length = {len(p)}")
        else:
            messagebox.showinfo("Path", f"{self.algo_var.get()}: no path")

    def on_next_turn(self):
        if self.mode_var.get() == "Human vs AI" and self.turn_name == "AI" and not self.game_over:
            self._ai_move()
        elif self.mode_var.get() == "Human vs Human":
            messagebox.showinfo("Hint", "人-人模式请直接在棋盘点击。")
        self.draw()

    def on_reset(self):
        # 重新进入：再次输入 m、n 并逐格填写
        self.canvas.delete("all")
        for w in self.root.grid_slaves(column=1):
            # 保留右侧控件（不删）
            pass

        rows = simpledialog.askinteger("Board", "Rows (m ≥ 2):", minvalue=2, parent=self.root)
        cols = simpledialog.askinteger("Board", "Cols (n ≥ 2):", minvalue=2, parent=self.root)
        if not rows or not cols:
            rows, cols = 10, 15

        self.rows, self.cols = rows, cols
        self.grid = [[0]*self.cols for _ in range(self.rows)]
        self.state = None
        self.turn_name = "HumanA"
        self.game_over = False
        self.ever_hinger = False
        self._hinger_snapshot = None
        self._hinger_snapshot_coords = None

        # 调整画布大小
        w = self.cols * CELL + 2 * PAD
        h = self.rows * CELL + 2 * PAD
        self.canvas.config(width=w, height=h)

        # 重新布置 Entry
        self.entries = []
        self._make_entries()

        # 控制按钮状态
        self.btn_confirm.config(state="normal")
        self.btn_show.config(state="disabled")
        self.btn_next.config(state="disabled")

        self.draw()

    # ---------- AI 落子 ----------
    def _ai_move(self):
        if self.state is None or self.game_over:
            return

        alg.grid = self.grid
        path = run_algo(self.algo_var.get(), self.start, self.end)

        # 找路径上可落的下一格
        next_rc = None
        if path and len(path) >= 2:
            for rc in path[1:]:
                r, c = rc
                if self.grid[r][c] > 0:
                    next_rc = rc
                    break

        if next_rc is None:
            messagebox.showinfo("Game Over", "AI 无路可走 / 无合法落子，人类获胜。")
            self.game_over = True
            self.draw()
            return

        # 刷新 + 判断 hinger
        self._refresh_hingers(next_rc)
        if self._is_hinger_from_a1(next_rc):
            self._hinger_snapshot = [row[:] for row in self.grid]
            self._hinger_snapshot_coords = list(getattr(self.state, "true_hinger_global_coords", []))
            self._apply_minus_one(next_rc)
            self.ever_hinger = True
            self.game_over = True
            messagebox.showinfo("Win", "AI 在 hinger 上落子，AI 获胜！")
            self.draw()
            return

        # 正常 -1
        self._apply_minus_one(next_rc)

        # 全盘清空但未触发 hinger → 平局
        if self._all_cleared() and not self.ever_hinger:
            self.game_over = True
            messagebox.showinfo("Draw", "所有格子清空且未触发 hinger，本局平局。")
            self.draw()
            return

        self._swap_turn()
        self.draw()

    # ---------- 小辅助 ----------
    def _apply_minus_one(self, rc: Coord):
        r, c = rc
        if self.grid[r][c] > 0:
            self.grid[r][c] -= 1
            if self.state is not None:
                self.state.result[r][c] = self.grid[r][c]

    def _all_cleared(self) -> bool:
        return all(v <= 0 for row in self.grid for v in row)

    def _swap_turn(self):
        if self.mode_var.get() == "Human vs AI":
            self.turn_name = "AI" if self.turn_name == "HumanA" else "HumanA"
        else:
            self.turn_name = "HumanB" if self.turn_name == "HumanA" else "HumanA"

    def _opponent_name(self) -> str:
        if self.mode_var.get() == "Human vs AI":
            return "AI" if self.turn_name == "HumanA" else "HumanA"
        return "HumanB" if self.turn_name == "HumanA" else "HumanA"

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    GameUI().run()