"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a State class for Task 1

@author: B20 (100528137 and 100531086 and 100331080)
@date: 11/10/2025

"""
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
from __future__ import annotations
from typing import Optional, Tuple, List
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

from a1_state import State
import a2_path as alg
from a3_agent import Agent
try:
    # 若 a3_agent 里把 May_Hinger 写成模块函数，也能直接导入使用
    from a3_agent import May_Hinger  # type: ignore
except Exception:
    May_Hinger = None  # 可能不存在（例如写在 Agent 里）

Coord = Tuple[int, int]

# ========== 工具 ==========
def is_valid(state: State, rc: Coord) -> bool:
    r, c = rc
    return 0 <= r < state.m and 0 <= c < state.n

def is_legal(state: State, rc: Coord) -> bool:
    return is_valid(state, rc) and state.result[rc[0]][rc[1]] > 0

def apply_minus_one(state: State, rc: Coord) -> None:
    r, c = rc
    if hasattr(state, "Change_Data"):
        state.Change_Data(r, c)
    elif hasattr(state, "change_data"):
        state.change_data(r, c)
    else:
        if state.result[r][c] > 0:
            state.result[r][c] -= 1

def all_cleared(state: State) -> bool:
    return all(v <= 0 for row in state.result for v in row)

def is_hinger_in_grid(grid: List[List[int]], rc: Coord) -> bool:
    """静态判桥（仅用于结算时基于快照的红色标记）"""
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    r, c = rc
    if not (0 <= r < rows and 0 <= c < cols): return False
    if grid[r][c] <= 0: return False
    # 左右
    z = 0
    if c-1 >= 0 and grid[r][c-1] == 0: z += 1
    if c+1 < cols and grid[r][c+1] == 0: z += 1
    if z >= 2: return True
    # 上下
    z = 0
    if r-1 >= 0 and grid[r-1][c] == 0: z += 1
    if r+1 < rows and grid[r+1][c] == 0: z += 1
    return z >= 2

def is_hinger_rc(state: State, rc: Coord) -> bool:
    """
    统一用 a3 的 May_Hinger 判桥，自动适配常见签名：
    - May_Hinger(state, r, c) / May_Hinger(grid, r, c)
    - May_Hinger(state, (r,c)) / May_Hinger(grid, (r,c))
    - Agent().May_Hinger(...)
    不可用时回退到静态判桥 is_hinger_in_grid。
    """
    r, c = rc

    # a3_agent 模块函数
    if May_Hinger:
        try:
            return bool(May_Hinger(state, r, c))        # type: ignore
        except TypeError:
            try:
                return bool(May_Hinger(state.result, r, c))  # type: ignore
            except TypeError:
                try:
                    return bool(May_Hinger(state, (r, c)))   # type: ignore
                except TypeError:
                    try:
                        return bool(May_Hinger(state.result, (r, c)))  # type: ignore
                    except Exception:
                        pass
        except Exception:
            pass

    # Agent 实例方法
    try:
        ag = Agent()
        fn = getattr(ag, "May_Hinger", None)
        if fn:
            try:
                return bool(fn(state, r, c))
            except TypeError:
                try:
                    return bool(fn(state.result, r, c))
                except TypeError:
                    try:
                        return bool(fn(state, (r, c)))
                    except TypeError:
                        return bool(fn(state.result, (r, c)))
    except Exception:
        pass

    # 兜底
    return is_hinger_in_grid(state.result, rc)

ALGO_NAMES = ["A*", "BFS", "DFS", "IDDFS", "MiniMax", "AlphaBeta"]

def run_path_algo(name: str, start: Coord, end: Coord) -> Optional[List[Coord]]:
    """仅路径型算法"""
    if name == "A*":    return alg.path_astar(start, end)
    if name == "BFS":   return alg.path_BFS(start, end)
    if name == "DFS":   return alg.path_DFS(start, end)
    if name == "IDDFS": return alg.path_IDDFS(start, end)
    return None

# ========== UI ==========
CELL = 28
PAD  = 20

class GameUI:
    def __init__(self, rows: int, cols: int):
        self.rows, self.cols = rows, cols
        self.root = tk.Tk()
        self.root.title("Hinger A4 – Manual Grid Input")

        # 棋面与状态
        self.grid: List[List[int]] = [[0]*cols for _ in range(rows)]  # 手动录入
        self.state = State([row[:] for row in self.grid])
        self.turn_name = "HumanA"
        self.game_over = False
        self.ever_hinger = False
        self.path_overlay: List[Coord] = []
        self._hinger_snapshot: Optional[List[List[int]]] = None

        self.data_entry = True     # 录入模式
        self.entry_cells: List[List[tk.Entry]] = []

        alg.grid = self.grid       # 同步给 a2

        # 画布
        w, h = self.cols * CELL + 2*PAD, self.rows * CELL + 2*PAD
        self.canvas = tk.Canvas(self.root, width=w, height=h, bg="white")
        self.canvas.grid(row=0, column=0, rowspan=20)
        self.canvas.bind("<Button-1>", self.on_click)

        # 控件区
        ctr = tk.Frame(self.root); ctr.grid(row=0, column=1, sticky="n")
        tk.Label(ctr, text="Mode").grid(row=0, column=0, sticky="w")
        self.mode_var = tk.StringVar(value="Human vs AI")
        ttk.Combobox(ctr, textvariable=self.mode_var,
                     values=["Human vs AI", "Human vs Human"],
                     state="readonly", width=18).grid(row=0, column=1, pady=2)

        tk.Label(ctr, text="AI Algorithm").grid(row=1, column=0, sticky="w")
        self.algo_var = tk.StringVar(value="A*")
        ttk.Combobox(ctr, textvariable=self.algo_var,
                     values=ALGO_NAMES, state="readonly",
                     width=18).grid(row=1, column=1, pady=2)

        self.btn_edit    = tk.Button(ctr, text="Edit Grid",        command=self.on_edit_grid)
        self.btn_confirm = tk.Button(ctr, text="Confirm Grid",     command=self.on_confirm_grid)
        self.btn_show    = tk.Button(ctr, text="Show Path",        command=self.on_show_path)
        self.btn_next    = tk.Button(ctr, text="Next Turn",        command=self.on_next_turn)
        self.btn_cmp     = tk.Button(ctr, text="Compare (timing)", command=self.on_compare)
        self.btn_reset   = tk.Button(ctr, text="Reset Game",       command=self.on_reset_game)

        self.btn_edit.grid(row=2, column=0, columnspan=2, sticky="ew", pady=4)
        self.btn_confirm.grid(row=3, column=0, columnspan=2, sticky="ew", pady=2)
        self.btn_show.grid(row=4, column=0, columnspan=2, sticky="ew", pady=8)
        self.btn_next.grid(row=5, column=0, columnspan=2, sticky="ew", pady=2)
        self.btn_cmp.grid(row=6, column=0, columnspan=2, sticky="ew", pady=2)
        self.btn_reset.grid(row=7, column=0, columnspan=2, sticky="ew", pady=8)

        # 初始禁用对局按钮
        self._set_controls_enabled(False)
        self.draw()
        self._create_entry_grid()

    # ---- 控件开关 ----
    def _set_controls_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        for btn in (self.btn_show, self.btn_next, self.btn_cmp, self.btn_reset):
            btn.config(state=state)

    # ---- 录入模式 ----
    def _create_entry_grid(self):
        self._destroy_entry_grid()
        self.entry_cells = []
        for r in range(self.rows):
            row_entries = []
            for c in range(self.cols):
                x = PAD + c*CELL + 2
                y = PAD + r*CELL + 2
                e = tk.Entry(self.root, width=3, justify="center")
                e.place(x=x, y=y, width=CELL-4, height=CELL-4)
                e.insert(0, str(self.grid[r][c]))
                row_entries.append(e)
            self.entry_cells.append(row_entries)
        self.data_entry = True
        self.draw()

    def _destroy_entry_grid(self):
        for row in self.entry_cells:
            for e in row:
                try: e.destroy()
                except: pass
        self.entry_cells = []

    def on_edit_grid(self):
        self.game_over = False
        self._hinger_snapshot = None
        self.path_overlay = []
        self._set_controls_enabled(False)
        self._create_entry_grid()

    def on_confirm_grid(self):
        if not self.entry_cells:
            return
        new_grid = [[0]*self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                txt = self.entry_cells[r][c].get().strip()
                try: v = int(txt)
                except: v = 0
                new_grid[r][c] = max(0, v)

        self.grid = new_grid
        self._destroy_entry_grid()
        self.state = State([row[:] for row in self.grid])
        if hasattr(self.state, "Get_Graph"):
            try: self.state.Get_Graph()
            except Exception: pass

        self.turn_name = "HumanA"
        self.game_over = False
        self.ever_hinger = False
        self._hinger_snapshot = None
        self.path_overlay = []
        alg.grid = self.grid
        self._set_controls_enabled(True)
        self.data_entry = False
        self.draw()

    # ---- 绘制 ----
    def draw(self):
        self.canvas.delete("all")
        alg.grid = self.grid

        # 背景 + 数字
        for r in range(self.rows):
            for c in range(self.cols):
                x1 = PAD + c*CELL
                y1 = PAD + r*CELL
                x2 = x1 + CELL
                y2 = y1 + CELL
                v = self.grid[r][c]
                fill = "#eaf7ff" if v > 0 else "#aaaaaa"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline="#999")
                if v > 0 and not self.data_entry:
                    self.canvas.create_text(
                        x1 + CELL/2, y1 + CELL/2, text=str(v), fill="#111",
                        font=("Arial", max(10, CELL//2), "bold")
                    )

        # 路径覆盖（对局中）
        if not self.data_entry:
            for rc in self.path_overlay:
                r, c = rc
                x1 = PAD + c*CELL
                y1 = PAD + r*CELL
                x2 = x1 + CELL
                y2 = y1 + CELL
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="#b2ffb2", outline="#999")
                v = self.grid[r][c]
                if v > 0:
                    self.canvas.create_text(
                        x1 + CELL/2, y1 + CELL/2, text=str(v), fill="#003300",
                        font=("Arial", max(10, CELL//2), "bold")
                    )

        # 结算后：用快照统一标红 hinger
        if self.game_over and self._hinger_snapshot:
            rows = len(self._hinger_snapshot)
            cols = len(self._hinger_snapshot[0])
            try:
                snap_state = State([row[:] for row in self._hinger_snapshot])
            except Exception:
                snap_state = None
            for r in range(rows):
                for c in range(cols):
                    rc = (r, c)
                    use_red = is_hinger_in_grid(self._hinger_snapshot, rc)
                    if snap_state is not None:
                        # 若希望也走 a3 的 May_Hinger，可替换上一行为：
                        # use_red = is_hinger_rc(snap_state, rc)
                        pass
                    if use_red:
                        x1 = PAD + c*CELL
                        y1 = PAD + r*CELL
                        x2 = x1 + CELL
                        y2 = y1 + CELL
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill="#ff8080", outline="#999")
                        v = self._hinger_snapshot[r][c]
                        if v > 0:
                            self.canvas.create_text(
                                x1 + CELL/2, y1 + CELL/2, text=str(v), fill="#111",
                                font=("Arial", max(10, CELL//2), "bold")
                            )

        mode = self.mode_var.get()
        algo = self.algo_var.get()
        title = "Editing" if self.data_entry else f"Turn: {self.turn_name}"
        self.root.title(f"Hinger A4 – {title} – Mode: {mode} – AI: {algo}")

    # 动态起终点（仅供路径算法内部使用）
    def _default_endpoints(self) -> Tuple[Optional[Coord], Optional[Coord]]:
        coords = [(r, c) for r in range(self.rows) for c in range(self.cols) if self.grid[r][c] > 0]
        if not coords: return (None, None)
        return (coords[0], coords[-1])

    # ---- 事件 ----
    def on_show_path(self):
        if self.data_entry or self.game_over: return
        if self.algo_var.get() in ("MiniMax", "AlphaBeta"):
            messagebox.showinfo("No Path", "MiniMax / AlphaBeta 为决策算法，不提供路径可视化。")
            return
        s, t = self._default_endpoints()
        if s is None or t is None:
            self.path_overlay = []; self.draw()
            messagebox.showinfo("No Path", "当前棋面没有可用起止点（全为 0）。")
            return
        p = run_path_algo(self.algo_var.get(), s, t)
        self.path_overlay = p or []
        self.draw()

    def on_compare(self):
        if self.data_entry or self.game_over: return
        if self.algo_var.get() in ("MiniMax", "AlphaBeta"):
            messagebox.showinfo("Unavailable", "MiniMax / AlphaBeta 不参与 a2.compare（路径计时对比）。")
            return
        s, t = self._default_endpoints()
        if s is None or t is None:
            messagebox.showinfo("No Path", "当前棋面没有可用起止点（全为 0）。")
            return
        alg.compare(s, t)

    def on_reset_game(self):
        if self.data_entry: return
        self.state = State([row[:] for row in self.grid])
        if hasattr(self.state, "Get_Graph"):
            try: self.state.Get_Graph()
            except Exception: pass
        self.turn_name = "HumanA"
        self.game_over = False
        self.ever_hinger = False
        self.path_overlay = []
        self._hinger_snapshot = None
        self.draw()

    def on_click(self, ev):
        if self.data_entry or self.game_over: return
        if self.mode_var.get() == "Human vs AI" and self.turn_name != "HumanA": return
        if self.mode_var.get() == "Human vs Human" and self.turn_name not in ("HumanA", "HumanB"): return

        r = (ev.y - PAD) // CELL
        c = (ev.x - PAD) // CELL
        if not (0 <= r < self.rows and 0 <= c < self.cols): return

        self.path_overlay = []  # 清除旧路径
        rc = (r, c)
        if not is_legal(self.state, rc):
            messagebox.showinfo("Illegal", "非法落子（越界/空格/不可落）。对手获胜。")
            self._end_with_winner(self._opponent_name()); return

        # 用 a3 的 May_Hinger 判桥
        is_hinger = is_hinger_rc(self.state, rc)
        if is_hinger:
            self._hinger_snapshot = [row[:] for row in self.state.result]

        apply_minus_one(self.state, rc)
        self.grid[r][c] = self.state.result[r][c]

        if is_hinger:
            self.ever_hinger = True
            messagebox.showinfo("Win", f"{self.turn_name} 在 hinger 上落子，获胜！（已在结算后标红）")
            self._lock_game(); return

        if all_cleared(self.state) and not self.ever_hinger:
            messagebox.showinfo("Draw", "全部清零且未触发 hinger，平局。")
            self._lock_game(); return

        self._swap_turn()
        if self.mode_var.get() == "Human vs AI" and self.turn_name == "AI":
            self._ai_move()
        self.draw()

    def on_next_turn(self):
        if self.data_entry or self.game_over: return
        if self.mode_var.get() == "Human vs AI" and self.turn_name == "AI":
            self._ai_move()
        elif self.mode_var.get() == "Human vs Human":
            messagebox.showinfo("Hint", "人–人模式请直接点击棋盘进行落子。")
        self.draw()

    # ---- AI ----
    def _ai_move(self):
        algo = self.algo_var.get()

        # 决策型：MiniMax / AlphaBeta
        if algo in ("MiniMax", "AlphaBeta"):
            ag = Agent()
            move = None
            try:
                if algo == "MiniMax":
                    if hasattr(ag, "MiniMax"):   move = ag.MiniMax(self.state)
                    elif hasattr(ag, "minimax"): move = ag.minimax(self.state)
                    elif hasattr(ag, "move"):    move = ag.move(self.state, "minimax")
                else:
                    if hasattr(ag, "AlphaBeta"):   move = ag.AlphaBeta(self.state)
                    elif hasattr(ag, "alphabeta"): move = ag.alphabeta(self.state)
                    elif hasattr(ag, "move"):      move = ag.move(self.state, "alphabeta")
            except Exception as e:
                messagebox.showerror("Agent Error", f"Agent 执行出错：\n{e}")
                self._lock_game(); return

            if (not isinstance(move, (tuple, list))) or len(move) != 2:
                messagebox.showinfo("Win", "AI 无法给出合法落子，人类获胜！")
                self._lock_game(); return

            rr, cc = int(move[0]), int(move[1])
            if not (0 <= rr < self.rows and 0 <= cc < self.cols) or self.state.result[rr][cc] <= 0:
                messagebox.showinfo("Win", "AI 给出无效落子，人类获胜！")
                self._lock_game(); return

            # 判桥
            if is_hinger_rc(self.state, (rr, cc)):
                self._hinger_snapshot = [row[:] for row in self.state.result]

            apply_minus_one(self.state, (rr, cc))
            self.grid[rr][cc] = self.state.result[rr][cc]

            if self._hinger_snapshot:
                self.ever_hinger = True
                messagebox.showinfo("Win", "AI 在 hinger 上落子，AI 获胜！（已在结算后标红）")
                self._lock_game(); return

            if all_cleared(self.state) and not self.ever_hinger:
                messagebox.showinfo("Draw", "全部清零且未触发 hinger，平局。")
                self._lock_game(); return

            self._swap_turn()
            return

        # 路径型：A*/BFS/DFS/IDDFS
        s, t = self._default_endpoints()
        if s is None or t is None:
            messagebox.showinfo("Win", "AI 无法行动（棋面全为 0）。人类获胜！")
            self._lock_game(); return

        p = run_path_algo(algo, s, t)
        if not p or len(p) < 2:
            messagebox.showinfo("Win", "AI 无可行路径。人类获胜！")
            self._lock_game(); return

        # 取路径上第一个可落点（跳过起点）
        next_rc = None
        for rc in p[1:]:
            if self.state.result[rc[0]][rc[1]] > 0:
                next_rc = rc; break
        if next_rc is None:
            messagebox.showinfo("Win", "AI 无合法落子。人类获胜！")
            self._lock_game(); return

        rr, cc = next_rc
        if is_hinger_rc(self.state, (rr, cc)):
            self._hinger_snapshot = [row[:] for row in self.state.result]

        apply_minus_one(self.state, (rr, cc))
        self.grid[rr][cc] = self.state.result[rr][cc]

        if self._hinger_snapshot:
            self.ever_hinger = True
            messagebox.showinfo("Win", "AI 在 hinger 上落子，AI 获胜！（已在结算后标红）")
            self._lock_game(); return

        if all_cleared(self.state) and not self.ever_hinger:
            messagebox.showinfo("Draw", "全部清零且未触发 hinger，平局。")
            self._lock_game(); return

        self._swap_turn()

    # ---- 小辅助 ----
    def _swap_turn(self):
        if self.mode_var.get() == "Human vs AI":
            self.turn_name = "AI" if self.turn_name == "HumanA" else "HumanA"
        else:
            self.turn_name = "HumanB" if self.turn_name == "HumanA" else "HumanA"

    def _opponent_name(self) -> str:
        if self.mode_var.get() == "Human vs AI":
            return "AI" if self.turn_name == "HumanA" else "HumanA"
        return "HumanB" if self.turn_name == "HumanA" else "HumanA"

    def _end_with_winner(self, name: str):
        messagebox.showinfo("Game Over", f"胜者：{name}")
        self._lock_game()

    def _lock_game(self):
        self.game_over = True
        self.draw()  # 结束后立即重绘，统一标红 hinger

    def run(self):
        self.root.mainloop()

# ========== 入口 ==========
if __name__ == "__main__":
    tmp = tk.Tk(); tmp.withdraw()
    rows = simpledialog.askinteger("Rows (m)", "请输入棋盘行数 m（建议 3–60）:", minvalue=2, maxvalue=200, parent=tmp)
    cols = simpledialog.askinteger("Cols (n)", "请输入棋盘列数 n（建议 3–60）:", minvalue=2, maxvalue=200, parent=tmp)
    tmp.destroy()
    if not rows or not cols:
        rows, cols = 15, 20
    ui = GameUI(rows, cols)
    ui.run()