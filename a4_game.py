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

"""
from __future__ import annotations
from typing import Optional, Tuple, List
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

from a1_state import State
import a2_path as alg          # 路径型算法
from a3_agent import Agent     # 决策型算法（MiniMax/AlphaBeta）

Coord = Tuple[int, int]

# ============ 工具函数 ============

def is_valid(state: State, rc: Coord) -> bool:
    r, c = rc
    return 0 <= r < state.m and 0 <= c < state.n

def is_legal(state: State, rc: Coord) -> bool:
    if not is_valid(state, rc):
        return False
    r, c = rc
    return state.result[r][c] > 0

def apply_minus_one(state: State, rc: Coord) -> None:
    """优先调用 A1.State 的 Change_Data/change_data；没有则直接 -1。"""
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
    """仅用于结算展示：在给定 grid 上判桥（grid[r][c] 必须>0；左右皆0 或 上下皆0）。"""
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    r, c = rc
    if not (0 <= r < rows and 0 <= c < cols):
        return False
    if grid[r][c] <= 0:
        return False
    # 左右
    zero_cnt = 0
    if c - 1 >= 0 and grid[r][c - 1] == 0: zero_cnt += 1
    if c + 1 < cols and grid[r][c + 1] == 0: zero_cnt += 1
    if zero_cnt >= 2: return True
    # 上下
    zero_cnt = 0
    if r - 1 >= 0 and grid[r - 1][c] == 0: zero_cnt += 1
    if r + 1 < rows and grid[r + 1][c] == 0: zero_cnt += 1
    return zero_cnt >= 2

ALGO_NAMES = ["A*", "BFS", "DFS", "IDDFS", "MiniMax", "AlphaBeta"]

def run_path_algo(name: str, start: Coord, end: Coord) -> Optional[List[Coord]]:
    """仅路径型算法（用于 Show Path / AI 路径跟随）"""
    if name == "A*":    return alg.path_astar(start, end)
    if name == "BFS":   return alg.path_BFS(start, end)
    if name == "DFS":   return alg.path_DFS(start, end)
    if name == "IDDFS": return alg.path_IDDFS(start, end)
    return None

# ============ Tkinter UI ============

CELL = 28
PAD  = 20

class GameUI:
    def __init__(self, rows: int, cols: int):
        self.rows, self.cols = rows, cols
        self.root = tk.Tk()
        self.root.title("Hinger A4 – Manual Grid Input (No Fixed Endpoints)")

        # 棋面与状态
        self.grid: List[List[int]] = [[0]*cols for _ in range(rows)]  # 初始录入模式，全 0
        self.state = State([row[:] for row in self.grid])
        self.turn_name = "HumanA"            # HumanA / HumanB / AI
        self.ever_hinger = False
        self.game_over: bool = False
        self.path_overlay: List[Coord] = []
        self._hinger_snapshot: Optional[List[List[int]]] = None

        # 录入模式
        self.data_entry: bool = True
        self.entry_cells: List[List[tk.Entry]] = []

        # 同步到 a2
        alg.grid = self.grid

        # 画布
        w, h = self.cols * CELL + 2 * PAD, self.rows * CELL + 2 * PAD
        self.canvas = tk.Canvas(self.root, width=w, height=h, bg="white")
        self.canvas.grid(row=0, column=0, rowspan=20)
        self.canvas.bind("<Button-1>", self.on_click)

        # 控制区
        ctr = tk.Frame(self.root)
        ctr.grid(row=0, column=1, sticky="n")

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

        # 按钮
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

        # 初始：录入模式，禁用对局相关按钮
        self._set_controls_enabled(False)
        self.draw()
        self._create_entry_grid()

    # ---------- 控件启停 ----------
    def _set_controls_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.btn_show.config(state=state)
        self.btn_next.config(state=state)
        self.btn_cmp.config(state=state)
        self.btn_reset.config(state=state)

    # ---------- 录入模式 ----------
    def _create_entry_grid(self):
        self._destroy_entry_grid()
        self.entry_cells = []
        for r in range(self.rows):
            row_entries = []
            for c in range(self.cols):
                x = PAD + c * CELL + 2
                y = PAD + r * CELL + 2
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
        """回到数据录入模式。"""
        self.game_over = False
        self._hinger_snapshot = None
        self.path_overlay = []
        self._set_controls_enabled(False)
        self._create_entry_grid()

    def on_confirm_grid(self):
        """确认录入，固化为网格，进入对局。"""
        if not self.entry_cells:
            return
        new_grid = [[0]*self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                txt = self.entry_cells[r][c].get().strip()
                try:
                    v = int(txt)
                except:
                    v = 0
                new_grid[r][c] = max(0, v)

        self.grid = new_grid
        self._destroy_entry_grid()
        self.state = State([row[:] for row in self.grid])

        # 若需要 A1 的区域链表：
        if hasattr(self.state, "Get_Graph"):
            try:
                self.state.Get_Graph()
            except Exception:
                pass

        self.turn_name = "HumanA"
        self.ever_hinger = False
        self.game_over = False
        self._hinger_snapshot = None
        self.path_overlay = []

        alg.grid = self.grid  # 同步到 a2
        self._set_controls_enabled(True)
        self.data_entry = False
        self.draw()

    # ---------- 绘制 ----------
    def draw(self):
        self.canvas.delete("all")
        alg.grid = self.grid  # 同步 a2

        # 底层格子
        for r in range(self.rows):
            for c in range(self.cols):
                x1 = PAD + c * CELL
                y1 = PAD + r * CELL
                x2 = x1 + CELL
                y2 = y1 + CELL
                v = self.grid[r][c]
                fill = "#eaf7ff" if v > 0 else "#aaaaaa"  # 可落点浅蓝；空格灰
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline="#999")

                # 录入模式不显示数字；对局显示数字
                if v > 0 and not self.data_entry:
                    self.canvas.create_text(
                        x1 + CELL/2, y1 + CELL/2,
                        text=str(v), fill="#111",
                        font=("Arial", max(10, CELL // 2), "bold")
                    )

        # 对局才显示路径（绿色覆盖）
        if not self.data_entry:
            for rc in self.path_overlay:
                r, c = rc
                x1 = PAD + c * CELL
                y1 = PAD + r * CELL
                x2 = x1 + CELL
                y2 = y1 + CELL
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="#b2ffb2", outline="#999")
                v = self.grid[r][c]
                if v > 0:
                    self.canvas.create_text(
                        x1 + CELL/2, y1 + CELL/2, text=str(v), fill="#003300",
                        font=("Arial", max(10, CELL // 2), "bold")
                    )

        # 结束后：基于胜利前快照把所有 hinger 标红
        if self.game_over and self._hinger_snapshot:
            rows = len(self._hinger_snapshot)
            cols = len(self._hinger_snapshot[0])
            for r in range(rows):
                for c in range(cols):
                    if is_hinger_in_grid(self._hinger_snapshot, (r, c)):
                        x1 = PAD + c * CELL
                        y1 = PAD + r * CELL
                        x2 = x1 + CELL
                        y2 = y1 + CELL
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill="#ff8080", outline="#999")
                        v = self._hinger_snapshot[r][c]
                        if v > 0:
                            self.canvas.create_text(
                                x1 + CELL/2, y1 + CELL/2,
                                text=str(v), fill="#111",
                                font=("Arial", max(10, CELL // 2), "bold")
                            )

        mode = self.mode_var.get()
        algo = self.algo_var.get()
        title = "Editing" if self.data_entry else f"Turn: {self.turn_name}"
        self.root.title(f"Hinger A4 – {title} – Mode: {mode} – AI: {algo}")

    # ---------- 动态起终点 ----------
    def _default_endpoints(self) -> tuple[Optional[Coord], Optional[Coord]]:
        """动态选择“第一个>0 与 最后一个>0”作为临时起终点（仅算法内部用，不显示）。"""
        coords = [(r, c) for r in range(self.rows) for c in range(self.cols) if self.grid[r][c] > 0]
        if not coords:
            return (None, None)
        return (coords[0], coords[-1])

    # ---------- 事件 ----------
    def on_show_path(self):
        if self.data_entry or self.game_over:
            return
        # MiniMax/AlphaBeta 无路径可视化
        if self.algo_var.get() in ("MiniMax", "AlphaBeta"):
            messagebox.showinfo("No Path", "MiniMax / AlphaBeta 是落子决策算法，不提供路径可视化。")
            return

        s, t = self._default_endpoints()
        if s is None or t is None:
            self.path_overlay = []
            self.draw()
            messagebox.showinfo("No Path", "当前棋盘没有可用起点与终点（全为 0）。")
            return
        p = run_path_algo(self.algo_var.get(), s, t)
        self.path_overlay = p or []
        self.draw()
        if p: print(f"{self.algo_var.get()} path length:", len(p))
        else: print(f"{self.algo_var.get()}: no path")

    def on_compare(self):
        if self.data_entry or self.game_over:
            return
        # 决策型算法不参与 compare
        if self.algo_var.get() in ("MiniMax", "AlphaBeta"):
            messagebox.showinfo("Unavailable", "MiniMax / AlphaBeta 不在 a2.compare 的对比范围内。")
            return

        s, t = self._default_endpoints()
        if s is None or t is None:
            messagebox.showinfo("No Path", "当前棋盘没有可用起点与终点（全为 0）。")
            return
        alg.compare(s, t)

    def on_reset_game(self):
        if self.data_entry:
            return
        self.state = State([row[:] for row in self.grid])
        self.turn_name = "HumanA"
        self.ever_hinger = False
        self.path_overlay = []
        self.game_over = False
        self._hinger_snapshot = None

        # 如果 A1 需要区域重建：
        if hasattr(self.state, "Get_Graph"):
            try:
                self.state.Get_Graph()
            except Exception:
                pass

        self.draw()

    def on_click(self, ev):
        """人类落子：点击格子 → (-1)；用 A1 的 IS_Hinger 判桥；结算后才标红。"""
        if self.data_entry or self.game_over:
            return
        # 人机：只有 HumanA 回合能点
        if self.mode_var.get() == "Human vs AI" and self.turn_name != "HumanA":
            return
        if self.mode_var.get() == "Human vs Human" and self.turn_name not in ("HumanA", "HumanB"):
            return

        r = (ev.y - PAD) // CELL
        c = (ev.x - PAD) // CELL
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return

        self.path_overlay = []  # 点击后清路径高亮
        rc = (r, c)
        if not is_legal(self.state, rc):
            messagebox.showinfo("Illegal", "非法落子（越界/空格/不可落）。对手获胜。")
            self._end_with_winner(self._opponent_name())
            return

        # —— 用 A1 的 IS_Hinger 判桥（不在对局中显示，只用于结算）
        if hasattr(self.state, "IS_Hinger"):
            is_hinger = self.state.IS_Hinger(r, c)
        else:
            # 兜底：若函数名大小写不同
            is_hinger = getattr(self.state, "Is_Hinger")(r, c)

        if is_hinger:
            # 赢之前拍快照，结束后用它把所有 hinger 标红
            self._hinger_snapshot = [row[:] for row in self.state.result]

        # 真正执行 -1
        apply_minus_one(self.state, rc)
        self.grid[r][c] = self.state.result[r][c]

        if is_hinger:
            self.ever_hinger = True
            messagebox.showinfo("Win", f"{self.turn_name} 在 hinger 上落子，获胜！（已在结算后标红 hinger）")
            self._lock_game()
            return

        if all_cleared(self.state) and not self.ever_hinger:
            messagebox.showinfo("Draw", "所有计数清空且未触发 hinger，本局平局。")
            self._lock_game()
            return

        self._swap_turn()
        if self.mode_var.get() == "Human vs AI" and self.turn_name == "AI":
            self._ai_move()
        self.draw()

    def on_next_turn(self):
        if self.data_entry or self.game_over:
            return
        if self.mode_var.get() == "Human vs AI":
            if self.turn_name == "AI":
                self._ai_move()
        elif self.mode_var.get() == "Human vs Human":
            messagebox.showinfo("Hint", "人–人模式请直接点击棋盘进行落子。")
        self.draw()

    # ---------- AI ----------
    def _ai_move(self):
        algo = self.algo_var.get()

        # 决策型算法：MiniMax / AlphaBeta
        if algo in ("MiniMax", "AlphaBeta"):
            agent = Agent()
            move = None
            try:
                if algo == "MiniMax":
                    if hasattr(agent, "MiniMax"):
                        move = agent.MiniMax(self.state)
                    elif hasattr(agent, "minimax"):
                        move = agent.minimax(self.state)
                    elif hasattr(agent, "move"):
                        move = agent.move(self.state, "minimax")
                else:  # AlphaBeta
                    if hasattr(agent, "AlphaBeta"):
                        move = agent.AlphaBeta(self.state)
                    elif hasattr(agent, "alphabeta"):
                        move = agent.alphabeta(self.state)
                    elif hasattr(agent, "move"):
                        move = agent.move(self.state, "alphabeta")
            except Exception as e:
                messagebox.showerror("Agent Error", f"Agent 执行出错：\n{e}")
                self._lock_game()
                return

            # 校验坐标合法性
            if (not isinstance(move, (tuple, list))) or len(move) != 2:
                messagebox.showinfo("Win", "AI 无法给出合法落子，人类获胜！")
                self._lock_game()
                return

            rr, cc = int(move[0]), int(move[1])
            if not (0 <= rr < self.rows and 0 <= cc < self.cols):
                messagebox.showinfo("Win", "AI 给出越界落子，人类获胜！")
                self._lock_game()
                return
            if self.state.result[rr][cc] <= 0:
                messagebox.showinfo("Win", "AI 给出不可落子的位置，人类获胜！")
                self._lock_game()
                return

            # 判桥（用 A1 的 IS_Hinger）
            if hasattr(self.state, "IS_Hinger"):
                is_hinger = self.state.IS_Hinger(rr, cc)
            else:
                is_hinger = getattr(self.state, "Is_Hinger")(rr, cc)

            if is_hinger:
                self._hinger_snapshot = [row[:] for row in self.state.result]

            # 扣 1
            apply_minus_one(self.state, (rr, cc))
            self.grid[rr][cc] = self.state.result[rr][cc]

            # 胜负
            if is_hinger:
                self.ever_hinger = True
                messagebox.showinfo("Win", "AI 在 hinger 上落子，AI 获胜！（已在结算后标红 hinger）")
                self._lock_game()
                return
            if all_cleared(self.state) and not self.ever_hinger:
                messagebox.showinfo("Draw", "所有计数清空且未触发 hinger，本局平局。")
                self._lock_game()
                return

            self._swap_turn()
            return

        # 路径型算法：A*/BFS/DFS/IDDFS
        s, t = self._default_endpoints()
        if s is None or t is None:
            messagebox.showinfo("Win", "AI 无法行动（棋盘全为 0），人类获胜！")
            self._lock_game()
            return

        p = run_path_algo(algo, s, t)
        if not p or len(p) < 2:
            messagebox.showinfo("Win", "AI 无路可走，人类获胜！")
            self._lock_game()
            return

        # 取路径上第一个可落点（跳过起点）
        next_rc = None
        for rc in p[1:]:
            rr, cc = rc
            if self.state.result[rr][cc] > 0:
                next_rc = rc
                break
        if next_rc is None:
            messagebox.showinfo("Win", "AI 无合法落子，人类获胜！")
            self._lock_game()
            return

        rr, cc = next_rc
        # 判桥（用 A1 的 IS_Hinger）
        if hasattr(self.state, "IS_Hinger"):
            is_hinger = self.state.IS_Hinger(rr, cc)
        else:
            is_hinger = getattr(self.state, "Is_Hinger")(rr, cc)

        if is_hinger:
            self._hinger_snapshot = [row[:] for row in self.state.result]

        # 扣 1
        apply_minus_one(self.state, (rr, cc))
        self.grid[rr][cc] = self.state.result[rr][cc]

        # 胜负
        if is_hinger:
            self.ever_hinger = True
            messagebox.showinfo("Win", "AI 在 hinger 上落子，AI 获胜！（已在结算后标红 hinger）")
            self._lock_game()
            return
        if all_cleared(self.state) and not self.ever_hinger:
            messagebox.showinfo("Draw", "所有计数清空且未触发 hinger，本局平局。")
            self._lock_game()
            return

        self._swap_turn()

    # ---------- 辅助 ----------
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
        self.draw()  # 结束后重绘以标红所有 hinger

    def run(self):
        self.root.mainloop()


# ============ 入口：启动先输入 m、n，进入录入模式 ============
if __name__ == "__main__":
    tmp = tk.Tk(); tmp.withdraw()
    rows = simpledialog.askinteger("Rows (m)", "请输入棋盘行数 m（建议 3–60）:", minvalue=2, maxvalue=200, parent=tmp)
    cols = simpledialog.askinteger("Cols (n)", "请输入棋盘列数 n（建议 3–60）:", minvalue=2, maxvalue=200, parent=tmp)
    tmp.destroy()
    if not rows or not cols:
        rows, cols = 15, 20  # 默认
    ui = GameUI(rows=rows, cols=cols)
    ui.run()

# ================= Tkinter UI =================

CELL = 28
PAD = 20

class GameUI:
    def __init__(self, rows=15, cols=20):
        self.rows, self.cols = rows, cols
        self.root = tk.Tk()
        self.root.title("Hinger A4 – Human/Human or Human/AI + BFS/DFS/IDDFS/A*")

        # 数据
        self.grid: List[List[int]] = make_grid(rows, cols, obstacle_rate=0.30, seed=random.randint(0, 9999))
        self.state = State([row[:] for row in self.grid])
        self.start: Coord = (0, 0)
        self.end:   Coord = (self.rows - 1, self.cols - 1)
        self.turn_name = "HumanA"  # HumanA / HumanB / AI
        self.ever_hinger = False
        self.path_overlay: List[Coord] = []
        self.game_over: bool = False
        self._hinger_snapshot: Optional[List[List[int]]] = None  # 胜利前棋面快照

        # 同步给 a2
        alg.grid = self.grid

        # 画布
        w, h = self.cols * CELL + 2 * PAD, self.rows * CELL + 2 * PAD
        self.canvas = tk.Canvas(self.root, width=w, height=h, bg="white")
        self.canvas.grid(row=0, column=0, rowspan=20)
        self.canvas.bind("<Button-1>", self.on_click)

        # 控制区
        ctr = tk.Frame(self.root)
        ctr.grid(row=0, column=1, sticky="n")

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

        tk.Button(ctr, text="New Random Map", command=self.on_new_map)\
            .grid(row=2, column=0, columnspan=2, sticky="ew", pady=4)
        tk.Button(ctr, text="Show Path", command=self.on_show_path)\
            .grid(row=3, column=0, columnspan=2, sticky="ew", pady=2)
        tk.Button(ctr, text="Next Turn", command=self.on_next_turn)\
            .grid(row=4, column=0, columnspan=2, sticky="ew", pady=8)
        tk.Button(ctr, text="Compare (timing)", command=self.on_compare)\
            .grid(row=5, column=0, columnspan=2, sticky="ew", pady=2)
        tk.Button(ctr, text="Reset Game", command=self.on_reset_game)\
            .grid(row=6, column=0, columnspan=2, sticky="ew", pady=8)

        self.draw()

    # ---------- 绘制 ----------
    def draw(self):
        self.canvas.delete("all")
        alg.grid = self.grid

        # 棋盘底层（颜色+数字）
        for r in range(self.rows):
            for c in range(self.cols):
                x1 = PAD + c * CELL
                y1 = PAD + r * CELL
                x2 = x1 + CELL
                y2 = y1 + CELL
                v = self.grid[r][c]

                if v <= 0:
                    fill = "#aaaaaa"        # 空格（不可点）
                else:
                    # 对局中隐藏 hinger；结束后在单独覆盖层揭示
                    fill = "#eaf7ff"        # 可点安全格（浅蓝）

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline="#999")

                if v > 0:
                    self.canvas.create_text(
                        x1 + CELL / 2, y1 + CELL / 2,
                        text=str(v), fill="#111",
                        font=("Arial", max(10, CELL // 2), "bold")
                    )

        # 起点/终点覆盖
        self._paint_cell(self.start, "#6cf")  # 蓝
        self._paint_cell(self.end,   "#fbb")  # 粉

        # 路径覆盖（仅在 Show Path 后存在）
        for rc in self.path_overlay:
            r, c = rc
            x1 = PAD + c * CELL
            y1 = PAD + r * CELL
            x2 = x1 + CELL
            y2 = y1 + CELL
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="#b2ffb2", outline="#999")
            v = self.grid[r][c]
            if v > 0:
                self.canvas.create_text(
                    x1 + CELL / 2, y1 + CELL / 2,
                    text=str(v), fill="#003300",
                    font=("Arial", max(10, CELL // 2), "bold")
                )

        # 游戏结束后：基于胜利前快照，揭示全部 hinger（红色）
        if self.game_over and self._hinger_snapshot:
            rows = len(self._hinger_snapshot)
            cols = len(self._hinger_snapshot[0])
            for r in range(rows):
                for c in range(cols):
                    if is_hinger_in_grid(self._hinger_snapshot, (r, c)):
                        x1 = PAD + c * CELL
                        y1 = PAD + r * CELL
                        x2 = x1 + CELL
                        y2 = y1 + CELL
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill="#ff8080", outline="#999")
                        v = self._hinger_snapshot[r][c]
                        if v > 0:
                            self.canvas.create_text(
                                x1 + CELL / 2, y1 + CELL / 2,
                                text=str(v), fill="#111",
                                font=("Arial", max(10, CELL // 2), "bold")
                            )

        # 标题
        self.root.title(
            f"Hinger A4 – Turn: {self.turn_name} – Mode: {self.mode_var.get()} – AI: {self.algo_var.get()}"
        )

    def _paint_cell(self, rc: Coord, color: str):
        r, c = rc
        x1 = PAD + c * CELL + 2
        y1 = PAD + r * CELL + 2
        x2 = x1 + CELL - 4
        y2 = y1 + CELL - 4
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    # ---------- 事件 ----------
    def on_new_map(self):
        self.grid = make_grid(self.rows, self.cols, obstacle_rate=0.30, seed=random.randint(0, 9999))
        self.state = State([row[:] for row in self.grid])
        self.start, self.end = (0, 0), (self.rows - 1, self.cols - 1)
        self.turn_name = "HumanA"
        self.ever_hinger = False
        self.path_overlay = []
        self.game_over = False
        self._hinger_snapshot = None
        self.draw()

    def on_reset_game(self):
        self.state = State([row[:] for row in self.grid])
        self.turn_name = "HumanA"
        self.ever_hinger = False
        self.path_overlay = []
        self.game_over = False
        self._hinger_snapshot = None
        self.draw()

    def on_show_path(self):
        p = run_algo(self.algo_var.get(), self.start, self.end)
        self.path_overlay = p or []
        self.draw()
        if p:
            print(f"{self.algo_var.get()} path length:", len(p))
        else:
            print(f"{self.algo_var.get()}: no path")

    def on_compare(self):
        alg.compare(self.start, self.end)

    def on_click(self, ev):
        # 玩家点击一次落子
        if self.game_over:
            return

        # 清掉上次路径高亮（避免每次点击都显示整条最短路）
        self.path_overlay = []

        r = (ev.y - PAD) // CELL
        c = (ev.x - PAD) // CELL
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return

        if self.mode_var.get() == "Human vs AI" and self.turn_name != "HumanA":
            return  # AI 回合不响应点击
        if self.mode_var.get() == "Human vs Human" and self.turn_name not in ("HumanA", "HumanB"):
            return

        rc = (r, c)
        if not is_legal(self.state, rc):
            messagebox.showinfo("Illegal", "非法落子（越界/空格/不可落）。对手获胜。")
            self._end_with_winner(self._opponent_name())
            return

        # 判桥（对局中）
        is_hinger = is_hinger_cell(self.state, rc)
        if is_hinger:
            # 保存胜利前的棋面快照
            self._hinger_snapshot = [row[:] for row in self.state.result]

        # 扣减
        apply_minus_one(self.state, rc)
        self.grid[r][c] = self.state.result[r][c]

        if is_hinger:
            self.ever_hinger = True
            messagebox.showinfo("Win", f"{self.turn_name} 在 hinger 上落子，获胜！\n已揭示所有 hinger。")
            self._lock_game()
            return

        if all_cleared(self.state) and not self.ever_hinger:
            messagebox.showinfo("Draw", "所有计数清空且未触发 hinger，本局平局。")
            self._lock_game()
            return

        self._swap_turn()
        # 人机：AI 立即行动
        if self.mode_var.get() == "Human vs AI" and self.turn_name == "AI":
            self._ai_move()

        self.draw()

    def on_next_turn(self):
        if self.mode_var.get() == "Human vs AI":
            if self.turn_name == "AI":
                self._ai_move()
        elif self.mode_var.get() == "Human vs Human":
            messagebox.showinfo("Hint", "人–人模式请直接点击棋盘进行落子。")
        self.draw()

    # ---------- AI 落子 ----------
    def _ai_move(self):
        if self.game_over:
            return

        p = run_algo(self.algo_var.get(), self.start, self.end)
        # 不自动显示路径（除非点 Show Path）
        # self.path_overlay = p or []

        if not p or len(p) < 2:
            messagebox.showinfo("Win", "AI 无法行动（无路/无合法格），人类获胜！")
            self._lock_game()
            return

        # 取路径上第一个可落点（非起点）
        next_rc = None
        for rc in p[1:]:
            rr, cc = rc
            if self.state.result[rr][cc] > 0:
                next_rc = rc
                break

        if next_rc is None:
            messagebox.showinfo("Win", "AI 无合法落子，人类获胜！")
            self._lock_game()
            return

        # 判桥
        is_hinger = is_hinger_cell(self.state, next_rc)
        if is_hinger:
            self._hinger_snapshot = [row[:] for row in self.state.result]

        # 扣减
        rr, cc = next_rc
        apply_minus_one(self.state, next_rc)
        self.grid[rr][cc] = self.state.result[rr][cc]

        if is_hinger:
            self.ever_hinger = True
            messagebox.showinfo("Win", "AI 在 hinger 上落子，AI 获胜！\n已揭示所有 hinger。")
            self._lock_game()
            return

        if all_cleared(self.state) and not self.ever_hinger:
            messagebox.showinfo("Draw", "所有计数清空且未触发 hinger，本局平局。")
            self._lock_game()
            return

        self._swap_turn()

    # ---------- 辅助 ----------
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
        self.draw()  # 结束后重绘揭示 hinger

    def run(self):
        self.root.mainloop()
    def win(self):
        pass


# ================= 入口 =================
if __name__ == "__main__":
    # 用一个临时 root 弹输入框（避免和 GameUI 的 Tk 重复）
    tmp = tk.Tk()
    tmp.withdraw()

    rows = simpledialog.askinteger("Rows (m)", "请输入棋盘行数 m（建议 5–60）:", minvalue=2, maxvalue=200, parent=tmp)
    cols = simpledialog.askinteger("Cols (n)", "请输入棋盘列数 n（建议 5–60）:", minvalue=2, maxvalue=200, parent=tmp)

    tmp.destroy()

    # 处理取消或无效输入：回退到默认 15×20
    if not rows or not cols:
        rows, cols = 15, 20

    ui = GameUI(rows=rows, cols=cols)
    ui.run()


