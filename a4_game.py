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
from typing import Optional, Tuple, List
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import random

from a1_state import State
import a2_path as alg  # 使用 a2_path 的 path_*、compare()，并同步其全局 grid
from a3_agent import Agent  # 仅导入以确保文件存在

Coord = Tuple[int, int]

# ================= 工具函数 =================

def make_grid(rows=15, cols=20, obstacle_rate=0.30, seed: Optional[int] = None) -> List[List[int]]:
    if seed is not None:
        random.seed(seed)
    g = [[1 if random.random() > obstacle_rate else 0 for _ in range(cols)] for _ in range(rows)]
    # 保证起止可通
    g[0][0] = 1
    g[rows - 1][cols - 1] = 1
    return g

def is_valid(state: State, rc: Coord) -> bool:
    r, c = rc
    return 0 <= r < state.m and 0 <= c < state.n

def is_legal(state: State, rc: Coord) -> bool:
    if not is_valid(state, rc):
        return False
    r, c = rc
    return state.result[r][c] > 0

def apply_minus_one(state: State, rc: Coord) -> None:
    r, c = rc
    # 兼容不同命名
    if hasattr(state, "Change_Data"):
        state.Change_Data(r, c)
    elif hasattr(state, "change_data"):
        state.change_data(r, c)
    else:
        if state.result[r][c] > 0:
            state.result[r][c] -= 1

def all_cleared(state: State) -> bool:
    return all(v <= 0 for row in state.result for v in row)

def is_hinger_cell(state: State, rc: Coord) -> bool:
    """
    对局中用于判胜：该格 >0 且左右皆 0 或上下皆 0
    """
    r, c = rc
    if not (0 <= r < state.m and 0 <= c < state.n):
        return False
    if state.result[r][c] <= 0:
        return False

    m, n = state.m, state.n

    # 左右两个邻居
    zero_cnt = 0
    if c - 1 >= 0 and state.result[r][c - 1] == 0:
        zero_cnt += 1
    if c + 1 < n and state.result[r][c + 1] == 0:
        zero_cnt += 1
    if zero_cnt >= 2:
        return True

    # 上下两个邻居
    zero_cnt = 0
    if r - 1 >= 0 and state.result[r - 1][c] == 0:
        zero_cnt += 1
    if r + 1 < m and state.result[r + 1][c] == 0:
        zero_cnt += 1
    return zero_cnt >= 2

def is_hinger_in_grid(grid: List[List[int]], rc: Coord) -> bool:
    """
    结束后揭示用：在传入的 grid 上判断 rc 是否是 hinger（grid[r][c] 必须 >0）
    """
    rows, cols = len(grid), len(grid[0]) if grid else 0
    r, c = rc
    if not (0 <= r < rows and 0 <= c < cols):
        return False
    if grid[r][c] <= 0:
        return False

    # 左右
    zero_cnt = 0
    if c - 1 >= 0 and grid[r][c - 1] == 0:
        zero_cnt += 1
    if c + 1 < cols and grid[r][c + 1] == 0:
        zero_cnt += 1
    if zero_cnt >= 2:
        return True

    # 上下
    zero_cnt = 0
    if r - 1 >= 0 and grid[r - 1][c] == 0:
        zero_cnt += 1
    if r + 1 < rows and grid[r + 1][c] == 0:
        zero_cnt += 1
    return zero_cnt >= 2

ALGO_NAMES = ["A*", "BFS", "DFS", "IDDFS", "MiniMax", "AlphaBeta"]

def run_algo(name: str, start: Coord, end: Coord) -> Optional[List[Coord]]:
    if name == "A*":
        return alg.path_astar(start, end)
    if name == "BFS":
        return alg.path_BFS(start, end)
    if name == "DFS":
        return alg.path_DFS(start, end)
    if name == "IDDFS":
        return alg.path_IDDFS(start, end)
    if name == "MiniMax":
        agent = Agent()
        return agent.MiniMax()
    if name == "AlphaBeta":
        agent = Agent()
        return agent.AlphaBeta()
    return None

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


