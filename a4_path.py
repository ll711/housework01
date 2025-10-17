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
from a1_state import State
from a3_path import Agent
def play(state, agentA, agentB):
    # main game loop
    #其中游戏开始的时候用户应该选择是人机对战还是机器对战
    #用户界面中只能展示网格
    #游戏开始后用户可以通过手动的方式输入m行和n列并且展示一个网格在每个网格中输入数据（详细见a1_state.py）
    #然后游戏开始并且调用a3的agent类并在过程中选择最佳的模型？
    #当用户或者机器没下一步之后进行胜负判断
    #最后展示胜负结果
    #其中在游戏过程中通过获取用户的点击进行改变在某个范围内的图形的变化
    # 如果发现链表中的图形边界发生变化应该调整该图像在链表中被记录的样子（即max,min的边界调整以及list存储的图形的边界即max应该减一）
    pass
def tester():
    # 测试以上所有函数
    pass
if __name__ == "__main__":
    tester()


