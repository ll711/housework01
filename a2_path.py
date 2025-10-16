"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a State class for Task 1

@author: B20 (100528137 and 100531086)
@date: 11/10/2025

"""
#在这个搜索文件中应该包含BFS,DFS,A*等算法
# 在其中这个文件中的所有数据应该先预处理 寻找出变化的大体范围在采用对应的算法
from a1_state import State
def path_BFS(start,end):
    # 这个项目的BFS通过list将每一个图形放到每一个节点上
    #其中list的max,min是标注每个图形的大致范围
    #在这个算法中应该填入每个链表中安全路径的变化即list1,list2,list3中的第一个并以此类推
    return None
def path_DFS(start, end):
    # 这个项目的BFS通过list将每一个图形放到每一个节点上
    # 其中list的max,min是标注每个图形的大致范围
    # 在这个算法中应该填入一个链表中安全路径的变化在搜索下一个list中的安全路径变化
    return None
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