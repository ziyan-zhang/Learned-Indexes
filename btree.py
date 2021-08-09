# BTree Index with Python
# todo 文章baseline B树中使用的是stx::btree, 并加入了catchline优化
import pandas as pd

# Node in BTree
class BTreeNode:
    def __init__(self, degree=3, number_of_keys=0, is_leaf=True, items=None, children=None,
                 index=None):
        # 度数：在树中，每个节点的子节点（子树）的个数就称为该节点的度（degree）。
        # 阶数：（Order）阶定义为一个节点的子节点数目的最大值。（自带最大值属性）
        self.isLeaf = is_leaf
        self.numberOfKeys = number_of_keys
        self.index = index
        if items is not None:
            self.items = items
        else:
            self.items = [None] * (degree * 2 - 1)
        if children is not None:
            self.children = children
        else:
            self.children = [None] * degree * 2

    def set_index(self, index):
        self.index = index

    def get_index(self):
        return self.index

    def search(self, b_tree, an_item):
        """
        Args:
            b_tree: 要被查找的树
            an_item:  要被查找的关键字
            self: 当前的节点

        Returns:
            如果关键字在该节点存在, 返回{'found': True, 'fileIndex': 关键字对应的查找节点(文件)索引, 'nodeIndex': 关键字在当前查找节点上的索引};
            如果关键字在该节点不存在, 往下查找, 返回{'found': False, 'fileIndex': self.index查找到的最后一个节点的索引, 'nodeIndex': i - 1, 也即最后一个值小于item的关键字索引}
        """
        i = 0
        while i < self.numberOfKeys and an_item > self.items[i]:
            i += 1
        if i < self.numberOfKeys and an_item == self.items[i]:
            return {'found': True, 'fileIndex': self.index, 'nodeIndex': i}
        if self.isLeaf:
            return {'found': False, 'fileIndex': self.index, 'nodeIndex': i - 1}
        else:
            return b_tree.get_node(self.children[i]).search(b_tree, an_item)

# BTree Class
class BTree:
    def __init__(self, degree=2, nodes=None, root_index=1, free_index=2):
        if nodes is None:
            nodes = {}
        self.degree = degree
        # 度数：在树中，每个节点的子节点（子树）的个数就称为该节点的度（degree）。
        # 阶数：（Order）阶定义为一个节点的子节点数目的最大值。（自带最大值属性）
        if len(nodes) == 0:
            self.rootNode = BTreeNode(degree)
            self.nodes = {}
            self.rootNode.set_index(root_index)
            self.write_at(1, self.rootNode)
        else:
            self.nodes = nodes
            self.rootNode = self.nodes[root_index]
        self.rootIndex = root_index
        self.freeIndex = free_index

    def build(self, keys, values):
        if len(keys) != len(values):
            return  # 无效创建
        for ind in range(len(keys)):
            self.insert(Item(keys[ind], values[ind]))  # 插入item

    def search(self, an_item):
        return self.rootNode.search(self, an_item)  # 从根节点开始搜索

    def predict(self, key):
        search_result = self.search(Item(key, 0))
        a_node = self.nodes[search_result['fileIndex']]
        if a_node.items[search_result['nodeIndex']] is None:
            return -1
        return a_node.items[search_result['nodeIndex']].v

    def split_child(self, p_node, i, c_node):  # 将c_node分裂, 中间关键字放在p_node的位置i上,
        # 第1/2步, 依据要分裂的, 被装满的, c_node建立新节点new_node.
        new_node = self.get_free_node()

        # 新节点的idLeaf继承自c_node, 关键字数是self.degree-1.
        new_node.isLeaf = c_node.isLeaf  # 新节点是不是叶子结点也从已经爆的旧当前孩子节点继承
        new_node.numberOfKeys = self.degree - 1  # 关键字数4等于度数5减一

        # 关键字: 将child后M-1个key(共2M-1个key, child满载)拷贝给新节点
        # 位于正中间的(第M个)key的下标为M-1, 因此拷贝从M下标开始, 到2M-2下标(第M-1, 也即末个key)结束, 共M-1个key
        for k in range(0, self.degree - 1):
            new_node.items[k] = c_node.items[k + self.degree]  # 新节点是子节点c_node右半部分

        # 孩子: 如果c_node不是叶子结点, 还要将c_node的后M个(共2M个)孩子拷贝给新节点
        # 位于正中间的(第M个)key的下标为M-1, 因此拷贝从他右边的 M child下标开始, 到2M-1下标(第2M个, 也即末个, key)结束, 共M个child
        if c_node.isLeaf is False:  # 如果子节点不为空, 新节点还要继承子节点的右M-1个孩子
            for k in range(0, self.degree):
                new_node.children[k] = c_node.children[k + self.degree]

        # 原来被装满的子节点c_node中间及右M-1个部分关闭访问, 只留前M-1个. 面向磁盘的数据结构, 管写不管改.
        c_node.numberOfKeys = self.degree - 1

        # 第2/2步, p_node的关键字和孩子节点分别右移, 并装入新的.
        # 孩子右移位; 插入新的孩子
        for k in range(p_node.numberOfKeys, i, -1):  # 这两个for注意第二个值并取不到
            p_node.children[k+1] = p_node.children[k]
        p_node.children[i+1] = new_node.get_index()  # 新key的索引i, child的是i+1

        # 关键字右移位; 插入新的关键字; 关键字个数加一
        for k in range(p_node.numberOfKeys-1, i-1, -1):  # 这两个for注意第二个值并取不到
            p_node.items[k+1] = p_node.items[k]
        p_node.items[i] = c_node.items[self.degree-1]
        p_node.numberOfKeys += 1  # 并且父节点关键字计数加一

    def insert(self, an_item):  # 这个应该是执行在节点对应的子树尺度上的, 而不是整棵树上的
        search_result = self.search(an_item)
        if search_result['found']:
            return None  # 插入的是原有的值, 无效插入
        r = self.rootNode  # rootNode是根节点
        if r.numberOfKeys == 2 * self.degree - 1:  # 如果在根节点已经满了的情况下插入新的关键字,
            # 要对根节点进行分裂. 从根节点往上开节点
            s = self.get_free_node()  # 给self新加了一个节点, 扩充后的树句柄给s
            self.set_root_node(s)
            s.isLeaf = False
            s.numberOfKeys = 0
            s.children[0] = r.get_index()
            self.split_child(s, 0, r)  # 将r作为子树放在s的位置0
            self.insert_not_full(s, an_item)
        else:  # 插入后的关键字个数小于m, 可直接插入
            self.insert_not_full(r, an_item)

    def insert_not_full(self, inNode, anItem):  # insertNode, 表示在哪个节点插入
        # 这个人对下标的引用明显没有c++那个实现引用的好, 不够straight forward.
        """inNode索引到的, 要插入在这里的节点"""
        i = inNode.numberOfKeys - 1  # 最后一个关键字(item)的索引. 也即i指向了最大的那个key
        if inNode.isLeaf:  # 新项是叶子节点好说, 直接把更大的项右移一位, 然后把新项放入对应位置
            while i >= 0 and anItem < inNode.items[i]:  # 不大于等于0就不判定后面的大小关系了
                inNode.items[i + 1] = inNode.items[i]
                i -= 1
            inNode.items[i + 1] = anItem
            inNode.numberOfKeys += 1
        else:  # 不是根节点的话, 看他左子树满不满, 满了的话, 分裂, 如果新入的关键字比要插入的小, 继续往右找.
            # 注意一定要插入到最底层的某个非叶节点, 因为只有到最底层才能比较完, 从而最终确定插入哪个位置
            while i >= 0 and anItem < inNode.items[i]:
                i -= 1
            i += 1  # 这个时候anItem是大于等于inNode.items[i]的第一个索引
            if self.get_node(inNode.children[i]).numberOfKeys == 2 * self.degree - 1:
                # 当前节点满了的话,当其父节点有要插入的值时, 将当前子节点分裂.
                # 也就是说, 父节点没有值要插入时, 不用管当前节点是否满, 减少不必要的操作
                self.split_child(inNode, i, self.get_node(inNode.children[i]))
                # 将self.get_node(inNode.children[i])分裂, 中间关键字放在inNode的位置i上,
                # 右半部分作为新节点放在i孩子位置, 然后将self.get_node(inNode.children[i])右半部分关闭访问
                if anItem > inNode.items[i]:
                    i += 1
            self.insert_not_full(self.get_node(inNode.children[i]), anItem)

    def delete(self, an_item):
        an_item = Item(an_item, 0)
        search_result = self.search(an_item)
        if search_result['found'] is False:
            return None
        r = self.rootNode
        self.delete_in_node(r, an_item, search_result)

    def merge_child(self, root, pos, y, z):  # 把z合并到y上
        # 将z中后半部分的节点拷贝到y的后半部分, root.key[pos]下降为y中间节点
        y.numberOfKeys = 2*self.degree-1
        for i in range(self.degree, 2*self.degree-1, 1):
            y.items[i] = z.items[i-self.degree]
        y.items[self.degree-1] = root.items[pos]

        # 如果z不是叶子节点, 需要拷贝z的孩子
        if (z.isLeaf == False):
            for i in range(self.degree, 2*self.degree, 1):
                y.children[i] = z.children[i-self.degree]

        #由于root.items[pos]下降到y中, 更新root的key和pointer
        for j in range(pos+1, root.numberOfKeys, 1):
            root.items[j-1] = root.items[j]
            root.children[j] = root.children[j+1]

        # 两次整理内存, 第1次包括判断删除后后不会引起回溯合并
        root.numberOfKeys -= 1
        if root.numberOfKeys == 0:
            del self.nodes[root.index]
            self.set_root_node(y)

        del self.nodes[z.get_index()]

    def delete_in_node(self, a_node, an_item, search_result):  # todo:目前只是通过实验确认了插入似乎可以正常运行, 删除查找等还没看.
        """从a_node中删除元素an_item, 参考信息为search_result"""
        # 第一种情况: 如果在当前节点a_node找到了要删除的an_item
        if a_node.index == search_result['fileIndex']:  # a_node.index != search_result['fileIndex']意思是要删除的节点不是当前节点
            i = search_result['nodeIndex']  # nodeIndex是在Node里面的索引
            # 如果是叶子结点, 直接通过覆写和改numberOfKeys删除
            if a_node.isLeaf:
                while i < a_node.numberOfKeys - 1:  # 如果要删除的元素位于叶子结点, 则索引之后的元素通通前移一位
                    a_node.items[i] = a_node.items[i + 1]
                    i += 1
                a_node.numberOfKeys -= 1  # 该节点关键字的数量减一
            else:  # 要删除的元素不位于叶子结点
                left = self.get_node(a_node.children[i])
                right = self.get_node(a_node.children[i + 1])
                if left.numberOfKeys >= self.degree:  # 这两种都是左右兄弟够借的情况. 由于面向对象的思想, 在另一个类函数中展开
                    a_node.items[i] = self.get_right_most(left)
                elif right.numberOfKeys >= self.degree:
                    a_node.items[i] = self.get_right_most(right)
                else:  # 左右兄弟都不够借
                    k = left.numberOfKeys
                    left.items[left.numberOfKeys] = an_item
                    left.numberOfKeys += 1
                    for j in range(0, right.numberOfKeys):
                        left.items[left.numberOfKeys] = right.items[j]
                        left.numberOfKeys += 1
                    del self.nodes[right.get_index()]
                    for j in range(i, a_node.numberOfKeys - 1):
                        a_node.items[j] = a_node.items[j + 1]
                        a_node.children[j + 1] = a_node.children[j + 2]
                    # 这里, 因为删除没有从root开始查起, 所以可能导致a_node是根节点, 下沉完一个元素之后变成了空
                    # 也即,  第一层的查找, root节点一个关键字, 两个孩子都是M-1个关键字
                    # 因为删除是从上到下走的, 这里没有对第一次删除(在root节点时)进行一次讨论
                    a_node.numberOfKeys -= 1
                    if a_node.numberOfKeys == 0:
                        del self.nodes[a_node.get_index()]
                        self.set_root_node(left)
                    self.delete_in_node(left, an_item, {'found': True, 'fileIndex': left.index, 'nodeIndex': k})
        # 第二种情况: 如果在当前节点a_node未找到要删除的an_item
        else:  # 也即a_node.index != search_result['fileIndex']
            i = 0
            # todo: 这里重复查找了吧. if那里可以确定一个i, 通过比较关键字大小的话.
            # 如果当前节点没有, 则从孩子中一个一个找
            while i < a_node.numberOfKeys and self.get_node(a_node.children[i]).search(self, an_item)['found'] is False:
                # 这里再次注释一下Node的serach功能
                """
                Args:
                    b_tree: 要被查找的树
                    an_item:  要被查找的关键字
                    self: 当前的节点
                Returns:
                    如果关键字在该节点存在, 返回{'found': True, 'fileIndex': 关键字对应的查找节点(文件)索引, 'nodeIndex': 关键字在当前查找节点上的索引};
                    如果关键字在该节点不存在, 往下查找, 返回
                    {'found': False, 'fileIndex': self.index查找到的最后一个节点的索引, 
                    'nodeIndex': i - 1, 也即最后一个值小于item的关键字索引}  
                """
                i += 1
            # 这时的i: 如果b树中包含要找的关键字an_item, 那么一定是能够找到的. 找到的时候an_item位于a_node的第i个孩子节点
            c_node = self.get_node(a_node.children[i])  # 当前孩子节点i包含要删除的元素. 注意这个时候的c_node可能是叶子节点了已经
            # 根据节点位置定义左右兄弟, 注意左右兄弟并不总是存在
            if i < a_node.numberOfKeys:
                zNode = self.get_node(a_node.children[i+1])  # 从这个get_node可以看出, 这个版本的children是一个数, 而cpp版本的是一个指针
            if i > 0:
                pNode = self.get_node(a_node.children[i-1])

            # 如果c_node的关键字数: =M-1, 借点后迭代删除; >M-1, 直接迭代删除; <M-1, 必是叶节点, 直接迭代删除. 后两者统一, 只不过叶节点首先判断, 运算更快
            if c_node.numberOfKeys == self.degree-1:
                if (i>0 and pNode.numberOfKeys > self.degree-1):
                    k = c_node.numberOfKeys
                    while k > 0:
                        c_node.items[k] = c_node.items[k - 1]
                        c_node.children[k + 1] = c_node.children[k]
                        k -= 1
                    c_node.children[1] = c_node.children[0]

                    c_node.items[0] = a_node.items[i - 1]
                    c_node.children[0] = pNode.children[pNode.numberOfKeys]
                    c_node.numberOfKeys += 1
                    a_node.items[i - 1] = pNode.items[pNode.numberOfKeys - 1]
                    pNode.numberOfKeys -= 1
                elif (i<a_node.numberOfKeys and zNode.numberOfKeys>self.degree-1):
                    zNode = self.get_node(a_node.children[j])
                    c_node.items[c_node.numberOfKeys] = a_node.items[i]
                    c_node.children[c_node.numberOfKeys + 1] = zNode.children[0]
                    a_node.items[i] = zNode.items[0]
                    for k in range(0, zNode.numberOfKeys):
                        zNode.items[k] = zNode.items[k + 1]
                        zNode.children[k] = zNode.children[k + 1]
                    zNode.children[k] = zNode.children[k + 1]
                    zNode.numberOfKeys -= 1
                elif (i>0):  # 左右兄弟都不够借, 只能合并. 左右兄弟哪个有跟哪个合并, 先看左兄弟.
                    # 现在左兄弟存在, 先将当前节点合并到左兄弟上
                    self.merge_child(a_node, i-1, pNode, c_node)
                    c_node = pNode
                else:  # 没有左兄弟, 只能合并到右兄弟上
                    self.merge_child(a_node, i, c_node, zNode)
                self.delete_in_node(c_node, an_item, c_node.search(self, an_item))
            else:
                # 这里是上面if不满足, 也即当前节点关键字很多, 可以直接删除; 或者开始关键字不够, 但是已经借好了的情况
                self.delete_in_node(c_node, an_item, c_node.search(self, an_item))

    def get_right_most(self, aNode):
        if aNode.children[aNode.numberOfKeys] is None:
            upItem = aNode.items[aNode.numberOfKeys - 1]
            self.delete_in_node(aNode, upItem,
                                {'found': True, 'fileIndex': aNode.index, 'nodeIndex': aNode.numberOfKeys - 1})
            return upItem
        else:
            return self.get_right_most(self.get_node(aNode.children[aNode.numberOfKeys]))

    def set_root_node(self, r):
        self.rootNode = r
        self.rootIndex = self.rootNode.get_index()

    def get_node(self, index):
        return self.nodes[index]

    def get_free_node(self):
        new_node = BTreeNode(self.degree)
        index = self.get_free_index()
        new_node.set_index(index)
        self.write_at(index, new_node)
        return new_node

    def get_free_index(self):  # 使得self.free_index加一, 返回当前free_index
        self.freeIndex += 1
        return self.freeIndex - 1

    def write_at(self, index, a_node):
        self.nodes[index] = a_node

# Value in Node
class Item():
    def __init__(self, k, v):
        self.k = k
        self.v = v

    def __gt__(self, other):
        if self.k > other.k:
            return True
        else:
            return False

    def __ge__(self, other):
        if self.k >= other.k:
            return True
        else:
            return False

    def __eq__(self, other):
        if self.k == other.k:
            return True
        else:
            return False

    def __le__(self, other):
        if self.k <= other.k:
            return True
        else:
            return False

    def __lt__(self, other):
        if self.k < other.k:
            return True
        else:
            return False

# For Test
def b_tree_main():
    # path = "last_data.csv"
    path = "data\\exponential_s.csv"

    data = pd.read_csv(path)
    b = BTree(3)
    for i in range(data.shape[0]):
        b.insert(Item(data.iloc[i, 0], data.iloc[i, 1]))

    pos = b.predict(30310)
    print(pos)


def b_tree_main2():
    b = BTree(2)  # 这里的degree指的是 math.ceil(M/2) - 1
    b.insert(Item(10, 10))
    b.insert(Item(20, 20))
    b.insert(Item(30, 30))
    b.insert(Item(40, 40))
    b.insert(Item(50, 50))
    b.insert(Item(25, 25))
    b.insert(Item(35, 35))
    b.insert(Item(36, 36))
    b.insert(Item(60, 60))
    b.insert(Item(70, 70))

    b.delete(10)

    print()

    # pos = b.predict(8)
    # print(pos)

if __name__ == '__main__':
    b_tree_main2()
