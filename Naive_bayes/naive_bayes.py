

def pretempe(train, test, indexlist):
    for index in indexlist:
        temp = []
        for i in range(len(train)):
            temp.append(train[i][index])  #依次取出[2,3,4]列保存在临时list中，用来对2，3，4列进行归一化
        tmax, tmin = max(temp), min(temp)  #求出index列的最大值和最小值
        p1 = tmin + (tmax - tmin) / 3 #总共对该数据分成3段：0，1，2  p1为0，1的分割
        p2 = tmax - (tmax - tmin) / 3 #p1为1，2的分割
        #对训练集对应列的数据进行分段
        for i in range(len(train)):
            if temp[i] < p1:
                train[i][index] = 0
            elif temp[i] < p2:
                train[i][index] = 1
            else:
                train[i][index] = 2
        # 对测试集对应列的数据进行分段
        if test[index] < p1:
            test[index] = 0
        elif test[index] < p2:
            test[index] = 1
        else:
            test[index] = 2
    return train, test


if __name__ == '__main__':
    n = 9
    #训练集对应的label
    label = [0, 0, 0, 0, 1, 1, 1, 1, 1]
    #训练集5个特征
    train = [[0, 0, 30, 450, 7],
             [1, 1, 5, 500, 3],
             [1, 0, 10, 150, 1],
             [0, 1, 40, 300, 6],
             [1, 0, 20, 100, 10],
             [0, 1, 25, 180, 12],
             [0, 0, 32, 50, 11],
             [1, 0, 23, 120, 9],
             [0, 0, 27, 200, 8]]
    #目标：根据测试集给出的5个特征，判断测试集对应的label
    test = [0, 0, 40, 180, 8]

    train, test = pretempe(train, test, [2, 3, 4])
    f_num = len(test) #特征个数
    s_num = len(train)  #训练集样本个数
    prior = [[0] * 3 for _ in range(f_num)] #先验概率
    #求各个特征先验的概率
    for i in range(f_num):
        tmp = []
        for k in range(s_num):
            tmp.append(train[k][i])
        for j in range(3):
            prior[i][j] = tmp.count(j) / s_num
    condition = [[[0] * 3 for i in range(f_num)] for j in range(2)] #条件概率
    label0 = []
    label1 = []
    #根据正负样本求条件概率，后验概率
    for i in range(s_num):
        if label[i] == 0:
            label0.append(i)
        else:
            label1.append(i)
    for i in range(f_num):
        p1 = p2 = p3 = 0
        for j in label0:

            if train[j][i] == 0:
                p1 += 1
            elif train[j][i] == 1:
                p2 += 1
            else:
                p3 += 1
        condition[0][i][0] = p1 / len(label0)
        condition[0][i][1] = p2 / len(label0)
        condition[0][i][2] = p3 / len(label0)
        p1 = p2 = p3 = 0
        for j in label1:
            if train[j][i] == 0:
                p1 += 1
            elif train[j][i] == 1:
                p2 += 1
            else:
                p3 += 1
        condition[1][i][0] = p1 / len(label1)
        condition[1][i][1] = p2 / len(label1)
        condition[1][i][2] = p3 / len(label1)
    #求正负样本的先验概率
    l0, l1 = len(label0) / s_num, len(label1) / s_num
    #分别求归为正样本的概率res1，归为负样本的概率res2
    t1, t2, t3, t4, t5 = test
    res1 = l1 * condition[1][0][t1] * condition[1][1][t2] * condition[1][2][t3] * condition[1][3][t4] * condition[1][4][
        t5]  # 根据各特征独立，求同时发生的概率
    res1 = res1 / prior[0][t1] / prior[1][t2] / prior[2][t3] / prior[3][t4] / prior[4][t5] #求的该特征下为正样本的概率（条件概率）

    res0 = l0 * condition[0][0][t1] * condition[0][1][t2] * condition[0][2][t3] * condition[0][3][t4] * condition[0][4][
        t5]# 根据各特征独立，求同时发生的概率
    res0 = res0 / prior[0][t1] / prior[1][t2] / prior[2][t3] / prior[3][t4] / prior[4][t5]#求的该特征下为负样本的概率（条件概率）
    print (res1,res0)
    print(round(res1 / res0, 4)) #大于 1 归为正样本，小于 1 归为负样本