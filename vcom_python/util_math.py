# encoding=utf-8




def hamming_dist(v1, v2):
    '''
    计算两个数组中不相同的元素个数
    :param v1:
    :param v2:
    :return:
    '''
    assert len(v1) == len(v2), 'input list must have same length, but has %d and %d' % (len(v1), len(v2))

    dist = 0
    for (e1, e2) in zip(v1, v2):
        if e1 != e2:
            dist += 1

    return dist


def online_mean(old_mean, x, x_ind):
    '''
    流式计算均值
    :param old_mean:
    :param x:
    :param x_ind:
    :return:
    '''
    assert x_ind > 0, 'index of x must be an positive integer, but input [%d]' % x_ind

    delta = x - old_mean
    new_mean = old_mean + delta * 1.0/ x_ind
    return new_mean


if __name__ == '__main__':
    x = list(range(0, 10, 1))
    print(x)
    m = 0
    for i, t in enumerate(x):
        m = online_mean(m, t, i+1)
        print(m)
