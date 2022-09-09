# -*- coding: utf-8 -*-
import operator
import collections

def sort_dict_by_value(d, big2small=True):
    sortedX = sorted(d.items(), key=operator.itemgetter(1), reverse=big2small)
    return sortedX


def flatten(x):
    '''
    将任意高维度向量展成一维数组
    :param x:
    :return:
    '''
    result = []
    for el in x:
        if isinstance(el, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result
