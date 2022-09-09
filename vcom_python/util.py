# encoding=utf-8
import logging
import sys
import os
import pandas


def pretty_args_str(args):
    args_str = ''
    for arg in vars(args):
        args_str += '\n\t%s:\t[%s]' % (arg, repr(getattr(args, arg)))

    return args_str

def init_logger(*, fn=None):
    from imp import reload
    reload(logging)

    logging_params = { 
        'level': logging.INFO,
        'format': '%(asctime)s__[%(levelname)s, %(module)s.%(funcName)s](%(name)s)__[L%(lineno)d] %(message)s',
    }   

    if fn is not None:
        logging_params['filename'] = fn

    logging.basicConfig(**logging_params)
    logging.debug('init basic configure of logging success')


def is_in_list_fgen(exclude_list=None, exclude_file=None, exclude_dir=None):
    '''
    生成名称列表，并返回判断指定名称是否在该列表中的函数
    :param exclude_list:
    :param exclude_file:
    :param exclude_dir:
    :return:
    '''
    if exclude_list is None:
        exclude_list = []

    if exclude_file is not None:
        assert os.path.exists(exclude_file), "file not eixst[%s]" % exclude_file
        assert os.path.isfile(exclude_file), "not file[%s]" % exclude_file
        lines = pandas.read_csv(exclude_file, header=None, delimiter=',', encoding='utf8')
        for l in lines:
            if len(l) < 1:
                continue
                exclude_list.append(l[0])

    if exclude_dir is not None:
        assert os.path.exists(exclude_dir), "file not eixst[%s]" % exclude_dir
        assert os.path.isdir(exclude_dir), "not directory[%s]" % exclude_dir

        files = os.listdir(exclude_dir)
        for f in files:
            exclude_list.append(f)

    def is_in_except_list(name):
        return name in exclude_list

    return is_in_except_list, exclude_list


if __name__ == '__main__':
    init_logger()
    logging.error(2)


