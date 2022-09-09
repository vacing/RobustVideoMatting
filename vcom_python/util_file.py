# -*- coding: utf-8 -*-
import os
import sys
import shutil
import glob
import logging
import random

def make_dir(p, del_old=False):
    if os.path.exists(p):  # 文件夹存在
        if del_old:
            shutil.rmtree(p)        # 删除旧文件夹
    else:
        try:
            os.mkdir(p)  # 创建文件夹
        except FileNotFoundError:
            os.makedirs(p, exist_ok=True)


def sep_path_segs(path):
    filepath, tempfilename= os.path.split(path)
    shortname, extension = os.path.splitext(tempfilename)
    return filepath, shortname, extension


def copy_file_noexcept(src, dst, force=True):
    ret = False
    try:
        # 文件已存在，且非强制拷贝
        if not force and os.path.exists(dst):
            pass
        elif os.path.exists(dst) and os.path.samefile(src, dst):
            logging.info('copy source and dst are the same file [%s]' % src)
        else:
            shutil.copyfile(src, dst)
            ret = True
    except:
        logging.error(sys.exc_info()[1])

    return ret


def move_file_noexcept(src, dst, force=True):
    ret = False
    try:
        # 文件已存在，且非强制移动
        if not force and os.path.exists(dst):
            pass
        elif os.path.exists(dst) and os.path.samefile(src, dst):
            logging.info('source and dst are the same file [%s]' % src)
        else:
            shutil.move(src, dst)
            ret = True
    except:
        logging.error(sys.exc_info()[1])

    return ret


def multi_copy_files(src_file, dst_dir, up_times, _copy_suff='_copy_', force_copy=False):
    if not 0 < up_times < 20:
        return -1
        logging.error('upsample time error, up_times=%d' % up_times)

    _, fn, suf = sep_path_segs(src_file)
    if not force_copy and fn.endswith(_copy_suff):
        logging.warning('[%s] is already a copy, ignored' % src_file)
        return 0

    real_cpy_cnt = 0
    for cnt in range(up_times):
        if cnt == 0:
            file_cnt = fn + suf
        else:
            # rand_num = random.randint(0, 1e4)
            rand_num = cnt
            file_cnt = fn + '_' + '{:04d}'.format(rand_num) + _copy_suff + suf

        checked_file_cpy = os.path.join(dst_dir, file_cnt)
        cpy_ret = copy_file_noexcept(src_file, checked_file_cpy, force=False)
        if cpy_ret:
            real_cpy_cnt += 1

    return real_cpy_cnt


img_suffixes = ['.jpg', '.jpeg', '.bmp', '.png']
def get_files_recursively(dir_path, suffixes, level=1):
    '''
    递归读取目录下指定后缀的文件。
    默认目录结构
    d1 -\
        |_x.img             # level 1
        |_ ...
        |_ d11 -\
            |_x.img         # level 2
            |_ ...
            |_ d111 -\
                |_x.img     # level 3
                |_ ...
    :param dir_path:
    :param suffixes:
    :param level:
    :return:
    '''
    if (isinstance(suffixes, str)):
        suffixes = [suffixes];

    if len(suffixes) == 0:
        raise ValueError('suffixes length is 0')

    # level = 1: dir_path/*.jpg
    # level = 3: dir_path/*/*/*.jpg
    if level < 1 or level > 4:
        raise ValueError('level must between [1, 3], but input=%d' % level)

    files = []
    for file_suffix in suffixes:
        # print(file_suffix)
        path_preffix = dir_path
        for i in range(level):
            path_preffix = os.path.join(path_preffix, '*')
            # print(path_preffix)
            files.extend(glob.glob(path_preffix + file_suffix))
    
    return files


def find_files_recursively(file_name, dir_path, level=1):
    '''
    递归一个文件夹及其子文件夹，搜索指定文件
    '''
    # level = 1: dir_path/*.jpg
    # level = 3: dir_path/*/*/*.jpg
    if level < 1 or level > 4:
        raise ValueError('level must between [1, 3], but input=%d' % level)

    _, fn, suff = sep_path_segs(file_name)
    file_name_raw = fn + suff

    files = []
    path_preffix = dir_path
    for i in range(level):
        # full file name, don't need to add * before it
        files.extend(glob.glob(path_preffix + file_name_raw))

        path_preffix = os.path.join(path_preffix, '*')
        # print(path_preffix)
    
    return files

if __name__ == '__main__':
    # path = '../'
    # # print(sep_path_segs('./a/b c/de f.txt'))
    # # print(sep_path_segs(r'.\a\b c\de f.txt'))
    # # make_dir(r'D:\DataTemp\game_type_new\p2p\csgo_dst\\')
    # files = find_files_recursively('util.py', path, level=2)
    # print(files)

    logging.getLogger().setLevel(logging.DEBUG)
    get_encd(r'D:/CloudDisk/内部项目/【负责】图片补全/1w名人/大国外交/大国外交.csv')


