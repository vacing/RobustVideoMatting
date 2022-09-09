# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
from sklearn import metrics
from . import util_file, util_container


def get_ordered_dir_list(dir_path):
    if not os.path.exists(dir_path):
        raise ValueError('input dirpath[%s] not exists' % dir_path)

    raw_file_all = os.listdir(dir_path)
    dir_list = [rf for rf in raw_file_all if os.path.isdir(os.path.join(dir_path, rf))]
    dir_list.sort()
    return dir_list


def get_ordered_file_list(dir_path):
    if not os.path.exists(dir_path):
        raise ValueError('input dirpath[%s] not exists' % dir_path)

    raw_file_all = os.listdir(dir_path)
    file_list = [rf for rf in raw_file_all if not os.path.isdir(os.path.join(dir_path, rf))]
    file_list.sort()
    return file_list


def get_ordered_dir_map(dir_path):
    dir_list = get_ordered_dir_list(dir_path)

    dir_lab_map = {}
    lab_dir_map = {}
    for i, d in enumerate(dir_list):
        dir_lab_map[d] = i
        lab_dir_map[i] = d

    return dir_lab_map, lab_dir_map, dir_list


def get_cat_files_labels(dir_path, suffixes, level=1, label_map=None):
    if label_map is None:
        label_map, _, _ = get_ordered_dir_map(dir_path)

    dir_list = get_ordered_dir_list(dir_path)
    images = []
    labels = []
    for d in dir_list:
        if d not in label_map.keys():
            raise ValueError('input label_map not contains [%s]' % d)
        d_full_path = os.path.join(dir_path, d)
        d_images = util_file.get_files_recursively(d_full_path, suffixes, level)
        d_labels = [label_map[d]] * len(d_images)

        logging.info('category [%s] has [%d] samples' %(d, len(d_images)))
        images.extend(d_images)
        labels.extend(d_labels)

    return images, labels, label_map


def performance_show(real_lab, pred_lab, labels=None, confusion_file=None):
    if labels is None:
        labels = list(set(real_lab))

    conf_mat = metrics.confusion_matrix(real_lab, pred_lab, labels=labels) 
    if confusion_file is not None:
        np.savetxt(confusion_file, conf_mat, fmt='%1.0f', delimiter=',')

    print(metrics.classification_report(real_lab, pred_lab))
        
    return conf_mat, labels


__confusion_file_cache = '__confusion_file_dump.csv'
def confusion_file_dump(sample_list, real_lab, pred_lab, dump_path, src_path=None, label2str_map=None):
    '''
    将分类分错的图片按照错误的类别进行保存
    :param sample_list: 图片名称或路径
    :param real_lab: 实际标签
    :param pred_lab: 预测标签
    :param dump_path: 错误图片保存根目录
    :param src_path: 如果图片只有名称，则此处填写图片路径前缀，否则忽略
    :param label2str_map: 标签值和有意义的标签名称之间的映射
    :return: 无
    '''
    if label2str_map is None:
        all_lab = list(set(set(pred_lab) + set(real_lab)))
        label2str_map = {}
        for l in all_lab:
            label2str_map[l] = l

    # collect wrong images
    for pl, rl, s in (zip(pred_lab, real_lab, sample_list)):
        if pl != rl:
            pl_str = label2str_map[pl]
            rl_str = label2str_map[rl]
            
            # source
            if src_path is None:
                sample_src_path = s 
            else:
                sample_src_path = os.path.join(src_path, s)

            # dstination
            sample_dst_path = os.path.join(dump_path, rl_str, pl_str)
            util_file.make_dir(sample_dst_path)
            # get file name
            _, name, suff = util_file.sep_path_segs(s)
            img_diff_dst = os.path.join(sample_dst_path, name + suff)

            # copy
            util_file.copy_file_noexcept(sample_src_path, img_diff_dst)

            # file name cache file
            file_list_path = os.path.join(dump_path, rl_str, __confusion_file_cache)
            # if file not exits, create; else append.
            with open(file_list_path, 'a+', encoding='utf-8') as flp:
                flp.write('%s\n' % (name+suff))

def checked_file_correct(checked_cat_name, checked_file_src_dir, checked_file_dst_dir, error_file_cache_dir,
                         error_file_list=None, ups_times_chk=1, ups_times_err=1, _copy_suff='_copy_'):
    """
    将检查过的错误图片修正到原始训练数据中，并从原始类别目录移除到备份文件夹。
    注意：一次操作一个类别。
    :param checked_cat_name: 类别名字
    :param checked_file_src_dir:
    :param checked_file_dst_dir:
    :param error_file_cache_dir:
    :param error_file_list: 预测认为错误的全部文件列表，包含确认无误后删除的
    :param ups_times_chk: 确认无误删除的样本上采样次数
    :param ups_times_err: upsample bad case
    :param _copy_suff: prevent duplicate copy, such as _copy -> _copy, _copy_copy, _copy_copy_copy
    :return: numbers of file has been corrected
    """
    assert os.path.exists((os.path.join(checked_file_dst_dir, checked_cat_name))), \
            'catgory[%s] not exist in [%s]' % (checked_cat_name, checked_file_dst_dir)
    assert ups_times_err > 0, 'upsample time error, ups_times_err=%d' % ups_times_err
    assert ups_times_chk > 0, 'upsample time error, ups_times_chk=%d' % ups_times_chk

    util_file.make_dir(error_file_cache_dir)
    copied_file_set = set()
    src_dir_list = get_ordered_dir_list(checked_file_src_dir)
    dst_dir_list = get_ordered_dir_list(checked_file_dst_dir)

    for src in src_dir_list:
        estr = 'class [%s] not in class list' % src
        assert src in dst_dir_list, estr

        src_full_path = os.path.join(checked_file_src_dir, src)
        file_list = get_ordered_file_list(src_full_path)

        for file in file_list:
            copied_file_set.add(file)   # 所有文件都是要删除的，不论是否拷贝
            if error_file_list is not None:
                if file in error_file_list:
                    error_file_list.remove(file)
                else:
                    logging.error('[%s] should in error file list, but not, ignored' % file)

            file_src = os.path.join(src_full_path, file)
            file_dst_dir = os.path.join(checked_file_dst_dir, src)
            cpy_ret = util_file.multi_copy_files(file_src, file_dst_dir, ups_times_err, 
                                                _copy_suff=_copy_suff, force_copy=True)
            if cpy_ret >= 1:
                logging.info('success: ' + repr([file_src, file_dst_dir]))

    # move error file to trash collection directory
    num_moved = len(copied_file_set)
    logging.warning(' %d files has been moved, now will be backup' % num_moved)
    logging.info(' begin to backup moved file')
    for copied in copied_file_set:
        find_copied_dir = os.path.join(checked_file_dst_dir, checked_cat_name)
        copied_file_path_arr = util_file.find_files_recursively(copied, find_copied_dir, level=3)
        if len(copied_file_path_arr) != 1:
            # same name but different is low probability, so remove 1 file, then you can check
            logging.error('backup: find [%d] files with name [%s]' %(len(copied_file_path_arr), copied))
            if len(copied_file_path_arr) <= 0:
                continue
            else:
                logging.error('backup: remove the first [%s]' % (copied_file_path_arr[0]))

        copied_file_path = copied_file_path_arr[0]
        copied_file_cache = os.path.join(error_file_cache_dir, copied)

        if not os.path.exists(copied_file_path):
            logging.error('delete file not exists [%s]' % copied_file_path)
            continue
        util_file.move_file_noexcept(copied_file_path, copied_file_cache)
    logging.info(' finish backup moved file')

    # enhance checked file
    num_enhanced = 0
    if error_file_list is not None:
        num_enhanced = len(error_file_list)
        if ups_times_chk > 1:
            logging.warning(' %d files will be enhanced %d times' % (num_enhanced, ups_times_chk))
            logging.info(' begin to enhance file')
            for enhanced in error_file_list:
                find_enhanced_dir = os.path.join(checked_file_dst_dir, checked_cat_name)
                enhanced_file_path_arr = util_file.find_files_recursively(enhanced, find_enhanced_dir, level=3)
                if len(enhanced_file_path_arr) != 1:
                    logging.error('enhance: find [%d] files with name [%s], ignored' %(len(enhanced_file_path_arr), enhanced))
                    continue

                enhanced_file_src = enhanced_file_path_arr[0]
                # copy file to the same directory with source
                checked_cat_dir, _, _ = util_file.sep_path_segs(enhanced_file_src)
                util_file.multi_copy_files(enhanced_file_src, checked_cat_dir, ups_times_chk, _copy_suff=_copy_suff)
            logging.info(' finish enhance file')
    else:
        logging.info('error file list is None')

    return num_moved, num_enhanced


def confusion_mat_detail_show(conf_mat, rows, row_names):
    # print(rows)
    # print(row_names)
    mat_detail = []
    for row in rows:
        print('-'*30 + str(row_names[row]) + '-'*30)
        conf_intrest = conf_mat[row]
        conf_res = {}
        for n, c in zip(row_names, conf_intrest):
            conf_res[n] = c 

        sortedX = util_container.sort_dict_by_value(conf_res)
        for cat, cnt in sortedX:
            print("%d,%s,%.3f" % (cnt, cat, cnt * 1./sum(conf_intrest)))

        mat_detail.append(sortedX)

    return sortedX
        

if __name__ == '__main__':
    dir_path = '/data/home/vacingfang/mydata/live_ad_data/dst_vac'

    dir_list = get_ordered_dir_list(dir_path)
    print(len(dir_list))
    print(dir_list)

    # r1, r2, _ = get_cat_files_labels(dir_path, util_file.img_suffixes, 3)
    # print(r1, r2)






