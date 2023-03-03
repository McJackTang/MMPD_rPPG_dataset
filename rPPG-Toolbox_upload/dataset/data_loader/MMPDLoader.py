""" The dataloader for Tang datasets.

"""

import os
import cv2
import glob
import numpy as np
import re
import dlib

from scipy.__config__ import get_info
from .BaseLoader import BaseLoader
from multiprocessing import Pool, Process, Value, Array, Manager
from tqdm import tqdm
import pandas as pd
import skvideo.io
import scipy.io as sio
import sys
import itertools
from warnings import simplefilter
from dataset.data_loader.process_tool import *
from scipy.signal import butter, filtfilt

simplefilter(action='ignore', category=FutureWarning)

class MMPDLoader(BaseLoader):
    """The data loader for the MMPD dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an MMPD dataloader.
            Args:
                data_path(str): path of a folder which mainly stores raw video and bvp data.
                e.g. data_path should be "mat_dataset" for below dataset structure:
                -----------------
                     mat_dataset/
                     |   |-- subject1/
                     |       |-- p1_0.mat
                     |       |-- p1_1.mat
                     |       |...
                     |   |-- subject2/
                     |       |-- p2_0.mat
                     |       |-- p2_1.mat
                     |       |...
                     |...
                     |   |-- subjectn/
                     |       |-- pn_0.mat
                     |       |-- pn_1.mat
                     |       |...
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        self.info = config_data.INFO
        super().__init__(name, data_path, config_data)


    def get_data(self, data_path):
        """Returns data directories under the path(For MMPD dataset)."""
        print(data_path)
        data_dirs = glob.glob(data_path + os.sep + "subject*")
        if (data_dirs == []):
            raise ValueError(self.name + " dataset get data error!")
        dirs = [{"index": int(re.search('subject(\d+)', data_dir).group(1)), 
                 "path": data_dir} for data_dir in data_dirs]
        return dirs


    def get_data_subset(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits"""

        data_info = {}
        subj_list = []
        for data in data_dirs:
            index = data['index']
            subj_list.append(index)
            data_info[index] = data
        num_subjs = len(subj_list)
        subj_range = list(range(num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin*num_subjs), int(end*num_subjs)))
        print('used subject ids for split:', [subj_list[i] for i in subj_range])

        file_into_list = []
        for i in subj_range:
            subj_num = subj_list[i]
            file_into_list.append(data_info[subj_num])

        return file_into_list


    def preprocess_dataset_subprocess(self, data_dir, config_preprocess, filename):
        """Invoked by preprocess_dataset for multi_process."""
        num = data_dir['index']
        saved_filename = 'subject' + str(num)
        frames, bvps, light, motion, exercise, skin_color,gender, glasser, hair_cover, makeup \
            = self.read_mat(os.path.join(data_dir['path'], filename))

        skin_color = int(skin_color)
        information = [light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup]
        frames = (np.round(frames * 255)).astype(np.uint8)
        frames_clips, bvps_clips = self.preprocess(
            frames, bvps, config_preprocess, config_preprocess.LARGE_FACE_BOX)

        count, input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips,
                                                                        saved_filename, information)


    def multi_process_manager(self, data_dirs, config_preprocess):
        file_num = len(data_dirs)
        choose_range = choose_range = range(0, file_num)
        pbar = tqdm(list(choose_range))

        # shared data resource
        manager = Manager()
        file_list_dict = manager.dict()
        p_list = []
        running_num = 0

        for i in choose_range:
            data_dir = data_dirs[i]
            data_dirs_filename = os.listdir(data_dir['path'])
            file_list_dict[i] = [data_dir['index']]
            for filename in data_dirs_filename:
                process_flag = True
                while process_flag:  # ensure that every i creates a process
                    if running_num < 32:  # in case of too many processes
                        p = Process(target=self.preprocess_dataset_subprocess, \
                                    args=(data_dir, config_preprocess, filename))
                        p.start()
                        p_list.append(p)
                        running_num += 1
                        process_flag = False
                    for p_ in p_list:
                        if not p_.is_alive():
                            p_list.remove(p_)
                            p_.join()
                            running_num -= 1
                            pbar.update(1)
        # join all processes
        for p_ in p_list:
            p_.join()
            pbar.update(1)
        pbar.close()

        return file_list_dict


    @staticmethod
    def read_mat(mat_file):
        try:
            mat = sio.loadmat(mat_file)
        except:
            for _ in range(20):
                print(mat_file)
        frames = np.array(mat['video'])
        bvps = np.array(mat['GT_ppg']).T.reshape(-1)
        light = mat['light']
        motion = mat['motion']
        exercise = mat['exercise']
        skin_color = mat['skin_color']
        gender = mat['gender']
        glasser = mat['glasser']
        hair_cover = mat['hair_cover']
        makeup = mat['makeup']

        return frames, bvps, light, motion, exercise,skin_color,gender,glasser,hair_cover,makeup


    def get_information(self, information):
        """Encodes information from dataset."""
        light = ''
        if information[0] == 'LED-low':
            light = 1
        elif information[0] == 'LED-high':
            light = 2
        elif information[0] == 'Incandescent':
            light = 3
        elif information[0] == 'Nature':
            light = 4
        elif information[0] == 'ALL':
            light = '?'

        motion = ''
        if information[1] == 'Stationary' or information[1] == 'Stationary (after exercise)':
            motion = 1
        elif information[1] == 'Rotation':
            motion = 2
        elif information[1] == 'Talking':
            motion = 3
        elif information[1] == 'Walking':
            motion = 4
        elif information[1] == 'ALL':
            motion = '?'
        
        exercise = ''
        if information[2] == 'True' or information[2] == True:
            exercise = 1
        elif information[2] == 'False' or information[2] == False:
            exercise = 2
        elif information[2] == 'ALL':
            exercise = '?'

        skin_color = ''
        if information[3] == 0:
            skin_color = '?'
        else:
            skin_color = information[3]

        gender = ''
        if information[4] == 'male':
            gender = 1
        elif information[4] == 'female':
            gender = 2
        elif information[4] == 'ALL':
            gender = '?'

        glasser = ''
        if information[5] == 'True' or information[5] == True:
            glasser = 1
        elif information[5] == 'False' or information[5] == False:
            glasser = 2
        elif information[5] == 'ALL':
            glasser = '?'

        hair_cover = ''
        if information[6] == 'True' or information[6] == True:
            hair_cover = 1
        elif information[6] == 'False' or information[6] == False:
            hair_cover = 2
        elif information[6] == 'ALL':
            hair_cover = '?'
        
        makeup = ''
        if information[7] == 'True' or information[7] == True:
            makeup = 1
        elif information[7] == 'False' or information[7] == False:
            makeup = 2
        elif information[7] == 'ALL':
            makeup = '?'

        return light, motion ,exercise, skin_color, gender, glasser, hair_cover, makeup


    def save_multi_process(self, frames_clips, bvps_clips, filename, information):
        """Saves the preprocessing data."""
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path)
        count = 0
        input_path_name_list = []
        label_path_name_list = []
        light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup \
            = self.get_information(information)    
        # print(len(bvps_clips))
        for i in range(len(bvps_clips)):
            assert (len(self.inputs) == len(self.labels))
            input_path_name = self.cached_path + os.sep + \
                "{0}_L{1}_MO{2}_E{3}_S{4}_GE{5}_GL{6}_H{7}_MA{8}_input{9}.npy"\
                    .format(filename, light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup, str(count))
            label_path_name = self.cached_path + os.sep + \
                "{0}_L{1}_MO{2}_E{3}_S{4}_GE{5}_GL{6}_H{7}_MA{8}_label{9}.npy"\
                    .format(filename, light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup, str(count))
            input_path_name_list.append(input_path_name)
            label_path_name_list.append(label_path_name)
            np.save(input_path_name, frames_clips[i])
            np.save(label_path_name, bvps_clips[i])
            count += 1
        return count, input_path_name_list, label_path_name_list

    
    def build_file_list_retroactive(self, data_dirs, begin, end):
        # get data split
        data_dirs = self.get_data_subset(data_dirs, begin, end)

        # generate a list of unique raw-data file names
        file_list = []
        for i in range(len(data_dirs)):
            file_list.append(data_dirs[i]['index'])
        file_list = list(set(file_list)) # ensure all indexes are unique

        if not file_list:
            raise ValueError(self.name, 'File list empty. Check preprocessed data folder exists and is not empty.')

        file_list_df = pd.DataFrame(file_list, columns = ['input_files'])   
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)


    def load(self):
        """Loads the preprocessing data."""
        file_list_path = self.file_list_path  # get list of files in

         # TO DO: Insert functionality to generate file list if it does not already exist

        file_list_df = pd.read_csv(file_list_path)
        subject_num = file_list_df['input_files'].tolist()
        # print(subject_num)    
        inputs = self.get_inputs(subject_num)
        if not inputs:
            raise ValueError(self.name+' dataset loading data error!')
        inputs = sorted(inputs) # sort input file name list
        # print(inputs)
        labels = [input.replace("input", "label") for input in inputs]
        assert (len(inputs) == len(labels))
        self.inputs = inputs
        self.labels = labels
        self.len = len(inputs)
        # for each in inputs:
        #     print(each[-40:])
        print("loaded data len:",self.len)
        

    def get_inputs(self, subject_num):
        information = list(itertools.product(self.info.LIGHT, self.info.MOTION, self.info.EXERCISE, self.info.SKIN_COLOR, \
            self.info.GENDER, self.info.GLASSER, self.info.HAIR_COVER, self.info.MAKEUP, subject_num))
        information = list(itertools.product(self.info.LIGHT, self.info.MOTION, self.info.EXERCISE, self.info.SKIN_COLOR, subject_num))
        inputs = list()
        for each_info in tqdm(information):
            light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup \
                = self.get_information(each_info[:-1]+('male', 'False', 'False', 'False'))
            num = each_info[-1]
            inputs += glob.glob(os.path.join(self.cached_path, f"subject{num}_L{light}_MO{motion}_E{exercise}_S{skin_color}*_input*.npy"))
        return inputs

    