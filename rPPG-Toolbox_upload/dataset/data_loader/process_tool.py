
import scipy
import cv2
import dlib
import scipy.signal as signal
import numpy as np
import math
from scipy.signal import butter, filtfilt
import scipy.io as sio

def get_ROI(image):
    predictor = dlib.shape_predictor('/data/rPPG-Toolbox/dataset/data_loader/shape_predictor_68_face_landmarks.dat')
    # image = cv2.imread(pic1_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # get all faces : get_frontal_face_detector
    detector = dlib.get_frontal_face_detector()
    dets = detector(image_gray, 1)
    det = dets[0] #get the biggest face

    shape = predictor(image, det) #got feature points
    s1 = (shape.part(49).x-shape.part(4).x)*(shape.part(50).y-shape.part(1).y)
    s2 = (shape.part(24).x-shape.part(18).x)*(shape.part(24).y+shape.part(37).y-2* shape.part(20).y)/2
    s3 = (shape.part(13).x-shape.part(47).x)*(shape.part(13).y-shape.part(28).y)
    if s1==max(s1,s2,s3):
        ROI = [shape.part(4).x, shape.part(1).y,shape.part(50).x, shape.part(50).y]
    elif s2 == max(s1,s2,s3):
        ROI = [shape.part(18).x,2* shape.part(20).y-shape.part(37).y,shape.part(24).x, shape.part(24).y]
    else:
        ROI = [shape.part(47).x,shape.part(28).y,shape.part(13).x, shape.part(13).y]

    # cv2.rectangle(image, (shape.part(5).x, shape.part(1).y), (shape.part(49).x, shape.part(50).y), (0, 0, 255), 2)
    # cv2.rectangle(image, (shape.part(18).x,2* shape.part(20).y-shape.part(28).y), (shape.part(26).x, shape.part(25).y), (255, 0, 0), 2)
    # cv2.rectangle(image, (shape.part(47).x,shape.part(29).y), (shape.part(13).x, shape.part(13).y), (0, 255, 0), 2)
    # cv2.imshow("region1", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return ROI

def get_face(image):
    predictor = dlib.shape_predictor('/data/rPPG-Toolbox/dataset/data_loader/shape_predictor_68_face_landmarks.dat')
    # image = cv2.imread(pic1_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # get all faces : get_frontal_face_detector
    detector = dlib.get_frontal_face_detector()
    dets = detector(image_gray, 1)
    det = dets[0] #get the biggest face
    # print(det)
    
    result = [[det.left(), det.top(), det.right(), det.bottom()]]
    return result



def get_muti_ROI(image):
    predictor = dlib.shape_predictor(r'/data/rPPG-Toolbox/dataset/data_loader/shape_predictor_68_face_landmarks.dat')
    # image = cv2.imread(pic1_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # get all faces : get_frontal_face_detector
    detector = dlib.get_frontal_face_detector()
    dets = detector(image_gray, 1)
    det = dets[0] #get the biggest face

    shape = predictor(image, det) #got feature points
    # s1 = (shape.part(49).x-shape.part(4).x)*(shape.part(50).y-shape.part(1).y)
    # s2 = (shape.part(24).x-shape.part(18).x)*(shape.part(24).y+shape.part(37).y-2* shape.part(20).y)/2
    # s3 = (shape.part(13).x-shape.part(47).x)*(shape.part(13).y-shape.part(28).y)
    ROI = []
    ROI.append( [shape.part(4).x, shape.part(28).y,shape.part(49).x, shape.part(49).y])
    
    ROI.append( [shape.part(18).x,2* shape.part(20).y-shape.part(37).y,shape.part(24).x, shape.part(24).y])
    
    ROI.append( [shape.part(47).x,shape.part(28).y,shape.part(13).x, shape.part(13).y])

    # cv2.rectangle(image, (shape.part(5).x, shape.part(1).y), (shape.part(49).x, shape.part(50).y), (0, 0, 255), 2)
    # cv2.rectangle(image, (shape.part(18).x,2* shape.part(20).y-shape.part(28).y), (shape.part(26).x, shape.part(25).y), (255, 0, 0), 2)
    # cv2.rectangle(image, (shape.part(47).x,shape.part(29).y), (shape.part(13).x, shape.part(13).y), (0, 255, 0), 2)
    # cv2.imshow("region1", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print('check :',ROI)
    return ROI

def bvpsnr(BVP, FS, HR, PlotTF):
    '''
    :param BVP:
    :param FS:
    :param HR:
    :param PlotTF:
    :return:
    '''
    HR_F = HR / 60
    NyquistF = FS / 2
    FResBPM = 0.5
    N = (60 * 2 * NyquistF) / FResBPM

    ##Construct Periodogram
    F, Pxx = signal.periodogram(BVP, FS, nfft=N, window="hamming")
    GTMask1 = (F >= HR_F - 0.1) & (F <= HR_F + 0.1)
    GTMask2 = (F >= (HR_F * 2 - 0.2)) & (F <= (HR_F * 2 + 0.2))
    temp = GTMask1 | GTMask2
    SPower = np.sum(Pxx[temp])
    FMask2 = (F >= 0.5) & (F <= 4)
    AllPower = np.sum(Pxx[FMask2])
    SNR = 10 * math.log10(SPower / (AllPower - SPower))
    print("SignalNoiseRatio", SNR)
    return SNR


def calculate_green(result, frame):
    """
    输入视频帧,完成人脸检测,返回green均值与人脸
    :param frame:  视频帧
    :return result: 人脸位置参数
    :return green_channel_mean: Green均值
    """
    face = frame[max(result[1], 0):min(result[3], frame.shape[0]),
                 max(result[0], 0):min(result[2], frame.shape[1])]
    face_green = face[:, :, 1]
    green_channel_mean = face_green.mean(axis=(0, 1))
    
    return green_channel_mean


def face_detection(frequency, frames):
    results = None
    results_default =  [[59, 107, 106, 157], [71, 49, 161, 66], [151, 107, 179, 139]]
    green_channel_mean_list = []

    for i in range(len(frames)):
        img = np.array(frames[i] * 255, dtype='uint8')
        green_channel_sum = 0
        if i % frequency == 0 or results is None:
            try:
                results = get_muti_ROI(img)
                # results = get_face(img)
                if results == None:
                    results=results_default
            except:
                pass
        if results is not None:
            for result in results:
                green_channel_sum += calculate_green(result, img)
        green_channel_mean = green_channel_sum/3
        # print(green_channel_mean)
        green_channel_mean_list.append(green_channel_mean)

    return green_channel_mean_list


def detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = signal.shape[0]

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = scipy.sparse.spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal


def cross_corr(s1, s2):
    """
    原来用的互相关函数
    :param s1: 视频Greens
    :param s2: bvps
    :return:
    """
    c21 = scipy.signal.correlate(s2, s1, mode='full', method='auto')
    # c21 = np.correlate(s2,s1,mode='full')
    # print(c21)
    t21 = np.argmax(c21)
    len_s = len(s1)
    index = t21 - len_s+1
    delay = index
    # 若index>0，则说明s1信号领先s2信号index个距离
    # 若index<0，则说明s2信号领先s1信号index个距离
    if index > 0:
        tt1 = s2[index:]
        tt2 = s2[0:index]
        s2_0 = np.concatenate((tt1, tt2), axis=0)
    else:
        index = len_s + index
        tt1 = s2[0:index]
        tt2 = s2[index:]

        s2_0 = np.concatenate((tt2, tt1), axis=0)

    return s1, s2_0, -delay

def cross_back(delay,s2):
    len_s = len(s2)
    index = -delay
    # 若index>0，则说明s1信号领先s2信号index个距离
    # 若index<0，则说明s2信号领先s1信号index个距离
    if index > 0:
        tt1 = s2[index:]
        tt2 = s2[0:index]
        s2_0 = np.concatenate((tt1, tt2), axis=0)
    else:
        index = len_s + index
        tt1 = s2[0:index]
        tt2 = s2[index:]

        s2_0 = np.concatenate((tt2, tt1), axis=0)

    return s2_0

def double_cross_back(delay,frames,s2):
    len_s = len(s2)
    index = -delay
    # 若index>0，则说明s1信号领先s2信号index个距离
    # 若index<0，则说明s2信号领先s1信号index个距离
    h = 320
    w = 240
    cross_frames = np.zeros((abs(index), h, w, 3))
    if index > 0:
        tt1 = s2[index:]
        # tt2 = s2[0:index]
        tt2 = np.zeros(index)
        s2_0 = np.concatenate((tt1, tt2), axis=0)
        frames = np.concatenate((frames[0:(len_s-index)],cross_frames),axis=0)
    else:
        # index = len_s + index
        tt1 = s2[0:len_s+index]
        tt2 = np.zeros(abs(index))

        s2_0 = np.concatenate((tt2, tt1), axis=0)
        frames = np.concatenate((cross_frames,frames[0:(len_s+index)]),axis=0)

    return frames,s2_0

def test_mat(mat_file):
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

        frequency = 100
        if motion == 'Stationary' or motion == 'Stationary (after exercise)':
            frequency = 100
        elif motion == 'Rotation':
            frequency = 10
        elif motion == 'Talking':
            frequency = 50
        elif motion == 'Walking':
            frequency = 10
        frequency = 1700
        fps = 30.0
        length = 1700
        frames_RGB = frames[:, :, :, [2, 1, 0]]
        try:
            green_channel_mean = face_detection(frequency, frames_RGB[:length, :, :, :])
            green_channel_mean = np.array(green_channel_mean)
            pulse_pred = detrend(green_channel_mean, 100)
            [b_pulse, a_pulse] = butter(1, [0.9 / fps * 2, 2.5 / fps * 2], btype='bandpass')
            green_channel_mean_list = filtfilt(b_pulse, a_pulse, np.double(pulse_pred))

            bvp_to_compare = bvps[:length]

            green_channel_mean_list = green_channel_mean_list - np.min(green_channel_mean_list)
            green_channel_mean_list = green_channel_mean_list / np.max(green_channel_mean_list)
            bvp_to_compare = bvp_to_compare - np.min(bvp_to_compare)
            bvp_to_compare = bvp_to_compare / np.max(bvp_to_compare)

            _, bvp_to_compare, delay = cross_corr(green_channel_mean_list, bvp_to_compare)
            if abs(delay)>10:
                print('over delay : ',delay)
            else:
                frames,bvps = double_cross_back(delay,frames,bvps)
                
                if delay >0:
                    green_channel_mean_list[0:delay] = np.zeros(delay)
                    bvp_to_compare[0:delay ] = np.zeros(delay)
                elif delay <0:
                    green_channel_mean_list[delay:] = np.zeros(abs(delay))
                    bvp_to_compare[delay :] = np.zeros(abs(delay))
                _,_,delay_new = cross_corr(green_channel_mean_list, bvp_to_compare)
            
                # print(delay_new)
                # print(bvp_to_compare)
                # print(delay,frames.shape,len(bvps))
                # print(bvps,frames)

        except TypeError:
            print(f'{mat_file} has problem with facial_detection')
            print(frames.shape)

        return frames,bvps

# def test_pure(filepath):

if __name__ == "__main__":
    matpath = "/data/rPPG_dataset/mat_dataset/subject6/p6_8.mat"
    frames_after,bvps_after = test_mat(matpath)
    
    # _, bvps, delay = cross_corr(frames_after,bvps_after)
    # print(delay)
