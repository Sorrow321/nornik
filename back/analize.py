
from image_preprocessing import get_frames, preprocess
from stream_processing import calc_stream_params
import numpy as np
import json
import pathlib

#Основной скрипт, осуществляющий видеоаналитику: как статичных кадров, так и сравнение кадров для измерения, например, скорости потока

#отключение warnings numpy
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def data_from_vid(vid_path, temp_max = None, aim_file = 'dataanswer.json'):
    '''
    vid_path - путь к видео файлу
    temp_max - сколько групп кадров записать, если None - то все
    aim_file - файл для записи статистики

    скрипт осуществляет запись кадров группами по 5 единиц, усредняя полученные значения
    '''
    c_temp = 0
    namings = ['objects_count','areas_mean', 'areas_var', 'colors_mean', 'colors_var', 
                'mean_distance', 'var_distance', 'sum_vec_destination', 'sum_vec_length', 'sum_vec_angle']
    count_every = 5
    mean_results = []
    results = []
    frames = get_frames(vid_path, do_save=False)

    previous_bin = None
    previous_centers = None

    dinamic_succes = False

    for i,frame in enumerate(frames):
        try:
            static = preprocess(input_numpy_img = frame)
            static_success = True 
        except:
            static_success = False 
            print('1 frame lost - s')
        if i > 0 and static_success:
            try:
                dinamic = calc_stream_params(previous_bin, static['binarizations']['S'], previous_centers, auto_best = False) 
                dinamic_succes = True
            except:
                dinamic_succes = False
                print('1 frame lost - d')
            if dinamic_succes:
                results.append([np.array(len(static['centers'])), static['areas']['areas_mean'], 
                                static['areas']['areas_var'], static['colors']['colors_mean'], 
                                static['colors']['colors_var'], dinamic['mean_distance'], dinamic['var_distance'], 
                                dinamic['sum_vec']['destination'], dinamic['sum_vec']['length'], dinamic['sum_vec']['angle']])
                if len(results)>=count_every:
                    mean_values = np.mean(results[count_every*(-1):], axis = 0)
                    temp_dict = {}
                    for obj, name in zip(mean_values,namings):
                        temp_dict[name] = obj.tolist()

                    with open(aim_file, 'a') as outfile:
                        outfile.write(','+json.dumps(temp_dict))
                    
                if temp_max is not None:
                    if c_temp>=temp_max:
                        break
                c_temp+=1
                print(c_temp)
        if static_success:
            previous_bin = static['binarizations']['S']
            previous_centers = static['centers']

if __name__=='__main__':
    vid_path = '/Nornikel/dataset1-1/F1_1_4_2.ts'
    data_from_vid(vid_path, temp_max = None)