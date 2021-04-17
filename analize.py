#relative import?
from image_preprocessing import get_frames, preprocess
import numpy as np
import json

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

#history data
absolute_min = 0
absolute_max = 1


vid_path = '/Users/romansmirnov/eclipse-workspace/Hackatons/Nornikel/dataset1-1/F1_1_1_1.ts'

def data_from_vid(vid_path, temp_max = None, aim_file = 'dataanswer.json'):
    c_temp = 0
    namings = ['objects_count','areas_mean', 'areas_var', 'colors_mean', 'colors_var']
    count_every = 5
    mean_results = []
    results = []
    frames = get_frames(vid_path, do_save=False)
    for frame in frames:
        data = preprocess(input_numpy_img = frame)
        results.append([np.array(len(data['centers'])),data['areas']['areas_mean'], data['areas']['areas_var'], data['colors']['colors_mean'], data['colors']['colors_var']])
        if len(results)>=count_every:
            mean_values = np.mean(results[count_every*(-1):], axis = 0)
            temp_dict = {}
            for obj, name in zip(mean_values,namings):
                temp_dict[name] = obj.tolist()
            try:
                with open(aim_file, 'a') as outfile:
                    json.dump(temp_dict, outfile)
            except:
                with open(aim_file, 'w') as outfile:
                    json.dump(temp_dict, outfile)
        if temp_max is not None:
            if c_temp>=temp_max:
                break
        c_temp+=1

data_from_vid(vid_path, temp_max = 30)