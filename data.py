import os
import numpy as np
import sunau
from python_speech_features import mfcc
import tensorflow as tf

class Data(object):
    
    def __init__(self):
        self.data_path = 'C:/Users/adpa.MOBICAPL/Desktop/genrex/genres/'
        self.data, self.labels = self.prepare_data(self.data_path)
        # print ("hahaha")
        self.iter = 0
        self.batch_size = 10
        
    def prepare_data(self,data_path):
        print ("LOLOLOLOOLOLOLOL")
        data_path = 'C:/Users/adpa.MOBICAPL/Desktop/genrex/genres/'
        genre_dict = {
            "blues" : 0,
            "classical" : 1,
            "country" : 2,
            "disco" : 3,
            "hiphop" : 4,
            "jazz" : 5,
            "metal" : 6,
            "pop" : 7,
            "reggae" : 8,
            "rock" : 9
        }
        
        XXX = np.zeros((0),np.float32)
        YYY = np.zeros((0),np.chararray)
        
        ### READ AUDIO DATA FRAMES AND CALC MFCC
        directory = data_path
        for subdir in next(os.walk(directory))[1]:
            subpath = directory + subdir + '/'
            for file in next(os.walk(subpath))[2]:
                
                file_path = subpath + file
                f=sunau.Au_read(file_path)
                audio_data = np.fromstring(f.readframes(10), dtype=np.float32)
                
                # FEATURES
                features = mfcc(audio_data)
                # print (features)
                XXX = np.append(XXX,features)
                # print (XXX)
                # return
                XXX = np.reshape(XXX,(-1,13))
                
                # LABELS
                label = file.split('.')[0]
                bit = genre_dict[label]
                label_score = np.zeros((10,1),np.float32)
                label_score[bit] = 1.0
                
                YYY = np.append(YYY,label_score)
                YYY = np.reshape(YYY,(-1,10))

                # print ("FILENAME: ",file,"   LABEL: ",label)
        print ('DATA LOADED')
        # print (XXX[1:8,:])
        return XXX,YYY
        
    def get_next_batch(self):
        batch_x = self.data[self.iter*self.batch_size:self.iter*self.batch_size + self.batch_size - 1]
        batch_y = self.labels[self.iter*self.batch_size:self.iter*self.batch_size + self.batch_size - 1]
        self.iter += 1
        
        return batch_x, batch_y

    def get_whole_data(self):
        return self.data, self.labels
        
    def reset_batch_counter(self):
        self.iter = 0
        return
    
        