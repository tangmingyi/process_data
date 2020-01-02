import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

class ConvImage():
    def __init__(self,**kwargs):
        self.image_root_path = kwargs["image_root_path"]
        self.conv_image_save_path = kwargs["conv_image_save_path"]
        self.data_formate = kwargs["data_formate"]
        self.image_size = kwargs["image_size"]

    def _check_path(self,path):
        if not os.path.exists(path):
            os.makedirs(path)

    def _save_image(self,image,path):
        image = cv2.resize(image,(self.image_size,self.image_size))
        plt.imsave(path,image)

    def _merge_image(self,image):
        if self.data_formate == "channel_first":
            axis = 0
        else:
            axis = -1
        return np.sum(image,axis=axis)

    def _save_merge_image(self,image,path):
        image = self._merge_image(image)
        self._save_image(image,path)


    def _load_numpy_dir(self,path):
        return np.load(path)

    def _get_channel_num(self,vector):
        if self.data_formate == "channel_first":
            channel_num = vector.shape[0]
        else:
            channel_num = vector.shape[-1]
        return channel_num

    def _process_numpy(self,numpy_list):
        for k,v in numpy_list.items():
            for i in range(self._get_channel_num(v)):
                if self.data_formate == "channel_first":
                    conv_image = v[i,:,:]
                    path = k
                    image_name = "channel_{}.png".format(i+1)
                else:
                    conv_image = v[:,:,i]
                    path = k
                    image_name = "channel_{}.png".format(i + 1)
                yield conv_image,path,image_name

    def _channel_conv_image(self,numpy_list,npz_name):
        for conv_image, path, image_name in self._process_numpy(numpy_list):
            pic_path = os.path.join(self.conv_image_save_path, os.path.basename(npz_name)[:-4], path)
            self._check_path(pic_path)
            self._save_image(conv_image, os.path.join(pic_path, image_name))

    def _merge_channel_conv_image(self,numpy_list,npz_name):
        for k,v in numpy_list.items():
            merge_path = os.path.join(self.conv_image_save_path, os.path.basename(npz_name)[:-4])
            self._check_path(merge_path)
            self._save_image(self._merge_image(v),os.path.join(merge_path,k+".png"))

    def process_conv_image_from_numpy(self):
        for npz in os.listdir(self.image_root_path):
            data = self._load_numpy_dir(os.path.join(self.image_root_path,npz))
            self._channel_conv_image(data,npz)
            self._merge_channel_conv_image(data,npz)


def conv_test():
    image_root_path = "/home/tangmy/programming/picture_class/result/conv_image"
    conv_image_save_path = "/home/tangmy/programming/process_data/result/conv_image"
    data_formate = "channel_first"
    image_size  = 104
    handly = ConvImage(image_root_path=image_root_path,conv_image_save_path=conv_image_save_path,data_formate=data_formate,image_size=image_size)
    handly.process_conv_image_from_numpy()


if __name__ == '__main__':
    conv_test()