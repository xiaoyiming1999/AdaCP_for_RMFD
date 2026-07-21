import pandas as pd
import os
import scipy

from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *

def AddWhiteGaussian(seq, SNR):
    Ps = np.sum(seq ** 2) / (seq.shape[0])
    Pn = Ps / (10 ** (SNR / 10))
    # np.random.seed(123)
    noise = np.random.normal(loc=0, scale=1, size=seq.shape)
    noise = noise * np.sqrt(Pn)
    # snr = 10 * np.log10(np.sum(seq ** 2)/np.sum(noise ** 2))
    signal_add_noise = seq + noise
    return signal_add_noise

class_name = {0: 'health_',
              1: 'gear_pitting_H_',
              2: 'gear_wear_H_',
              3: 'miss_teeth_',
              4: 'teeth_break_H_',
              5: 'teeth_crack_H_',
              6: 'teeth_break_and_bearing_inner_H_',
              7: 'teeth_break_and_bearing_outer_H_', }

work_condition = ['speed_circulation_10Nm-1000rpm', 'speed_circulation_10Nm-2000rpm', 'speed_circulation_10Nm-3000rpm',
                  'speed_circulation_20Nm-1000rpm', 'speed_circulation_20Nm-2000rpm', 'speed_circulation_20Nm-3000rpm',
                  'torque_circulation_1000rpm_10Nm', 'torque_circulation_1000rpm_20Nm',
                  'torque_circulation_2000rpm_10Nm', 'torque_circulation_2000rpm_20Nm',
                  'torque_circulation_3000rpm_10Nm', 'torque_circulation_3000rpm_20Nm']

def data_load(root, conditions, SNR, num_train_samples, num_val_samples, num_test_samples, sig_size, class_label, flag):
    data = []
    label = []
    for lab in class_label:
        name = class_name[lab]
        for condition in conditions:
            num_sample = 0
            path = os.path.join(root, name + work_condition[condition] + '.csv')
            data_temp = pd.read_csv(path, header=0, usecols=['gearbox_vibration_y'])
            data_temp = np.array(data_temp)

            # randomize data
            # data_frame = []
            # start, end = 0, sig_size
            # while end <= data_temp.shape[0]:
                # current_sample = data_temp[start:end]
                # data_frame.append(current_sample)
                # start += sig_size
                # end += sig_size
            # data_frame = np.concatenate(data_frame, axis=1)
            # np.random.shuffle(data_frame)
            # data_temp = data_frame.reshape(-1, 1)

            # train
            if flag == 'train':
                start, end = 0, sig_size
                while end <= data_temp.shape[0] and num_sample < num_train_samples:
                    current_sample = data_temp[start:end]
                    current_sample = AddWhiteGaussian(current_sample, SNR)
                    data.append(current_sample)
                    label.append(lab)
                    start += sig_size
                    end += sig_size
                    num_sample += 1
            # val
            elif flag == 'val':
                start, end = sig_size * num_train_samples, sig_size + sig_size * num_train_samples
                while end <= data_temp.shape[0] and num_sample < num_val_samples:
                    current_sample = data_temp[start:end]
                    current_sample = AddWhiteGaussian(current_sample, SNR)
                    data.append(current_sample)
                    label.append(lab)
                    start += sig_size
                    end += sig_size
                    num_sample += 1
            # test
            elif flag == 'test':
                start, end = sig_size * (num_train_samples + num_val_samples), sig_size + sig_size * (
                        num_train_samples + num_val_samples)
                while end <= data_temp.shape[0] and num_sample < num_test_samples:
                    current_sample = data_temp[start:end]
                    current_sample = AddWhiteGaussian(current_sample, SNR)
                    data.append(current_sample)
                    label.append(lab)
                    start += sig_size
                    end += sig_size
                    num_sample += 1

    return [data, label]

class MCC5_data_split(object):
    def __init__(self, data_length, num_train_samples, num_val_samples, num_test_samples, noise_SNR, classes,
                 op_conditions):

        self.data_length = data_length

        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples

        self.noise_SNR = noise_SNR
        self.classes = classes
        self.op_conditions = op_conditions

        self.data_transforms = Reshape()

    def data_split(self):
        list_train_data = data_load(root=r'D:\datasets\MCC5',
                              conditions=self.op_conditions,
                              SNR=self.noise_SNR,
                              num_train_samples=self.num_train_samples,
                              num_val_samples=self.num_val_samples,
                              num_test_samples=self.num_test_samples,
                              sig_size=self.data_length, class_label=self.classes, flag='train')

        list_val_data = data_load(root=r'D:\datasets\MCC5',
                              conditions=self.op_conditions,
                              SNR=self.noise_SNR,
                              num_train_samples=self.num_train_samples,
                              num_val_samples=self.num_val_samples,
                              num_test_samples=self.num_test_samples,
                              sig_size=self.data_length, class_label=self.classes, flag='val')

        list_test_data = data_load(root=r'D:\datasets\MCC5',
                              conditions=self.op_conditions,
                              SNR=self.noise_SNR,
                              num_train_samples=self.num_train_samples,
                              num_val_samples=self.num_val_samples,
                              num_test_samples=self.num_test_samples,
                              sig_size=self.data_length, class_label=self.classes, flag='test')

        train_data_pd = pd.DataFrame({"data": list_train_data[0], "label": list_train_data[1]})
        val_data_pd = pd.DataFrame({"data": list_val_data[0], "label": list_val_data[1]})
        test_data_pd = pd.DataFrame({"data": list_test_data[0], "label": list_test_data[1]})
        train = dataset(list_data=train_data_pd, transform=self.data_transforms)
        val = dataset(list_data=val_data_pd, transform=self.data_transforms)
        test = dataset(list_data=test_data_pd, transform=self.data_transforms)

        return train, val, test