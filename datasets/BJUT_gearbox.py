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

class_name = {0: 'B1_', 1: 'M1_', 2: 'N1_', 3: 'R1_', 4: 'W1_'}

work_condition = ['20', '25', '30', '35', '40', '45', '50', '55']

def data_load(root, conditions, SNR, num_train_samples, num_val_samples, num_test_samples, sig_size, class_label, flag):
    data = []
    label = []
    for lab in class_label:
        name = class_name[lab]
        for condition in conditions:
            num_sample = 0
            path = os.path.join(root, name + work_condition[condition] + '.MAT')
            data_temp = scipy.io.loadmat(path)
            data_temp = data_temp['Data'][:, 1]
            data_temp = np.array(data_temp)
            data_temp = np.expand_dims(data_temp, axis=1)

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


class BJUT_data_split(object):
    def __init__(self, data_length, num_train_samples, num_val_samples, num_test_samples, noise_SNR, classes,
                 op_conditions):

        self.data_length = data_length

        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples

        self.noise_SNR = noise_SNR
        self.classes = classes
        self.op_conditions = op_conditions

        self.data_transforms = {
            'train': Compose([
                Reshape(),
                # AddGaussian(),
                # RandomAddGaussian(),
                RandomScale(),
                RandomStretch(),
                RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Retype(),
                # Scale(1)
            ]),
            'test': Compose([
                Reshape(),
                Retype(),
                # Scale(1)
            ])
        }

    def data_split(self):
        list_train_data = data_load(root=r'D:\datasets\WT-planetary-gearbox-dataset',
                              conditions=self.op_conditions,
                              SNR=self.noise_SNR,
                              num_train_samples=self.num_train_samples,
                              num_val_samples=self.num_val_samples,
                              num_test_samples=self.num_test_samples,
                              sig_size=self.data_length, class_label=self.classes, flag='train')

        list_val_data = data_load(root=r'D:\datasets\WT-planetary-gearbox-dataset',
                              conditions=self.op_conditions,
                              SNR=self.noise_SNR,
                              num_train_samples=self.num_train_samples,
                              num_val_samples=self.num_val_samples,
                              num_test_samples=self.num_test_samples,
                              sig_size=self.data_length, class_label=self.classes, flag='val')

        list_test_data = data_load(root=r'D:\datasets\WT-planetary-gearbox-dataset',
                              conditions=self.op_conditions,
                              SNR=self.noise_SNR,
                              num_train_samples=self.num_train_samples,
                              num_val_samples=self.num_val_samples,
                              num_test_samples=self.num_test_samples,
                              sig_size=self.data_length, class_label=self.classes, flag='test')

        train_data_pd = pd.DataFrame({"data": list_train_data[0], "label": list_train_data[1]})
        val_data_pd = pd.DataFrame({"data": list_val_data[0], "label": list_val_data[1]})
        test_data_pd = pd.DataFrame({"data": list_test_data[0], "label": list_test_data[1]})
        train = dataset(list_data=train_data_pd, transform=self.data_transforms['train'])
        val = dataset(list_data=val_data_pd, transform=self.data_transforms['val'])
        test = dataset(list_data=test_data_pd, transform=self.data_transforms['test'])

        return train, val, test