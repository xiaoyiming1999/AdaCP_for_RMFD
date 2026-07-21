import os

from nptdms import TdmsFile
import pandas as pd

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

class_name = {0: 'aero-asymmetry2', 1: 'back-bearingbase2', 2: 'front-bearingbase2',
              3: 'gundongti3', 4: 'horizontal-misalignment1', 5: 'inter-ring2',
              6: 'mass-imbalance1', 7: 'normal2', 8: 'outer-ring2',
              9: 'pianhang2', 10: 'three-papers1', 11: 'vertical-misalignment1'}

work_condition = ['-24', '-28', '-32', '-36', '-40', '-44']

def data_load(root, conditions, SNR, num_train_samples, num_val_samples, num_test_samples, sig_size, class_label, flag):

    num_sampling = 63
    data = []
    label = []
    for lab in class_label:
        name = class_name[lab]
        for condition in conditions:
            num_sample = 0
            path = os.path.join(root, name, name + work_condition[condition] + '.tdms')
            with TdmsFile.open(path) as tdms_file:

                all_channel_data = []
                for i in range(num_sampling):
                    group_name = i + 1
                    group = tdms_file[str(group_name)]
                    # channels = ['电压通道0', '电压通道1', '电压通道2', '电压通道3', '电压通道4', '电压通道5', '电压通道6', '电压通道7', '串行通信']
                    channel = group['电压通道3']
                    channel_data = channel[0:51200]
                    channel_data = np.array(channel_data)
                    all_channel_data.append(channel_data)
                all_channel_data = np.concatenate(all_channel_data)
                data_temp = np.expand_dims(all_channel_data, axis=1)

                # train
                if flag == 'train':
                    start, end = 0, sig_size
                    while end <= data_temp.shape[0] and num_sample < num_train_samples:
                        current_sample = data_temp[start:end]
                        current_sample = AddWhiteGaussian(current_sample, SNR)
                        # current_sample = np.fft.fft(current_sample)
                        # current_sample = np.abs(current_sample) / len(current_sample)
                        # current_sample = current_sample[range(int(current_sample.shape[0] / 2))]
                        # current_sample = current_sample.reshape(-1, 1)
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
                        # current_sample = np.fft.fft(current_sample)
                        # current_sample = np.abs(current_sample) / len(current_sample)
                        # current_sample = current_sample[range(int(current_sample.shape[0] / 2))]
                        # current_sample = current_sample.reshape(-1, 1)
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
                        # current_sample = np.fft.fft(current_sample)
                        # current_sample = np.abs(current_sample) / len(current_sample)
                        # current_sample = current_sample[range(int(current_sample.shape[0] / 2))]
                        # current_sample = current_sample.reshape(-1, 1)
                        data.append(current_sample)
                        label.append(lab)
                        start += sig_size
                        end += sig_size
                        num_sample += 1

    return [data, label]


class THU_data_split(object):
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
        list_train_data = data_load(root=r'D:\datasets\直驱型风力机故障数据-清华大学',
                              conditions=self.op_conditions,
                              SNR=self.noise_SNR,
                              num_train_samples=self.num_train_samples,
                              num_val_samples=self.num_val_samples,
                              num_test_samples=self.num_test_samples,
                              sig_size=self.data_length, class_label=self.classes, flag='train')

        list_val_data = data_load(root=r'D:\datasets\直驱型风力机故障数据-清华大学',
                              conditions=self.op_conditions,
                              SNR=self.noise_SNR,
                              num_train_samples=self.num_train_samples,
                              num_val_samples=self.num_val_samples,
                              num_test_samples=self.num_test_samples,
                              sig_size=self.data_length, class_label=self.classes, flag='val')

        list_test_data = data_load(root=r'D:\datasets\直驱型风力机故障数据-清华大学',
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
