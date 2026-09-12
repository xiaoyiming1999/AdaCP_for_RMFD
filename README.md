# A Model Calibration Method for Trustworthy Mechanical Fault Diagnosis Called AdaCP, Developed by the HNU Intelligent Fault Diagnosis Group

### Description of the proposed model calibration method for RMFD

A regularization-based calibration method called confidence penalty (CP), proposed by Nobel laureate Geoffrey Hinton, has attracted widespread attention due to its convenience, effectiveness, and intuitiveness. CP directly adds an entropy maximization term to the standard cross-entropy loss, explicitly flattening the predicted distributions of all samples. Meanwhile, it uses a trade-off parameter to control the strength of entropy maximization. However, in our experiments, CP also shows some limitations. The performance of CP is highly dependent on the selected trade-off parameter, which needs to be tuned for the target data by performing cross-validation test on a validation set. On one hand, although the parameter value selected through the cross-validation test offers better overall performance, when we divide the confidence interval [0,1] into different bins, it can be seen that this value does not outperform the other values in every confidence bin. On the other hand, CP's uniform penalty strength, while preventing overconfidence for some samples, may also lead to underconfidence for others. These may suggest that it is necessary to specify a trade-off parameter value for each confidence bin, rather than using only one value uniformly. For this reason, we propose a novel calibration method named adaptive CP (AdaCP), which can learn a trade-off parameter for each confidence bin based on the calibration error of the samples within that bin.

---

## Abstract

This repository provides the official PyTorch implementation of **AdaCP (Adaptive Confidence Penalty)**, a calibration method tailored for **Rotating Machinery Fault Diagnosis (RMFD)**. While deep models achieve high classification accuracy on vibration signals, their softmax confidences are often miscalibrated — overconfident on some inputs and underconfident on others — which undermines trustworthiness in safety-critical fault-diagnosis scenarios. AdaCP generalizes the classic confidence penalty (CP) by maintaining **a dedicated, learnable trade-off coefficient per confidence bin**, adaptively driven by the per-bin calibration gap. The result is a model that is well calibrated across the whole confidence range, not merely on average.

## Key Highlights

- **Per-bin adaptive calibration.** Instead of a single global trade-off parameter as in standard CP, AdaCP learns one coefficient for each confidence bin, so overconfident and underconfident bins are corrected with different strengths and even different directions.
- **Closed-loop update.** Each coefficient is updated from the calibration gap (`mean confidence − accuracy`) measured within its bin on the validation set, so the calibration is data-driven and requires no expensive grid search over the trade-off parameter.
- **Plug-and-play.** AdaCP only modifies the training loss; it is agnostic to the backbone and can be combined with any 1-D CNN (ResNet, WideResNet, MobileNetV2, DRSN, VGG).
- **Comprehensive benchmark.** Five public rotating-machinery datasets, five backbone networks, six calibration/robustness baselines, and three calibration metrics (ECE, NLL, Brier score) with reliability diagrams.

## Method

### 1. Confidence penalty (CP) revisited

Let `p = softmax(z)` be the predicted distribution and `y` the one-hot ground-truth label. Standard CP augments the cross-entropy loss with an entropy-maximization term:

```
L = CE(y, p) − β · H(p)
```

where `H(p) = −Σ_c p_c log p_c` is the predictive entropy and `β ≥ 0` is a fixed trade-off coefficient. Increasing `β` flattens `p`, counteracting overconfidence, but the same `β` is applied uniformly to every sample, which may over-penalize already well-calibrated samples and induce underconfidence.

### 2. Adaptive confidence penalty (AdaCP)

AdaCP relaxes the single global `β` into a **per-bin coefficient vector** `λ = [λ_1, …, λ_M]` over `M` confidence bins:

```
L = CE(y, p) − λ_b · H(p)
```

where `b` is the index of the confidence bin to which the sample belongs (determined by the confidence the model assigns to its ground-truth class). Each `λ_b` is allowed to be **positive or negative**:

- `λ_b > 0` penalizes entropy (flattens the distribution), correcting overconfidence;
- `λ_b < 0` rewards entropy (sharpens the distribution), correcting underconfidence.

After each training epoch, the validation set is partitioned into `M` equal-mass confidence bins, and the calibration gap of bin `i` is computed as

```
gap_i = mean_confidence_i − accuracy_i
```

Each coefficient is then updated multiplicatively according to the sign of its current value:

```
if λ_i ≥ 0:   λ_i ← clamp(λ_i · exp(λ · gap_i),     0,    0.30)
if λ_i <  0:   λ_i ← clamp(λ_i · exp(−λ · gap_i),  −0.15,  0)
```

with a dead-zone threshold to stabilize the sign switching. Here `λ` (hyper-parameter `--lamda`) controls the adaptation step size, `--initial_trade` sets the initial value of every `λ_i`, and `--threshold` controls the dead-zone. In this way, each bin's coefficient is continuously tuned by the calibration error of the samples within that bin, rather than being a single hand-tuned constant.

### 3. Temperature scaling post-processing

As a final post-processing step, a single temperature parameter `T` is fitted on the validation set (via L-BFGS minimizing NLL), scaling the logits `z/T` to further reduce the calibration error. Reliability diagrams are generated before/after temperature scaling for the test set.

## Supported baselines (`--method`)

| Method            | Loss / description                                                                 |
|-------------------|------------------------------------------------------------------------------------|
| `Vanilla`         | Standard cross-entropy                                                             |
| `confidence`      | Classic confidence penalty: `CE − β·H(p)` with a fixed trade-off `β`               |
| `label_smooth`    | Label-smoothing cross-entropy (smoothing = 0.02)                                   |
| `focal_loss`      | Focal loss (γ = 1, uniform α)                                                      |
| `LogNorm`         | Logit normalization (t = 0.2)                                                      |
| `Ada_cp`          | **Proposed** adaptive confidence penalty                                          |

## Supported backbones (`--model`)

| Backbone        | Module / factory                 | Note                              |
|-----------------|----------------------------------|-----------------------------------|
| `Resnet18`      | `resnet18` (1-D)                 | Default, 1-D ResNet-18            |
| `WideResnet`    | `wrn_16_4`                       | Wide ResNet (depth 16, width 4)   |
| `MobileNet`     | `mobilenet_half`                 | 1-D MobileNetV2 (width ratio 0.5) |
| `DRSN`          | `rsnet18`                        | Deep residual shrinkage network   |
| `vgg11`         | `vgg11` (1-D)                    | 1-D VGG-11                        |

## Datasets (`--data_name`)

| Name   | Object                              | Classes | Format     | Source note                          |
|--------|-------------------------------------|---------|------------|--------------------------------------|
| `MCC5` | Gearbox + bearing compound faults   | 8       | `.csv`     | speed/torque circulation conditions  |
| `BJUT` | Wind-turbine planetary gearbox       | 5       | `.MAT`     | multiple working conditions          |
| `THU`  | Direct-drive wind turbine            | 12      | `.tdms`    | Tsinghua wind-turbine dataset        |
| `HUST` | Rolling bearing                      | 7       | `.mat`     | HUST bearing dataset                 |
| `HNU`  | Rolling bearing                      | 6       | `.mat`     | HNU bearing test rig                 |

> **Note:** dataset root paths are currently hard-coded as Windows paths (e.g. `D:\datasets\...`) inside each `datasets/<name>.py` file. Please edit the `root` argument in the corresponding `data_split()` method to point to your own data location before running.

## Evaluation metrics

The following metrics are logged per epoch (and reported after temperature scaling) for the validation and test sets:

- **ECE (Equal-Mass)** — expected calibration error with equal-mass (quantile) bins; primary metric.
- **ECE (Equal-Width)** — expected calibration error with equal-width bins (provided for reference).
- **NLL** — negative log-likelihood (cross-entropy on softmax).
- **Brier score (BS)** — mean squared error between predicted distribution and one-hot label.
- **Reliability diagram** — accuracy-vs-confidence plot plus confidence histogram, saved as a figure.

## Requirements

- Python ≥ 3.7
- PyTorch ≥ 1.8
- numpy, pandas, scipy
- nptdms (for `.tdms` datasets: `THU`, `HUST`)
- matplotlib
- torchvision, Pillow

Install with:

```bash
pip install torch numpy pandas scipy nptdms matplotlib torchvision pillow
```

## Usage

### 1. Prepare data

Download the target dataset(s) and update the `root` path in the corresponding file under `datasets/`.

### 2. Run training

Train AdaCP with the default ResNet-18 backbone on BJUT:

```bash
python AdaCP_main.py --method Ada_cp --model Resnet18 --data_name BJUT
```

Train a baseline (e.g. standard confidence penalty) for comparison:

```bash
python AdaCP_main.py --method confidence --model Resnet18 --data_name BJUT
```

### 3. Key arguments

| Argument            | Type  | Default            | Description                                        |
|---------------------|-------|--------------------|----------------------------------------------------|
| `--method`          | str   | `Ada_cp`           | Training method (see baselines table)              |
| `--model`           | str   | `Resnet18`         | Backbone network (see backbones table)             |
| `--data_name`       | str   | `BJUT`             | Dataset (see datasets table)                       |
| `--data_length`     | int   | `1024`             | Signal segment length (per sample)                 |
| `--num_classes`     | int   | `5`                | Number of classes                                  |
| `--classes`         | list  | `[0,1,2,3,4]`      | Class indices to use                               |
| `--op_conditions`   | list  | `[0]`              | Operating condition indices to use                 |
| `--noise_SNR`       | int   | `100`              | SNR (dB) of additive white Gaussian noise          |
| `--num_train_samples`| int  | `400`              | Samples per class for training                     |
| `--num_val_samples` | int   | `100`              | Samples per class for validation                   |
| `--num_test_samples`| int   | `100`              | Samples per class for testing                      |
| `--epoch`           | int   | `50`               | Number of epochs                                   |
| `--batch_size`      | int   | `64`               | Batch size                                         |
| `--lr`              | float | `1e-3`             | Initial learning rate                              |
| `--weight_decay`    | float | `1e-5`             | L2 weight decay                                    |
| `--gamma`           | float | `0.1`              | LR scheduler decay factor                          |
| `--steps`           | str   | `15,30`            | LR decay milestone epochs (comma-separated)        |
| `--num_bins`        | int   | `10`               | Number of confidence bins                          |
| `--initial_trade`   | float | `0.1`              | Initial per-bin trade-off coefficient (AdaCP)      |
| `--lamda`           | float | `1.5`              | AdaCP adaptation step size                         |
| `--threshold`       | float | `0.1`              | AdaCP dead-zone threshold for sign switching       |
| `--cuda_device`     | str   | `0`                | GPU device id (`-1` for CPU-only setups not yet wired; use CUDA_VISIBLE_DEVICES) |

Checkpoints and logs are written to `./checkpoint/<model_name>_<timestamp>/train.log`.

## Project structure

```
AdaCP_for_RMFD/
├── AdaCP_main.py                 # entry point: argument parsing & trainer dispatch
├── models/
│   ├── Resnet1d.py               # 1-D ResNet (18/34/50/101/152)
│   ├── WideResnet.py             # 1-D Wide ResNet
│   ├── MobilenetV2.py            # 1-D MobileNetV2
│   ├── DRSN.py                   # Deep residual shrinkage network (RSNet)
│   ├── vgg.py                    # 1-D VGG
│   ├── Temperature.py            # Temperature scaling module
│   └── loss.py                   # Losses & metrics (CE, CP, focal, LS, LogNorm, ECE, BS)
├── datasets/
│   ├── MCC5_gearbox_bearing.py   # MCC5 loader
│   ├── BJUT_gearbox.py           # BJUT loader
│   ├── THU_WindTurbine.py        # THU loader (.tdms)
│   ├── HUST_bearing.py           # HUST loader
│   ├── HNU_bearing.py            # HNU loader
│   ├── SequenceDatasets.py       # torch Dataset wrapper
│   ├── sequence_aug.py           # Data augmentation (scale/stretch/crop/noise)
│   └── data_preprocess.py        # Mean-std / min-max normalization
└── utils/
    ├── AdaCP_train.py            # AdaCP training loop + per-bin coefficient update
    ├── Vanilla_train.py          # Baseline training loop + temperature scaling
    ├── logger.py                 # File + console logger
    └── reliability_diagram.py    # Reliability diagram plotting
```

## Citation

If you find this repository useful in your research, please consider citing the following paper:

> Haidong Shao, Yiming Xiao, Jiewu Leng, Xiaoli Zhao, Bin Liu. Collaborative human-computer fault diagnosis via calibrated confidence estimation[J]. Advanced Engineering Informatics, 2025, 65: 103349.

BibTeX:

```bibtex
@article{shao2025collaborative,
  author  = {Shao, Haidong and Xiao, Yiming and Leng, Jiewu and Zhao, Xiaoli and Liu, Bin},
  title   = {Collaborative human-computer fault diagnosis via calibrated confidence estimation},
  journal = {Advanced Engineering Informatics},
  volume  = {65},
  pages   = {103349},
  year    = {2025},
  doi     = {10.1016/j.aei.2025.103349}
}
```

## Acknowledgements

This work is developed by the HNU Intelligent Fault Diagnosis Group. The 1-D backbone implementations are adapted from public PyTorch model zoos, and the reliability-diagram tool is adapted from open-source calibration toolkits. The data preprocessing and unsupervised deep transfer learning (UDTL) components are adapted from the work proposed by Zhao et al., *Applications of Unsupervised Deep Transfer Learning to Intelligent Fault Diagnosis: A Survey and Comparative Study*.

## Contact

- **Author:** Yiming Xiao — xiaoym@hnu.edu.cn
- **Mentor:** Haidong Shao — hdshao@hnu.edu.cn

## License

The code is released for research and educational purposes. Please contact the authors for further usage.
