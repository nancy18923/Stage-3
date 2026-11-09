"""
Compress Stage_3_Report.docx to fit the 5-page limit:
- rewrite every text section in a more compact form
- merge the three convergence figures into one 3-row grid figure
- merge related rows of the configuration table
Run:  python script/stage_3_script/script_compress_report.py
"""
import copy
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import docx
from docx.shared import Inches, Pt

REPORT = os.path.join(os.path.dirname(PROJECT_ROOT), 'Stage_3_Report.docx')
GRID_FIG = os.path.join(PROJECT_ROOT, 'result', 'stage_3_result', 'figures', 'curve_grid_all.png')


def set_text(p, text):
    """replace the paragraph text, keeping the formatting of the first run"""
    runs = p.runs
    if not runs:
        p.add_run(text)
        return
    runs[0].text = text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)


def set_two_runs(p, bold_part, rest):
    """two-run paragraph: bold lead-in + normal remainder (used for the bullets)"""
    runs = p.runs
    if len(runs) < 2:
        set_text(p, bold_part + rest)
        return
    runs[0].text = bold_part
    runs[0].font.bold = True
    runs[1].text = rest
    for r in runs[2:]:
        r._r.getparent().remove(r._r)


def set_cell(cell, text, bold=False, size=10):
    for p in cell.paragraphs[1:]:
        p._p.getparent().remove(p._p)
    p0 = cell.paragraphs[0]
    for r in list(p0.runs):
        r._r.getparent().remove(r._r)
    run = p0.add_run(text)
    run.font.size = Pt(size)
    run.font.bold = bold


def main():
    d = docx.Document(REPORT)
    paras = list(d.paragraphs)

    # ---------------- text rewrites (indices per the dumped structure) ----------------
    set_text(paras[8],
        'This stage studies image classification for object recognition with convolutional '
        'neural network (CNN) models. Given the raw pixels of an image, the task is to predict '
        'its class: which digit (0-9) a 28 x 28 MNIST image shows, which of the 40 people a '
        '112 x 92 ORL face belongs to, and which of the 10 object categories a 32 x 32 CIFAR-10 '
        'image contains. Following the ECS 170 template, we write a new dataset loader and three '
        'CNN method classes, train them with mini-batch gradient descent in PyTorch, evaluate '
        'them with Accuracy, Precision, Recall and F1 score, and run an ablation study over the '
        'main design choices.')

    set_text(paras[11],
        'All three models share one pipeline (Figure 1). Each convolutional block applies '
        'Conv2d (3 x 3, stride 1, padding 1), BatchNorm2d, ReLU and a 2 x 2 MaxPool2d, so the '
        'spatial size is preserved inside a block and halved between blocks. AdaptiveAvgPool2d '
        '(2 x 2) + Flatten fixes the feature size independently of the input resolution, so the '
        'same classifier head serves 28 x 28, 112 x 92 and 32 x 32 inputs. The head is '
        'Linear(128) + ReLU + Dropout followed by Linear(n_classes); the predicted label is the '
        'arg-max of the logits (Softmax is omitted because CrossEntropyLoss fuses log-softmax '
        'with the NLL). The variants differ only in the input shape, trunk depth/width, dropout '
        'rate and number of classes: MNIST 2 blocks [32, 64], dropout 0.25, 10 classes; ORL '
        '2 blocks [32, 64], dropout 0.5, 40 classes; CIFAR 3 blocks [32, 64, 128], dropout 0.3, '
        '10 classes. ORL and CIFAR use heavier dropout because their classes are harder to '
        'separate.')

    set_text(paras[13],
        'Figure 1. CNN architecture: shared pipeline (top) and per-dataset configuration '
        'differences (bottom).')

    set_text(paras[19],
        'We use the three image datasets provided by the instructor, each stored as a pickle '
        'file and loaded by our new Dataset_Loader class:')
    set_two_runs(paras[20], 'MNIST: ',
        '60,000 training / 10,000 testing 28 x 28 grayscale digit images, 10 classes.')
    set_two_runs(paras[21], 'ORL: ',
        '400 face images of 40 people (112 x 92); the given split keeps 9 images per person for '
        'training (360) and 1 for testing (40). The raw file stores three identical colour '
        'channels, so we keep a single channel.')
    set_two_runs(paras[22], 'CIFAR-10: ',
        '50,000 training / 10,000 testing 32 x 32 RGB images of 10 object classes.')
    set_text(paras[23],
        'Pixels are divided by 255 (range [0, 1]) and reshaped to (C, H, W) tensors; ORL labels '
        'are shifted to 0-39. The partitions are given in advance, so no extra split or '
        'cross-validation is performed.')

    set_text(paras[26],
        'All experiments use a single train/test protocol with the pre-given partition (no '
        'cross-validation), implemented in Setting_Train_Test, which injects the test loader so '
        'that test loss and accuracy are recorded at every epoch. The three CNNs share the trunk '
        'above and differ only in:')
    set_text(paras[27],
        'The ablation study in Section 3.7 changes exactly one row of this baseline at a time, '
        'so every performance difference is attributable to a single factor.')

    set_text(paras[30],
        'We report Accuracy, Precision, Recall and F1 score computed with scikit-learn on the '
        'test set. Precision (TP/(TP+FP)), Recall (TP/(TP+FN)) and F1 (2PR/(P+R)) are computed '
        'per class and averaged weighted by the class support, which fits the multiclass setting '
        '(10, 40 and 10 classes) and accounts for its mild class imbalance.')

    set_text(paras[33], 'https://github.com/nancy18923/Stage-3')
    set_text(paras[34],
        'Key code files: script/stage_3_script/script_cnn.py (driver), script_ablation.py '
        '(ablation sweep), script_plot_curves.py and script_plot_architecture.py (figures); '
        'local_code/stage_3_code/ (Dataset_Loader, Method_CNN_MNIST/ORL/CIFAR, '
        'Setting_Train_Test, Evaluate_Metrics, Result_Saver).')
    set_text(paras[35],
        'The MNIST, ORL and CIFAR files belong in data/stage_3_data but are too large for '
        'GitHub (MNIST 424 MB, CIFAR 181 MB).')

    set_text(paras[39],
        'Figure 2 shows the training convergence curves of the three CNNs (loss = '
        'CrossEntropyLoss, Adam, lr = 1e-3; train in blue, test in red). On MNIST the training '
        'loss falls from 0.393 to 0.103 within two epochs and then smoothly to 0.036 (98.8% '
        'training accuracy) at epoch 10, while the test accuracy saturates at 98.8-98.9%. On '
        'ORL the loss decreases from 3.69 to about 0.85 over 40 epochs (training accuracy 3.9% '
        '-> 75.3%); the test accuracy peaks at 92.5% (epoch 31) and ends at 82.5%, fluctuating '
        'because only 360 training images are available. On CIFAR the loss decreases from 1.31 '
        'to 0.26 (training accuracy 90.6%), but the test loss stops improving after about epoch '
        '8 and the test accuracy stays around 77-79%, i.e. the model starts to overfit. In all '
        'three cases the loss decreases monotonically overall and flattens towards the end, so '
        'the models converge gradually under mini-batch gradient descent.')

    set_text(paras[49],
        'The baseline CNNs are evaluated on the test set of each dataset; Accuracy, Precision, '
        'Recall and F1 (weighted over classes) are summarized below:')
    set_text(paras[50],
        'MNIST is nearly solved: 98.8% accuracy means 117 errors out of 10,000. ORL reaches '
        '82.5% (7 errors of 40); because the test set holds exactly one image per person, every '
        'mistake costs 2.5% and the weighted precision (0.7375) is pulled below accuracy by the '
        'confused identities. CIFAR-10 is intrinsically harder: 77.7% is a typical score for a '
        'plain CNN of this size without data augmentation, given the small objects and the large '
        'variation in pose, scale, lighting and background.')

    set_text(paras[53],
        'We change one configuration knob of the baseline at a time (kernel size, padding, '
        'depth, width, pooling type, batch norm, dropout, hidden size, optimizer, loss function, '
        'learning rate). Because a full sweep on CPU is too expensive, MNIST and CIFAR train for '
        '3 epochs on 300 images per class and ORL for 20 epochs on all 360 training images; '
        'script_ablation.py sub-classes the method classes on the fly and reuses the same '
        'Setting/Result/Evaluate objects as the baseline run. The numbers below are therefore '
        'not comparable with Section 3.6.')
    set_text(paras[55],
        'Removing batch normalization is the most damaging change (MNIST 0.8803 -> 0.6710, ORL '
        '0.5750 -> 0.4250), and the optimizer matters as much: SGD at the same learning rate '
        'collapses (0.4767 / 0.2250), while lr = 1e-4 converges too slowly in the few epochs '
        'available. More capacity helps almost everywhere: an extra conv block gives the largest '
        'gain on MNIST (0.9563) and ORL (0.8500), a wider trunk is best on CIFAR (0.4460), and '
        '256 hidden units improve all three datasets. The remaining knobs are dataset-dependent: '
        'a 5 x 5 kernel helps MNIST and ORL but hurts CIFAR, average pooling hurts CIFAR badly '
        '(0.2112) yet helps ORL, and removing dropout boosts the quick protocol (CIFAR 0.4823) '
        'but discards the regularization the full training runs need. NLLLoss reproduces the '
        'baseline exactly because it is mathematically equivalent to CrossEntropyLoss. Overall, '
        'batch norm, Adam with lr = 1e-3, and sufficient depth/width are the most important '
        'choices.')

    # ---------------- figures ----------------
    # 1) shrink the architecture figure a bit
    arch = d.inline_shapes[0]
    ratio = arch.height / arch.width
    arch.width = Inches(5.0)
    arch.height = int(Inches(5.0) * ratio)

    # 2) replace the three separate curve figures (p44/45/46) with one grid figure in p44
    p44 = paras[44]
    for r in list(p44.runs):          # removes the drawing run and the old caption
        r._r.getparent().remove(r._r)
    img_run = p44.add_run()
    img_run.add_picture(GRID_FIG, width=Inches(5.8))
    cap_run = p44.add_run('Figure 2. Convergence curves of the three CNNs: loss (left) and '
                          'accuracy (right) per epoch; train in blue, test in red.')
    cap_run.font.size = Pt(9)
    cap_run.font.italic = True

    # ---------------- delete now-redundant paragraphs ----------------
    for idx in (45, 46, 40, 41, 42, 43, 56, 57, 15, 37):
        p = paras[idx]
        p._p.getparent().remove(p._p)

    # ---------------- configuration table: merge related rows ----------------
    tbl = d.tables[1]
    for ri in (14, 12, 10, 6, 4):      # pooling / FC head / optimizer / epochs / seed rows
        tbl._tbl.remove(tbl.rows[ri]._tr)
    rows = tbl.rows
    set_cell(rows[3].cells[0], 'Conv kernel / pooling')
    set_cell(rows[3].cells[1], '3 x 3, stride 1, pad 1; MaxPool 2 x 2')
    set_cell(rows[3].cells[2], 'same')
    set_cell(rows[3].cells[3], 'same')
    set_cell(rows[4].cells[0], 'Head')
    set_cell(rows[4].cells[1], 'AdaptiveAvgPool 2 x 2 + Flatten; Linear 128 + ReLU + Dropout')
    set_cell(rows[4].cells[2], 'same')
    set_cell(rows[4].cells[3], 'same')
    set_cell(rows[7].cells[0], 'Loss / optimizer')
    set_cell(rows[7].cells[1], 'CrossEntropyLoss; Adam, lr = 1e-3')
    set_cell(rows[7].cells[2], 'same')
    set_cell(rows[7].cells[3], 'same')
    set_cell(rows[8].cells[0], 'Batch size / epochs')
    set_cell(rows[8].cells[1], '64 / 10')
    set_cell(rows[8].cells[2], '32 / 40')
    set_cell(rows[8].cells[3], '64 / 20')
    set_cell(rows[9].cells[0], 'Init / seed')
    set_cell(rows[9].cells[1], 'PyTorch default init (reset per run); numpy & torch seed = 2')
    set_cell(rows[9].cells[2], 'same')
    set_cell(rows[9].cells[3], 'same')

    d.save(REPORT)
    print('saved', REPORT)
    print('paragraphs now:', len(docx.Document(REPORT).paragraphs))
    print('tables:', [(len(t.rows), len(t.columns)) for t in docx.Document(REPORT).tables])


if __name__ == '__main__':
    main()
