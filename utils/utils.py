import numpy as np
from PIL import Image
import matplotlib.pyplot as pl


def preprocess_input(img):
    img /= 255.0
    return img


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


def plot_matrix(hist, labels_name, title=None, name=None, thresh=0.5, axis_labels=None):
    # 利用sklearn中的函数生成混淆矩阵并归一化
    cm = hist.astype('float') / hist.sum(axis=1)[:, np.newaxis]  # 归一化

    # 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    pl.imshow(cm, interpolation='nearest', cmap=pl.get_cmap('Oranges'))
    pl.colorbar()  # 绘制图例

    # 图像标题
    if title is not None:
        pl.title(title)
    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    pl.xticks(num_local, axis_labels, rotation=0)  # 将标签印在x轴坐标上， 并倾斜45度
    pl.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    pl.ylabel('Real label')
    pl.xlabel('Prediction')

    # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) >= 0:
                pl.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd'),
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
    # 显示
    if name is not None:
        pl.savefig(name, dpi=600, bbox_inches='tight')


def fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


if __name__ == "__main__":
    hist = np.array([[1, 2, 0],
                     [0, 3, 0],
                     [0, 0, 3]])
    plot_matrix(hist, [0, 1, 2], title='abc', name='test.png',
                axis_labels=['airplane', 'automobile', 'bird'])
