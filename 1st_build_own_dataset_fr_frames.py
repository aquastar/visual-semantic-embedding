import demo, tools, datasets
import numpy as np
from numpy import array
import random

# def f(x):
#     return x.stripe() != ''

# class MyList(list):
#     def __init__(self, *args):
#         super(MyList, self).__init__(args)
#
#     def __sub__(self, other):
#         return self.__class__(*[item for item in self if item not in other])


# read caption
f = open('43_captions', 'rb')
text_list = array([x.strip() for x in f.readlines()])
print 'read caption'

# read images
net = demo.build_convnet()
img_list = demo.compute_fromfile(net, './43_img_list', '/home/czq/visual-semantic-embedding/43_img/')

# select indexes which is not empty string
non_empty_indexes = []
for k, v in enumerate(text_list):
    if v.strip() != '':
        non_empty_indexes.append(k)

non_empty_indexes = sorted(non_empty_indexes)

# filter by non-empty string
non_empty_img_list = img_list[non_empty_indexes, :]
non_empty_text_list = text_list[non_empty_indexes]

# sampling to make train/dev/test to 8/1/1
total_num = len(non_empty_indexes)
train_num = total_num * 8 / 10
dev_num = (total_num - train_num) / 2
test_num = total_num - train_num - dev_num
train_indexes = sorted(random.sample(non_empty_indexes, train_num))
dev_indexes = sorted(random.sample([item for item in non_empty_indexes if item not in train_indexes], dev_num))
test_indexes = sorted([item for item in non_empty_indexes if item not in train_indexes and item not in dev_indexes])

# filter by split and output to captions(txt) and images(npy)

# train
train_img_list = img_list[train_indexes, :]
np.save("43_train_ims.npy", train_img_list)
# r = np.load("result.npy")

train_caps_file = open('43_train_caps.txt', 'wb')
train_text_list = text_list[train_indexes]
for item in train_text_list:
    print >> train_caps_file, item

# dev
dev_img_list = img_list[dev_indexes, :]
np.save("43_dev_ims.npy", dev_img_list)

dev_caps_file = open('43_dev_caps.txt', 'wb')
dev_text_list = text_list[dev_indexes]
for item in dev_text_list:
    print >> dev_caps_file, item

# test
test_img_list = img_list[test_indexes, :]
np.save("43_test_ims.npy", test_img_list)

test_caps_file = open('43_test_caps.txt', 'wb')
test_text_list = text_list[test_indexes]
for item in test_text_list:
    print >> test_caps_file, item


