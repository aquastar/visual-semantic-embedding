import demo, tools, datasets
import numpy as np
from numpy import array
import random
import cPickle as pickle
import numpy as np

dataset_prefix = './data/face/face'

# read data from pkl
text_list = np.array([])
filename_face_img_list = './face_img_list'
f_face_img_list = open(filename_face_img_list, 'wb')
entity_face_pairs = pickle.load(open('entity_face_pairs.pkl', 'rb'))
for key, value in entity_face_pairs.iteritems():
    if value.strip() != '':
        text_list = np.append(text_list, '. '.join(value.strip().split()) )
        f_face_img_list.write(key + '\n')

f_face_img_list.close()
#read images
net = demo.build_convnet()
img_list = demo.compute_fromfile(net, filename_face_img_list, '/home/czq/identify_face/corpface/')

non_empty_img_list = img_list
non_empty_text_list = text_list

non_empty_indexes = list(xrange(text_list.shape[0]))

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
np.save(dataset_prefix + "_train_ims.npy", train_img_list)
# r = np.load("result.npy")

train_caps_file = open(dataset_prefix + '_train_caps.txt', 'wb')
train_text_list = text_list[train_indexes]
for item in train_text_list:
    print >> train_caps_file, item

# dev
dev_img_list = img_list[dev_indexes, :]
np.save(dataset_prefix + "_dev_ims.npy", dev_img_list)

dev_caps_file = open(dataset_prefix + '_dev_caps.txt', 'wb')
dev_text_list = text_list[dev_indexes]
for item in dev_text_list:
    print >> dev_caps_file, item

# test
test_img_list = img_list[test_indexes, :]
np.save(dataset_prefix + "_test_ims.npy", test_img_list)

test_caps_file = open(dataset_prefix + '_test_caps.txt', 'wb')
test_text_list = text_list[test_indexes]
for item in test_text_list:
    print >> test_caps_file, item
