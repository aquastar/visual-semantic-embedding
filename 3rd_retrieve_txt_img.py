import demo, tools, datasets

# retrieve img and text by each other
net = demo.build_convnet()
model = tools.load_model(path_to_model='./data/43/43.npz')
train = datasets.load_dataset('43', load_train=True)[0]
vectors = tools.encode_sentences(model, train[0], verbose=False)
# good cases: 10-29
print demo.retrieve_captions(model, net, train[0], vectors, './out_img.jpg', k=10)
print demo.retrieve_captions(model, net, train[0], vectors, './out_img_1.jpg', k=10)
print demo.retrieve_captions(model, net, train[0], vectors, './out_img_2.jpg', k=10)
print demo.retrieve_captions(model, net, train[0], vectors, './out_img_3.jpg', k=10)
print demo.retrieve_captions(model, net, train[0], vectors, './out_img_4.jpg', k=10)
print demo.retrieve_captions(model, net, train[0], vectors, './out_img_5.jpg', k=10)
print demo.retrieve_captions(model, net, train[0], vectors, './out_img_6.jpg', k=10)
print demo.retrieve_captions(model, net, train[0], vectors, './out_img_7.jpg', k=10)
print demo.retrieve_captions(model, net, train[0], vectors, './43_img/34-50.png', k=10)
print demo.retrieve_captions(model, net, train[0], vectors, './43_img/32-10.png', k=10)
print demo.retrieve_captions(model, net, train[0], vectors, './43_img/32-24.png', k=10)
print demo.retrieve_captions(model, net, train[0], vectors, './43_img/25-17.png', k=10)
print demo.retrieve_captions(model, net, train[0], vectors, './43_img/24-12.png', k=10)
print demo.retrieve_captions(model, net, train[0], vectors, './43_img/22-20.png', k=10)
print demo.retrieve_captions(model, net, train[0], vectors, './43_img/20-45.png', k=10)
print demo.retrieve_captions(model, net, train[0], vectors, './43_img/18-50.png', k=10)
print demo.retrieve_captions(model, net, train[0], vectors, './43_img/11-31.png', k=10)
print demo.retrieve_captions(model, net, train[0], vectors, './43_img/11-26.png', k=10)
print demo.retrieve_captions(model, net, train[0], vectors, './43_img/11-10.png', k=10)
print demo.retrieve_captions(model, net, train[0], vectors, './43_img/10-55.png', k=10)
print demo.retrieve_captions(model, net, train[0], vectors, './43_img/10-47.png', k=10)
print demo.retrieve_captions(model, net, train[0], vectors, './43_img/08-38.png', k=10)


# sentence_vectors = tools.encode_sentences(model, X, verbose=True)
# img_list_vec = demo.compute_fromfile(net, './43_img_list', '/home/czq/visual-semantic-embedding/43_img/')
# f_img_list = open('./43_img_list', 'rb')
# img_list=.read().splitlines()                                                                           
# rst = demo.retrieve_imgs(model, net, sentence_vectors, img_list_vec, file_name, k=5)
