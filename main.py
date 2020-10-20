#%%
"""温馨提示：目标检测是非常非常复杂的任务，建议使用6GB以上的显卡运行！ """
import os,glob 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np 
import tensorflow as tf 
from tensorflow import keras 

tf.random.set_seed(2234)
np.random.seed(2234)

# Shift+Enter
# tf2
#%%
print(tf.__version__)
print(tf.test.is_gpu_available())
# %%
import xml.etree.ElementTree as ET

def parse_annotation(img_dir, ann_dir, labels):
    # img_dir: image path
    # ann_dir: annotation xml file path
    # labels: ('sugarweet', 'weed')
    # parse annotation info from xml file
    """
    <annotation> 
        <object>
            <name>sugarbeet</name> 
            <bndbox>
                <xmin>1</xmin>
                <ymin>250</ymin>
                <xmax>53</xmax>
                <ymax>289</ymax>
            </bndbox>
        </object>
        <object>....

    """
    imgs_info = []

    max_boxes = 0
    # for each annotation xml file
    for ann in os.listdir(ann_dir): 
        tree = ET.parse(os.path.join(ann_dir, ann))

        img_info = dict()
        boxes_counter = 0
        img_info['object'] = []
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img_info['filename'] = os.path.join(img_dir,elem.text)
            if 'width' in elem.tag:
                img_info['width'] = int(elem.text)
                assert img_info['width'] == 512
            if 'height' in elem.tag:
                img_info['height'] = int(elem.text)
                assert img_info['height'] == 512

            if 'object' in elem.tag or 'part' in elem.tag:
                # x1-y1-x2-y2-label
                object_info = [0,0,0,0,0]
                boxes_counter += 1
                for attr in list(elem):
                    if 'name' in attr.tag:
                        label = labels.index(attr.text) + 1
                        object_info[4] = label
                    if 'bndbox' in attr.tag:
                        for pos in list(attr):
                            if 'xmin' in pos.tag:
                                object_info[0] = int(pos.text)
                            if 'ymin' in pos.tag:
                                object_info[1] = int(pos.text)
                            if 'xmax' in pos.tag:
                                object_info[2] = int(pos.text)
                            if 'ymax' in pos.tag:
                                object_info[3] = int(pos.text)
                img_info['object'].append(object_info)

        imgs_info.append(img_info) # filename, w/h/box_info
        # (N,5)=(max_objects_num, 5)
        if boxes_counter > max_boxes:
            max_boxes  = boxes_counter
    # the maximum boxes number is max_boxes
    # [b, 40, 5] 
    boxes = np.zeros([len(imgs_info), max_boxes, 5])
    print(boxes.shape)
    imgs = [] # filename list
    for i, img_info in enumerate(imgs_info):
        # [N,5]
        img_boxes = np.array(img_info['object'])
        # overwrite the N boxes info
        boxes[i,:img_boxes.shape[0]] = img_boxes
 
        imgs.append(img_info['filename'])

        # print(img_info['filename'], boxes[i,:5])
    # imgs: list of image path
    # boxes: [b,40,5]
    return imgs, boxes
            


# %%
obj_names = ('sugarbeet', 'weed')
imgs, boxes = parse_annotation('data/train/image', 'data/train/annotation', obj_names)


# %%

def preprocess(img, img_boxes):
    # img: string
    # img_boxes: [40,5]
    x = tf.io.read_file(img)
    x = tf.image.decode_png(x, channels=3)
    x = tf.image.convert_image_dtype(x, tf.float32)

    return x, img_boxes

# 1.2 
def get_dataset(img_dir, ann_dir, batchsz):
    # return tf dataset 
    # [b], boxes [b, 40, 5]
    imgs, boxes = parse_annotation(img_dir, ann_dir, obj_names)
    db = tf.data.Dataset.from_tensor_slices((imgs, boxes))
    db = db.shuffle(1000).map(preprocess).batch(batchsz).repeat()

    print('db Images:', len(imgs))

    return db 

# %%
train_db = get_dataset('data/train/image', 'data/train/annotation', 10)
print(train_db)

# %%
# 1.3 visual the db
from matplotlib import pyplot as plt 
from matplotlib import patches 

def db_visualize(db):
    # imgs:[b, 512, 512, 3]
    # imgs_boxes: [b, 40, 5]
    imgs, imgs_boxes = next(iter(db))
    img, img_boxes = imgs[0], imgs_boxes[0]

    f,ax1 = plt.subplots(1,figsize=(10,10))
    # display the image, [512,512,3]
    ax1.imshow(img)
    for x1,y1,x2,y2,l in img_boxes: # [40,5]
        x1,y1,x2,y2 = float(x1), float(y1), float(x2), float(y2)
        w = x2 - x1 
        h = y2 - y1 

        if l==1: # green for sugarweet
            color = (0,1,0)
        elif l==2: # red for weed
            color = (1,0,0) # (R,G,B)
        else: # ignore invalid boxes
            break

        rect = patches.Rectangle((x1,y1), w, h, linewidth=2,
        edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
    


# %%
db_visualize(train_db)


# %%
# 1.4 data augmentation
import imgaug as ia
from    imgaug import augmenters as iaa
def augmentation_generator(yolo_dataset):
    '''
    Augmented batch generator from a yolo dataset

    Parameters
    ----------
    - YOLO dataset
    
    Returns
    -------
    - augmented batch : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
        batch : tupple(images, annotations)
        batch[0] : images : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
        batch[1] : annotations : tensor (shape : batch_size, max annot, 5)
    '''
    for batch in yolo_dataset:
        # conversion tensor->numpy
        img = batch[0].numpy()
        boxes = batch[1].numpy()
        # conversion bbox numpy->ia object
        ia_boxes = []
        for i in range(img.shape[0]):
            ia_bbs = [ia.BoundingBox(x1=bb[0],
                                       y1=bb[1],
                                       x2=bb[2],
                                       y2=bb[3]) for bb in boxes[i]
                      if (bb[0] + bb[1] +bb[2] + bb[3] > 0)]
            ia_boxes.append(ia.BoundingBoxesOnImage(ia_bbs, shape=(512, 512)))
        # data augmentation
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Multiply((0.4, 1.6)), # change brightness
            #iaa.ContrastNormalization((0.5, 1.5)),
            #iaa.Affine(translate_px={"x": (-100,100), "y": (-100,100)}, scale=(0.7, 1.30))
            ])
        #seq = iaa.Sequential([])
        seq_det = seq.to_deterministic()
        img_aug = seq_det.augment_images(img)
        img_aug = np.clip(img_aug, 0, 1)
        boxes_aug = seq_det.augment_bounding_boxes(ia_boxes)
        # conversion ia object -> bbox numpy
        for i in range(img.shape[0]):
            boxes_aug[i] = boxes_aug[i].remove_out_of_image().clip_out_of_image()
            for j, bb in enumerate(boxes_aug[i].bounding_boxes):
                boxes[i,j,0] = bb.x1
                boxes[i,j,1] = bb.y1
                boxes[i,j,2] = bb.x2
                boxes[i,j,3] = bb.y2
        # conversion numpy->tensor
        batch = (tf.convert_to_tensor(img_aug), tf.convert_to_tensor(boxes))
        #batch = (img_aug, boxes)
        yield batch
#%%
aug_train_db = augmentation_generator(train_db)
db_visualize(aug_train_db)

# %%
IMGSZ = 512
GRIDSZ = 16
ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]



# %%
def process_true_boxes(gt_boxes, anchors):
    # gt_boxes: [40,5]
    # 512//16=32
    scale = IMGSZ // GRIDSZ
    # [5,2]
    anchors = np.array(anchors).reshape((5, 2))

    # mask for object
    detector_mask = np.zeros([GRIDSZ, GRIDSZ, 5, 1])
    #x-y-w-h-l
    matching_gt_box = np.zeros([GRIDSZ, GRIDSZ, 5, 5])
    # [40,5] x1-y1-x2-y2-l => x-y-w-h-l
    gt_boxes_grid = np.zeros(gt_boxes.shape)
    # DB: tensor => numpy
    gt_boxes = gt_boxes.numpy()

    for i,box in enumerate(gt_boxes): # [40,5]
        # box: [5], x1-y1-x2-y2-l
        # 512 => 16
        x = ((box[0]+box[2])/2)/scale
        y = ((box[1]+box[3])/2)/scale
        w = (box[2] - box[0]) / scale
        h = (box[3] - box[1]) / scale
        # [40,5] x-y-w-h-l
        gt_boxes_grid[i] = np.array([x,y,w,h,box[4]])

        if w*h > 0: # valid box
            # x,y: 7.3, 6.8
            best_anchor = 0
            best_iou = 0
            for j in range(5):
                interct = np.minimum(w, anchors[j,0]) * np.minimum(h, anchors[j,1])
                union = w*h + (anchors[j,0]*anchors[j,1]) - interct
                iou = interct / union

                if iou > best_iou: # best iou
                    best_anchor = j 
                    best_iou = iou 
            # found the best anchors
            if best_iou>0:
               x_coord = np.floor(x).astype(np.int32)
               y_coord = np.floor(y).astype(np.int32)
               # [b,h,w,5,1]
               detector_mask[y_coord, x_coord, best_anchor] = 1
               # [b,h,w,5,x-y-w-h-l]
               matching_gt_box[y_coord, x_coord, best_anchor] = \
                   np.array([x,y,w,h,box[4]])

    # [40,5] => [16,16,5,5]
    # [16,16,5,5]
    # [16,16,5,1]
    # [40,5]
    return matching_gt_box, detector_mask, gt_boxes_grid


# %%
# 2.2 
def ground_truth_generator(db):

    for imgs, imgs_boxes in db:
        # imgs: [b,512,512,3]
        # imgs_boxes: [b,40,5]

        batch_matching_gt_box = []
        batch_detector_mask = []
        batch_gt_boxes_grid = []

        # print(imgs_boxes[0,:5])

        b = imgs.shape[0]
        for i in range(b): # for each image
            matching_gt_box, detector_mask, gt_boxes_grid = \
                process_true_boxes(imgs_boxes[i], ANCHORS)
            batch_matching_gt_box.append(matching_gt_box)
            batch_detector_mask.append(detector_mask)
            batch_gt_boxes_grid.append(gt_boxes_grid)
        # [b, 16,16,5,1]
        detector_mask = tf.cast(np.array(batch_detector_mask), dtype=tf.float32)
        # [b,16,16,5,5] x-y-w-h-l
        matching_gt_box = tf.cast(np.array(batch_matching_gt_box), dtype=tf.float32)
        # [b,40,5] x-y-w-h-l
        gt_boxes_grid = tf.cast(np.array(batch_gt_boxes_grid), dtype=tf.float32)

        # [b,16,16,5]
        matching_classes = tf.cast(matching_gt_box[...,4], dtype=tf.int32)
        # [b,16,16,5,3]
        matching_classes_oh = tf.one_hot(matching_classes, depth=3)
        # x-y-w-h-conf-l1-l2
        # [b,16,16,5,2]
        matching_classes_oh = tf.cast(matching_classes_oh[...,1:], dtype=tf.float32)


        # [b,512,512,3]
        # [b,16,16,5,1]
        # [b,16,16,5,5]
        # [b,16,16,5,2]
        # [b,40,5]
        yield imgs, detector_mask, matching_gt_box, matching_classes_oh,gt_boxes_grid





# %%
# 2.3 visualize object mask
# train_db -> aug_train_db -> train_gen
train_gen = ground_truth_generator(aug_train_db)

img, detector_mask, matching_gt_box, matching_classes_oh,gt_boxes_grid = \
    next(train_gen)
img, detector_mask, matching_gt_box, matching_classes_oh, gt_boxes_grid = \
    img[0], detector_mask[0], matching_gt_box[0], matching_classes_oh[0], gt_boxes_grid[0]

fig,(ax1,ax2) = plt.subplots(2,figsize=(5,10))
ax1.imshow(img)
# [16,16,5,1] => [16,16,1]
mask = tf.reduce_sum(detector_mask, axis=2)
ax2.matshow(mask[...,0]) # [16,16]


# %%
from tensorflow.keras import layers


import  tensorflow.keras.backend as K 

class SpaceToDepth(layers.Layer):

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        super(SpaceToDepth, self).__init__(**kwargs)

    def call(self, inputs):
        x = inputs
        batch, height, width, depth = K.int_shape(x)
        batch = -1
        reduced_height = height // self.block_size
        reduced_width = width // self.block_size
        y = K.reshape(x, (batch, reduced_height, self.block_size,
                             reduced_width, self.block_size, depth))
        z = K.permute_dimensions(y, (0, 1, 3, 2, 4, 5))
        t = K.reshape(z, (batch, reduced_height, reduced_width, depth * self.block_size **2))
        return t

    def compute_output_shape(self, input_shape):
        shape =  (input_shape[0], input_shape[1] // self.block_size, input_shape[2] // self.block_size,
                  input_shape[3] * self.block_size **2)
        return tf.TensorShape(shape)


# 3.1
input_image = layers.Input((IMGSZ,IMGSZ, 3), dtype='float32')

# unit1
x = layers.Conv2D(32, (3,3), strides=(1,1),padding='same', name='conv_1', use_bias=False)(input_image)
x = layers.BatchNormalization(name='norm_1')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

x = layers.MaxPooling2D(pool_size=(2,2))(x)

# unit2
x = layers.Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2',use_bias=False)(x)
x = layers.BatchNormalization(name='norm_2')(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)


# Layer 3
x = layers.Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_3')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 4
x = layers.Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_4')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 5
x = layers.Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_5')(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

# Layer 6
x = layers.Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_6')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 7
x = layers.Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_7')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 8
x = layers.Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_8')(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

# Layer 9
x = layers.Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_9')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 10
x = layers.Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_10')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 11
x = layers.Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_11')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 12
x = layers.Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_12')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 13
x = layers.Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_13')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# for skip connection
skip_x = x  # [b,32,32,512]


x = layers.MaxPooling2D(pool_size=(2, 2))(x)

# Layer 14
x = layers.Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_14')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 15
x = layers.Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_15')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 16
x = layers.Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_16')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 17
x = layers.Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_17')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 18
x = layers.Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_18')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 19
x = layers.Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_19')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 20
x = layers.Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_20')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 21
skip_x = layers.Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_x)
skip_x = layers.BatchNormalization(name='norm_21')(skip_x)
skip_x = layers.LeakyReLU(alpha=0.1)(skip_x)

skip_x = SpaceToDepth(block_size=2)(skip_x)
 
 # concat
 # [b,16,16,1024], [b,16,16,256],=> [b,16,16,1280]
x = tf.concat([skip_x, x], axis=-1)

# Layer 22
x = layers.Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_22')(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Dropout(0.5)(x) # add dropout
# [b,16,16,5,7] => [b,16,16,35]

x = layers.Conv2D(5*7, (1,1), strides=(1,1), padding='same', name='conv_23')(x)

output = layers.Reshape((GRIDSZ,GRIDSZ,5,7))(x)
# create model
model = keras.models.Model(input_image, output)
x = tf.random.normal((4,512,512,3))
out = model(x)
print('out:', out.shape)

# %%
# 3.2
class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4
weight_reader = WeightReader('yolo.weights')


weight_reader.reset()
nb_conv = 23

for i in range(1, nb_conv + 1):
    conv_layer = model.get_layer('conv_' + str(i))
    conv_layer.trainable = True

    if i < nb_conv:
        norm_layer = model.get_layer('norm_' + str(i))
        norm_layer.trainable = True

        size = np.prod(norm_layer.get_weights()[0].shape)

        beta = weight_reader.read_bytes(size)
        gamma = weight_reader.read_bytes(size)
        mean = weight_reader.read_bytes(size)
        var = weight_reader.read_bytes(size)

        weights = norm_layer.set_weights([gamma, beta, mean, var])

    if len(conv_layer.get_weights()) > 1:
        bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2, 3, 1, 0])
        conv_layer.set_weights([kernel, bias])
    else:
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2, 3, 1, 0])
        conv_layer.set_weights([kernel])


layer = model.layers[-2]  # last convolutional layer
# print(layer.name)
layer.trainable = True

weights = layer.get_weights()

new_kernel = np.random.normal(size=weights[0].shape) / (GRIDSZ * GRIDSZ)
new_bias = np.random.normal(size=weights[1].shape) / (GRIDSZ * GRIDSZ)

layer.set_weights([new_kernel, new_bias])

#%%

# model.load_weights('G:\\yolov2-tf2\\weights\\ckpt.h5')

#%%
img, detector_mask, matching_gt_boxes, matching_classes_oh, gt_boxes_grid = next(train_gen)

img, detector_mask, matching_gt_boxes, matching_classes_oh, gt_boxes_grid = \
    img[0], detector_mask[0], matching_gt_boxes[0], matching_classes_oh[0], gt_boxes_grid[0]

# [b,512,512,3]=>[b,16,16,5,7]=>[16,16,5,x-y-w-h-conf-l1-l2]
y_pred = model(tf.expand_dims(img, axis=0))[0][...,4]
# [16,16,5] => [16,16]
y_pred = tf.reduce_sum(y_pred,axis=2)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.imshow(img)
# [16,16,5,1]=>[16,16]
ax2.matshow(tf.reduce_sum(detector_mask,axis=2)[...,0])
ax3.matshow(y_pred)


# %%
from tensorflow.keras import losses

def compute_iou(x1,y1,w1,h1, x2,y2,w2,h2):
    # x1...:[b,16,16,5]
    xmin1 = x1 - 0.5*w1
    xmax1 = x1 + 0.5*w1
    ymin1 = y1 - 0.5*h1 
    ymax1 = y1 + 0.5*h1 

    xmin2 = x2 - 0.5*w2
    xmax2 = x2 + 0.5*w2
    ymin2 = y2 - 0.5*h2
    ymax2 = y2 + 0.5*h2

    # (xmin1,ymin1,xmax1,ymax1) (xmin2,ymin2,xmax2,ymax2)
    interw = np.minimum(xmax1,xmax2) - np.maximum(xmin1,xmin2)
    interh = np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2)
    inter = interw * interh
    union = w1*h1 +w2*h2 - inter 
    iou = inter / (union + 1e-6)
    # [b,16,16,5]
    return iou 

#%%
# 4.1 coordinate loss
def yolo_loss(detector_mask, matching_gt_boxes, matching_classes_oh, gt_boxes_grid, y_pred):
    # detector_mask: [b,16,16,5,1]
    # matching_gt_boxes: [b,16,16,5,5] x-y-w-h-l
    # matching_classes_oh: [b,16,16,5,2] l1-l2
    # gt_boxes_grid: [b,40,5] x-y-wh-l
    # y_pred: [b,16,16,5,7] x-y-w-h-conf-l0-l1

    anchors = np.array(ANCHORS).reshape(5,2)

    # create starting position for each grid anchors
    # [16,16]
    x_grid = tf.tile(tf.range(GRIDSZ),[GRIDSZ])
    # [1,16,16,1,1]
    # [b,16,16,5,2]
    x_grid = tf.reshape(x_grid, (1,GRIDSZ,GRIDSZ,1,1))
    x_grid = tf.cast(x_grid, tf.float32)
    # [b,16_1,16_2,1,1]=>[b,16_2,16_1,1,1]
    y_grid = tf.transpose(x_grid, (0,2,1,3,4))
    xy_grid = tf.concat([x_grid, y_grid],axis=-1)
    # [1,16,16,1,2]=> [b,16,16,5,2]
    xy_grid = tf.tile(xy_grid, [y_pred.shape[0], 1,1,5,1])

    # [b,16,16,5,7] x-y-w-h-conf-l1-l2
    pred_xy = tf.sigmoid(y_pred[...,0:2])
    pred_xy = pred_xy + xy_grid
    # [b,16,16,5,2]
    pred_wh = tf.exp(y_pred[...,2:4])
    # [b,16,16,5,2] * [5,2] => [b,16,16,5,2]
    pred_wh = pred_wh * anchors

    n_detector_mask = tf.reduce_sum(tf.cast(detector_mask>0., tf.float32))
    # [b,16,16,5,1] * [b,16,16,5,2]
    #
    xy_loss = detector_mask * tf.square(matching_gt_boxes[...,:2]-pred_xy) / (n_detector_mask+1e-6)
    xy_loss  = tf.reduce_sum(xy_loss)
    wh_loss = detector_mask * tf.square(tf.sqrt(matching_gt_boxes[...,2:4])-\
    	tf.sqrt(pred_wh)) / (n_detector_mask+1e-6)
    wh_loss  = tf.reduce_sum(wh_loss)

    # 4.1 coordinate loss
    coord_loss = xy_loss + wh_loss 

    # 4.2 class loss
    # [b,16,16,5,2]
    pred_box_class = y_pred[...,5:]
    # [b,16,16,5]
    true_box_class = tf.argmax(matching_classes_oh,-1)
    # [b,16,16,5] vs [b,16,16,5,2]
    class_loss = losses.sparse_categorical_crossentropy(\
        true_box_class, pred_box_class, from_logits=True)
    # [b,16,16,5] => [b,16,16,5,1]* [b,16,16,5,1]
    class_loss = tf.expand_dims(class_loss,-1) * detector_mask
    class_loss = tf.reduce_sum(class_loss) / (n_detector_mask+1e-6)
 

    # 4.3 object loss
    # nonobject_mask
    # iou done!
    # [b,16,16,5]
    x1,y1,w1,h1 = matching_gt_boxes[...,0],matching_gt_boxes[...,1],\
        matching_gt_boxes[...,2],matching_gt_boxes[...,3]
    # [b,16,16,5]
    x2,y2,w2,h2 = pred_xy[...,0],pred_xy[...,1],pred_wh[...,0],pred_wh[...,1]
    ious = compute_iou(x1,y1,w1,h1, x2,y2,w2,h2)
    # [b,16,16,5,1]
    ious = tf.expand_dims(ious, axis=-1)

    # [b,16,16,5,1]
    pred_conf = tf.sigmoid(y_pred[...,4:5])
    # [b,16,16,5,2] => [b,16,16,5, 1, 2]
    pred_xy = tf.expand_dims(pred_xy, axis=4)
    # [b,16,16,5,2] => [b,16,16,5, 1, 2]
    pred_wh = tf.expand_dims(pred_wh, axis=4)
    pred_wh_half = pred_wh /2. 
    pred_xymin = pred_xy - pred_wh_half
    pred_xymax = pred_xy + pred_wh_half

    # [b, 40, 5] => [b, 1, 1, 1, 40, 5]
    true_boxes_grid = tf.reshape(gt_boxes_grid, \
        [gt_boxes_grid.shape[0], 1, 1 ,1, gt_boxes_grid.shape[1], gt_boxes_grid.shape[2]])
    true_xy = true_boxes_grid[...,0:2]
    true_wh = true_boxes_grid[...,2:4]
    true_wh_half = true_wh /2. 
    true_xymin = true_xy - true_wh_half
    true_xymax = true_xy + true_wh_half
    # predxymin, predxymax, true_xymin, true_xymax
    # [b,16,16,5,1,2] vs [b,1,1,1,40,2]=> [b,16,16,5,40,2]
    intersectxymin = tf.maximum(pred_xymin, true_xymin)
    # [b,16,16,5,1,2] vs [b,1,1,1,40,2]=> [b,16,16,5,40,2]
    intersectxymax = tf.minimum(pred_xymax, true_xymax)
    # [b,16,16,5,40,2]
    intersect_wh = tf.maximum(intersectxymax - intersectxymin, 0.)
    # [b,16,16,5,40] * [b,16,16,5,40]=>[b,16,16,5,40]
    intersect_area = intersect_wh[...,0] * intersect_wh[...,1]
    # [b,16,16,5,1]
    pred_area = pred_wh[...,0] * pred_wh[...,1]
    # [b,1,1,1,40]
    true_area = true_wh[...,0] * true_wh[...,1]
    # [b,16,16,5,1]+[b,1,1,1,40]-[b,16,16,5,40]=>[b,16,16,5,40]
    union_area = pred_area + true_area - intersect_area
    # [b,16,16,5,40]
    iou_score = intersect_area / union_area
    # [b,16,16,5]
    best_iou = tf.reduce_max(iou_score, axis=4)
    # [b,16,16,5,1]
    best_iou = tf.expand_dims(best_iou, axis=-1)

    nonobj_detection = tf.cast(best_iou<0.6, tf.float32)
    nonobj_mask = nonobj_detection * (1-detector_mask)
    # nonobj counter
    n_nonobj = tf.reduce_sum(tf.cast(nonobj_mask>0.,tf.float32))

    nonobj_loss = tf.reduce_sum(nonobj_mask * tf.square(-pred_conf))\
        /(n_nonobj+1e-6)
    obj_loss = tf.reduce_sum(detector_mask * tf.square(ious - pred_conf))\
         / (n_detector_mask+1e-6)

    loss = coord_loss + class_loss + nonobj_loss + 5 * obj_loss

    return loss, [nonobj_loss + 5 * obj_loss, class_loss, coord_loss]



#%%
img, detector_mask, matching_gt_boxes, matching_classes_oh, gt_boxes_grid = next(train_gen)

img, detector_mask, matching_gt_boxes, matching_classes_oh, gt_boxes_grid = \
    img[0], detector_mask[0], matching_gt_boxes[0], matching_classes_oh[0], gt_boxes_grid[0]

y_pred = model(tf.expand_dims(img, axis=0))[0]

loss, sub_loss =  yolo_loss(tf.expand_dims(detector_mask,axis=0), 
tf.expand_dims(matching_gt_boxes,axis=0), 
tf.expand_dims(matching_classes_oh,axis=0), 
tf.expand_dims(gt_boxes_grid,axis=0), 
tf.expand_dims(y_pred,axis=0)
)



# %%
# 5.1 train
val_db = get_dataset('data/val/image', 'data/val/annotation', 4)
val_gen = ground_truth_generator(val_db)

def train(epoches):
    optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9,
        beta_2=0.999,epsilon=1e-08)

    for epoch in range(epoches):

        for step in range(30):
            img, detector_mask, matching_true_boxes, matching_classes_oh, true_boxes = next(train_gen)
            with tf.GradientTape() as tape:
                y_pred = model(img, training=True)
                loss, sub_loss = yolo_loss(detector_mask, \
                    matching_true_boxes, matching_classes_oh, \
                    true_boxes, y_pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            print(epoch, step, float(loss), float(sub_loss[0]), float(sub_loss[1]), float(sub_loss[2]))

        
#%%
train(10)
model.save_weights('weights/epoch10.ckpt')
# coordinate loss
# batch size
# val float32
# print sub loss
# comment ckpt.h5
# get_dataset use parameters
# add save models







# %%
# model.load_weights('weights/ckpt.h5')

import cv2
# 5.2
def visualize_result(img, model):
    # [512,512,3] 0~255, BGR
    img = cv2.imread(img)
    img = img[...,::-1]/255. 
    img = tf.cast(img, dtype=tf.float32)
    # [1,512,512,3]
    img = tf.expand_dims(img, axis=0)
    # [1,16,16,5,7]
    y_pred = model(img, training=False)

    x_grid = tf.tile(tf.range(GRIDSZ), [GRIDSZ])
    # [1, 16,16,1,1]
    x_grid = tf.reshape(x_grid, (1, GRIDSZ, GRIDSZ, 1, 1))
    x_grid = tf.cast(x_grid, dtype=tf.float32)
    y_grid = tf.transpose(x_grid, (0,2,1,3,4))
    xy_grid = tf.concat([x_grid,y_grid], axis=-1)
    # [1, 16, 16, 5, 2]
    xy_grid = tf.tile(xy_grid, [1, 1, 1, 5, 1])

    anchors = np.array(ANCHORS).reshape(5,2)
    pred_xy = tf.sigmoid(y_pred[...,0:2])
    pred_xy = pred_xy + xy_grid
    # normalize 0~1
    pred_xy = pred_xy / tf.constant([16.,16.])

    pred_wh = tf.exp(y_pred[...,2:4])
    pred_wh = pred_wh * anchors
    pred_wh = pred_wh / tf.constant([16.,16.])

    # [1,16,16,5,1]
    pred_conf = tf.sigmoid(y_pred[...,4:5])
    # l1 l2
    pred_prob = tf.nn.softmax(y_pred[...,5:])

    pred_xy, pred_wh, pred_conf, pred_prob = \
        pred_xy[0], pred_wh[0], pred_conf[0], pred_prob[0]

    boxes_xymin = pred_xy - 0.5 * pred_wh
    boxes_xymax = pred_xy + 0.5 * pred_wh
    # [16,16,5,2+2]
    boxes = tf.concat((boxes_xymin, boxes_xymax),axis=-1)
    # [16,16,5,2]
    box_score = pred_conf * pred_prob
    # [16,16,5]
    box_class = tf.argmax(box_score, axis=-1)
    # [16,16,5]
    box_class_score = tf.reduce_max(box_score, axis=-1)
    # [16,16,5]
    pred_mask = box_class_score > 0.45
    # [16,16,5,4]=> [N,4]
    boxes = tf.boolean_mask(boxes, pred_mask)
    # [16,16,5] => [N]
    scores = tf.boolean_mask(box_class_score, pred_mask)
    # 【16,16，5】=> [N]
    classes = tf.boolean_mask(box_class, pred_mask)

    boxes = boxes * 512. 
    # [N] => [n]
    select_idx = tf.image.non_max_suppression(boxes, scores, 40, iou_threshold=0.3)
    boxes = tf.gather(boxes, select_idx)
    scores = tf.gather(scores, select_idx)
    classes = tf.gather(classes, select_idx)

    # plot 
    fig, ax = plt.subplots(1, figsize=(10,10))
    ax.imshow(img[0])
    n_boxes = boxes.shape[0]
    ax.set_title('boxes:%d'%n_boxes)
    for i in range(n_boxes):
        x1,y1,x2,y2 = boxes[i]
        w = x2 - x1 
        h = y2 - y1 
        label = classes[i].numpy()

        if label==0: # sugarweet
            color = (0,1,0)
        else:
            color = (1,0,0)

        rect = patches.Rectangle((x1.numpy(), y1.numpy()), w.numpy(), h.numpy(), linewidth = 3, edgecolor=color,facecolor='none')
        ax.add_patch(rect)



#%%
files = glob.glob('data/val/image/*.png')
for x in files:
    visualize_result(x, model)
plt.show()



