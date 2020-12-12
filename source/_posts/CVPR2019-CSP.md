---
title: High-level Semantic Feature Detection A New Perspective for Pedestrian Detection
date: 2019-05-29 10:16:27
description: CVPR-19 High-level Semantic Feature Detection A New Perspective for Pedestrian Detection
categories: Pedestrian Detection
tags:
- Pedestrian Detection
- Deep Learning
---

[Paper Link](https://arxiv.org/abs/1904.02948v1)

[Code](https://github.com/liuwei16/CSP)

# 简介

CSP: Center and Scale Prediction


![CVPR19_CSP_pipeline](./cVPR19_CSP_pipeline.png)

![20190529_CVPR19_CSP_architecture](./20190529_CVPR19_CSP_architecture.png)


![20190529_CVPR19_CSP_annotations](./20190529_CVPR19_CSP_annotations.png)


# 方法

# 实验

![20190529_CVPR19_CSP_points](./20190529_CVPR19_CSP_points.png)


![20190529_CVPR19_CSP_scale](./20190529_CVPR19_CSP_scale.png)

![20190529_CVPR19_CSP_downsampling_factors](./20190529_CVPR19_CSP_downsampling_factors.png)

![20190529_CVPR19_CSP_multi_scale](./20190529_CVPR19_CSP_multi_scale.png)

![20190529_CVPR19_CSP_Caltech_new](./20190529_CVPR19_CSP_Caltech_new.png)

![20190529_CVPR19_CSP_CityPersons](./20190529_CVPR19_CSP_CityPersons.png)


# 代码

## 准备ground truth

```python
def calc_gt_center(C, img_data,r=2, down=4,scale='h',offset=True):
	def gaussian(kernel):
		sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
		s = 2*(sigma**2)
		dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		return np.reshape(dx,(-1,1))
	gts = np.copy(img_data['bboxes'])
	igs = np.copy(img_data['ignoreareas'])
	scale_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 2))
	if scale=='hw':
		scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
	if offset:
		offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
	seman_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 3))
	seman_map[:,:,1] = 1
	if len(igs) > 0:
		igs = igs/down
		for ind in range(len(igs)):
			x1,y1,x2,y2 = int(igs[ind,0]), int(igs[ind,1]), int(np.ceil(igs[ind,2])), int(np.ceil(igs[ind,3]))
			seman_map[y1:y2, x1:x2,1] = 0  # 被忽视的区域在第１个通道上置０
	if len(gts)>0:
		gts = gts/down
		for ind in range(len(gts)):
			# x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
			x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
			c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
			dx = gaussian(x2-x1)
			dy = gaussian(y2-y1)
			gau_map = np.multiply(dy, np.transpose(dx))
			seman_map[y1:y2, x1:x2,0] = np.maximum(seman_map[y1:y2, x1:x2,0], gau_map)  # 在第０个通道上置高斯值
			seman_map[y1:y2, x1:x2,1] = 1  # 前景在第１个通道上置１
			seman_map[c_y, c_x, 2] = 1  # 在第２个通道上目标中心位置１

			if scale == 'h':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = 1
			elif scale=='w':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = 1
			elif scale=='hw':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 2] = 1
			if offset:
				offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
				offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
				offset_map[c_y, c_x, 2] = 1

	if offset:
		return seman_map,scale_map,offset_map
	else:
		return seman_map, scale_map
```

seman_map有三个位面，第一个是高斯值mask，第二个是学习权重，第三个是目标中心点的位置。

![20190529_CVPR19_CSP_seman_map0](./20190529_CVPR19_CSP_seman_map0.png)

![20190529_CVPR19_CSP_seman_map2](./20190529_CVPR19_CSP_seman_map2.png)

![20190529_CVPR19_CSP_scale_map0](./20190529_CVPR19_CSP_scale_map0.png)



## 网络结构
```python
def nn_p3p4p5(img_input=None, offset=True, num_scale=1, trainable=False):
    bn_axis = 3
    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable=False)(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable=False)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable=False)
    stage2 = identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable=False)
    # print('stage2: ', stage2._keras_shape[1:])
    x = conv_block(stage2, 3, [128, 128, 512], stage=3, block='a', trainable=trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', trainable=trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', trainable=trainable)
    stage3 = identity_block(x, 3, [128, 128, 512], stage=3, block='d', trainable=trainable)
    # print('stage3: ', stage3._keras_shape[1:])
    x = conv_block(stage3, 3, [256, 256, 1024], stage=4, block='a', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', trainable=trainable)
    stage4 = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', trainable=trainable)
    # print('stage4: ', stage4._keras_shape[1:])
    x = conv_block(stage4, 3, [512, 512, 2048], stage=5, block='a', strides=(1, 1), dila=(2, 2),
                   trainable=trainable)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', dila=(2, 2), trainable=trainable)
    stage5 = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', dila=(2, 2), trainable=trainable)
    # print('stage5: ', stage5._keras_shape[1:])

    P3_up = Deconvolution2D(256, kernel_size=4, strides=2, padding='same',
                            kernel_initializer='glorot_normal', name='P3up', trainable=trainable)(stage3)
    # print('P3_up: ', P3_up._keras_shape[1:])
    P4_up = Deconvolution2D(256, kernel_size=4, strides=4, padding='same',
                            kernel_initializer='glorot_normal', name='P4up', trainable=trainable)(stage4)
    # print('P4_up: ', P4_up._keras_shape[1:])
    P5_up = Deconvolution2D(256, kernel_size=4, strides=4, padding='same',
                            kernel_initializer='glorot_normal', name='P5up', trainable=trainable)(stage5)
    # print('P5_up: ', P5_up._keras_shape[1:])

    P3_up = L2Normalization(gamma_init=10, name='P3norm')(P3_up)
    P4_up = L2Normalization(gamma_init=10, name='P4norm')(P4_up)
    P5_up = L2Normalization(gamma_init=10, name='P5norm')(P5_up)
    conc = Concatenate(axis=-1)([P3_up, P4_up, P5_up])

    feat = Convolution2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal', name='feat',
                         trainable=trainable)(conc)
    feat = BatchNormalization(axis=bn_axis, name='bn_feat')(feat)
    feat = Activation('relu')(feat)

    x_class = Convolution2D(1, (1, 1), activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability_onecls(probability=0.01),
                            name='center_cls', trainable=trainable)(feat)
    x_regr = Convolution2D(num_scale, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                           name='height_regr', trainable=trainable)(feat)

    if offset:
        x_offset = Convolution2D(2, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                                 name='offset_regr', trainable=trainable)(feat)
        return [x_class, x_regr, x_offset]
    else:
        return [x_class, x_regr]
```

## Loss

```python
def cls_center(y_true, y_pred):

	classification_loss = K.binary_crossentropy(y_pred[:, :, :, 0], y_true[:, :, :, 2])
	# firstly we compute the focal weight
	positives = y_true[:, :, :, 2]
	negatives = y_true[:, :, :, 1]-y_true[:, :, :, 2]
	foreground_weight = positives * (1.0 - y_pred[:, :, :, 0]) ** 2.0
	background_weight = negatives * ((1.0 - y_true[:, :, :, 0])**4.0)*(y_pred[:, :, :, 0] ** 2.0)

	focal_weight = foreground_weight + background_weight

	assigned_boxes = tf.reduce_sum(y_true[:, :, :, 2])
	class_loss = 0.01*tf.reduce_sum(focal_weight*classification_loss) / tf.maximum(1.0, assigned_boxes)
 assigned_boxes)

	return class_loss
```
