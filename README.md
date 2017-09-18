# Multi-Task Self-Supervised Resnet-v2-101

[Multi-Task Self-Supervised Visual Learning](https://arxiv.org/abs/1708.07860) uses several "self-supervised" tasks--i.e., supervised tasks for which data can be collected without manual labeling--to train a single network for vision.  Self-supervised learning is a type of unsupervised learning, and pre-training on self-supervised tasks can be a useful starting-point for practical vision tasks like classification, detection, and geometry estimation.  Several self-supervised tasks outperform the best reconstruction-based models (e.g. autoencoders) and generative models in these domains, and combining tasks makes performance even stronger.

This repository contains our best performing model, which was trained on four tasks: [patch relative position](http://graphics.cs.cmu.edu/projects/deepContext/), [colorization](http://richzhang.github.io/colorization/), [exemplar training](https://arxiv.org/abs/1406.6909), and [motion segmentation](https://arxiv.org/abs/1612.06370) (RP+Col+Ex+MS in the paper).

This is research, not an official Google product.

The checkpoint is available [here](TODO).

# Loading the model
This model is based on Resnet-v2-101 available in [tf-slim](https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v2.py).  To load it, first create a Resnet-v2-101 in slim:

```python
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.slim.nets import resnet_utils

with slim.arg_scope(resnet_utils.resnet_arg_scope(
        batch_norm_epsilon=1e-5, batch_norm_scale=True)):
  logits, activation_handles = resnet_v2.resnet_v2_101(
          image_tensor, scope = 'resnet_v2_101')
  representation = activation_handles['resnet_v2_101/block3']
```

Here, `image_tensor` is a batch of images, preprocessed in VGG-style (i.e. the images should be scaled between 0 and 255, and then a per-channel mean value should be subtracted; we subtracted RGB `[123.68, 116.779, 103.939]` as computed from ImageNet).  Note that the procedure only trains Resnet-v2-101 through block 3; hence the final line selects the block 3 output and discards later layers.

To load the checkpoint, you will most likely only load the variables that exist both in the checkpoint and in your graph.  You can do this with slim as follows:

```
import tensorflow as tf
from tensorflow.contrib.framework.python.framework import checkpoint_utils

var_names=[x[0] for x in checkpoint_utils.list_variables(ckpt_file)]
var_list=([v for v in tf.global_variables()
           if v.name[:-2].encode('ascii','ignore') in var_names])
variables_to_restore = {var.op.name:var for var in var_list}
restore_op, feed_dict = slim.assign_from_checkpoint(init_ckpt,
                                                    variables_to_restore)
```

Now `restore_op` can be run with the associated `feed_dict` to load the variables.
