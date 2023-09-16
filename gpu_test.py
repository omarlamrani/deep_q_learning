import tensorflow as tf
import timeit
import warnings

warnings.filterwarnings('ignore')

setup = "import tensorflow as tf"

print('First test to check computation time...\n')

with tf.device('/cpu:0'):
    a_c = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a-cpu')
    b_c = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b-cpu')
    c_c = "tf.matmul(tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a-cpu')" \
          ",tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b-cpu') , name='c-cpu')"

with tf.device('/gpu:0'):
    a_g = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a-gpu')
    b_g = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b-gpu')
    c_g = "tf.matmul(tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a-gpu')" \
          ", tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b-gpu'), name='c-gpu')"

with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as sess:
    print(f"--> Execution time for cpu is: {timeit.timeit(setup=setup, stmt=c_c, number=5000)}")
    print(f"--> Execution time for gpu is: {timeit.timeit(setup=setup, stmt=c_g, number=5000)}\n")

print('Second test to check if gpu is available and setup with CUDA...\n')

import torch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Device is: ",device)

