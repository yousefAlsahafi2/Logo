
# coding: utf-8

# In[6]:

caffe_root = '/home/yalsahaf/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
# Your Chose to use CPU or GPU
#caffe.set_device(0)
#caffe.set_mode_gpu()
caffe.set_mode_cpu()

import numpy as np
from pylab import *
#get_ipython().magic(u'matplotlib inline')
import tempfile

# set display defaults
#plt.rcParams['figure.figsize'] = (10, 10)        # large images
#plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
#plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

def deprocess_net_image(image):
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image

import os
weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
assert os.path.exists(weights)


# Load ImageNet labels to imagenet_labels
imagenet_label_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
imagenet_labels = list(np.loadtxt(imagenet_label_file, str, delimiter='\t'))
#print 'Loaded ImageNet labels:\n', '\n'.join(imagenet_labels[:10] + ['...'])

# Load logo labels to logo_labels
logo_label_file = caffe_root + '/examples/FlickrLogos-v2/label.txt'
logo_labels = list(np.loadtxt(logo_label_file, str, delimiter='\t'))
#print 'Loaded logo labels:\n', '\n'.join(logo_labels[:10] + ['...'])

from caffe import layers as L
from caffe import params as P

# initializes weight and bias
weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)

learned_param = [weight_param, bias_param]
# frozen_param that means when we do not want any layer to run we froze its parameters
frozen_param = [dict(lr_mult=0)] * 2



def conv_relu(bottom,      ks,           nout,            stride=1,               pad=0, 
              group=1, param=learned_param, 
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
        
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,num_output=nout, pad=pad, group=group, param=param, 
                         
                         weight_filler=weight_filler,
                         bias_filler=bias_filler)
                    
    return conv, L.ReLU(conv, in_place=True)
    

def fc_relu(bottom, nout, param=learned_param,
            weight_filler =dict(type='gaussian', std=0.005), bias_filler=dict(type='constant', value=0.1)):
            
    fc = L.InnerProduct(bottom, num_output=nout, param=param, weight_filler=weight_filler,   bias_filler=bias_filler)
    
    return fc, L.ReLU(fc, in_place=True)


def max_pool(bottom, ks, stride=1):
    
    # return the result dircotly
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)



def caffenet(data, label=None, train=True, num_classes=1000, classifier_name='fc8', learn_all=False):
    
    """Returns a NetSpec specifying CaffeNet, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""

    n = caffe.NetSpec()
    n.data = data
    
    # So here we decided if we want to learn the layer of or stop the layer py frozen_param
    param = learned_param if learn_all else frozen_param
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=param)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)

    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
        
    else:
        fc7input = n.relu6
        
    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
    
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7
    
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)

    n.__setattr__(classifier_name, fc8)

    if not train:
        n.probs = L.Softmax(fc8)
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc = L.Accuracy(fc8, n.label)
        
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(n.to_proto()))
        return f.name

 

dummy_data = L.DummyData(shape=dict(dim=[1, 3, 200, 200]))
imagenet_net_filename = caffenet(data=dummy_data, train=False) # When train=False is a test
imagenet_net = caffe.Net(imagenet_net_filename, weights, caffe.TEST)


def logo_net(train=True, learn_all=False, subset=None):
    
    if subset is None:
        subset = 'train' if train else 'test'
        print subset

    source ='/home/yalsahaf/caffe/examples/FlickrLogos-v2/%s.txt' % subset
    print source 
    transform_param = dict(mirror=train, crop_size=227,mean_file=caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto')
    
    logo_data, logo_label = L.ImageData( transform_param=transform_param, source=source,
                                        batch_size= 1, new_height=20, new_width=20, ntop=2)
    
    
    return caffenet(data=logo_data, label=logo_label, train=train,
                    num_classes=32,
                    classifier_name='logo_fc8',
                    learn_all=learn_all)


# In[7]:

from caffe.proto import caffe_pb2
def solver(train_net_path, test_net_path=None, base_lr=0.001):
   
    s = caffe_pb2.SolverParameter()
    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 1000  # Test after every 1000 training iterations.
        s.test_iter.append(100) # Test on 100 batches each time we test.
    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1
    s.max_iter = 100000     # # of times to update the net (training iterations)
    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'
    # Set the initial learning rate for SGD.
    s.base_lr = base_lr
    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 20000
    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4
    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000
    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 10000
    s.snapshot_prefix = caffe_root + 'models/finetune_flickr_style/finetune_flickr_style'
    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    # Write the solver to a temporary file and return its filename.
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(s))
        return f.name


# In[8]:

def run_solvers(niter, solvers, disp_interval=10):
        
    blobs = ('loss', 'acc')
    #solvers = [('pretrained', logo_solver),
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers} for _ in blobs)
             
    for it in range(niter):      # for 200 times we do 
        for name, s in solvers: # once for 'pretrained', logo_solver and second one for ('scratch', scratch_logo_solver)
            
        #Then, you can perform a solver iteration, that is, a forward/backward pass with weight update, typing just
        #solver.step(1)
        #or run the solver until the last iteration, with
        #solver.solve()
        
            s.solve()  
            #s.step(1)  # run a single SGD step in Caffe
            
            loss[name][it],      acc[name][it] = (s.net.blobs[b].data.copy() for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it])) for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)   
        
    # Save the learned weights from both nets.
    #weight_dir = tempfile.mkdtemp()
    # if you want to save weight in spcific dirctiory
    weight_dir = '/home/yalsahaf/caffe/examples/FlickrLogos-v2/'
    weights = {}
    
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    return loss, acc, weights


# In[9]:

def eval_logo_net(weights, test_iters=10):
    test_net = caffe.Net(logo_net(train=False), weights, caffe.TEST)
    accuracy = 0                
    for it in xrange(test_iters):
        accuracy += test_net.forward()['acc']
        
    accuracy /= test_iters
    
    return test_net, accuracy


# In[ ]:

niter = 200  # number of iterations to train

end_to_end_net = logo_net(train=True, learn_all=True)



base_lr = 0.001
# We give the solver end_to_end Network with learning rate 0.001
logo_solver_filename = solver(end_to_end_net, base_lr=base_lr)
logo_solver = caffe.get_solver(logo_solver_filename)
#logo_solver.net.copy_from(logo_weights)
logo_solver.net.copy_from(weights)

print 'Running solvers for %d iterations...' % niter

solvers = [('end-to-end', logo_solver)]
_, _, finetuned_weights = run_solvers(niter, solvers)
print 'Done.'


logo_weights_ft = finetuned_weights['end-to-end']

test_net, accuracy = eval_logo_net(logo_weights_ft)
print 'Accuracy, finetuned from ImageNet initialization: %3.1f%%' % (100*accuracy, )


# In[7]:

def disp_preds(net, image, labels, k=5, name='ImageNet'):
    
    input_blob = net.blobs['data']                     # we put net data into input_blob
    net.blobs['data'].data[0, ...] = image             # insert the image to data
    probs = net.forward(start='conv1')['probs'][0]     # run net form conv1 to end and reutn probs
    top_k = (-probs).argsort()[:k]                     # sort them and choose best 5
    
    print 'top %d predicted %s labels =' % (k, name)
    print '\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
                    for i, p in enumerate(top_k))
    
def disp_logo_preds(net, image):
    disp_preds(net, image, logo_labels, name='logo')


# In[10]:

logo_data_batch = logo_solver.net.blobs['data'].data.copy()
logo_label_batch = np.array(logo_solver.net.blobs['label'].data, dtype=np.int32)

batch_index = 8
#image = logo_data_batch[batch_index]
#plt.imshow(deprocess_net_image(image))
print 'actual label =', logo_labels[logo_label_batch[batch_index]] 

disp_logo_preds(test_net, image)


# In[ ]:



