import tensorflow as tf
import numpy as np
import os
import vgg
import cv2
import timestr

from utils import write_image, check_image, check_resize, preprocess, maybe_make_directory

class StyleTransfer:

    def __init__(self, args, sess):
        self.sess = sess

        self.content_img = args.content_img
        self.style_imgs = args.style_imgs
        self.init_img_type = args.init_img_type

        self.content_weight = args.content_weight
        self.style_weight = args.style_weight
        self.tv_weight = args.tv_weight

        self.content_layers = args.content_layers
        self.style_layers = args.style_layers
        self.style_layer_weights = args.style_layer_weights
        self.content_layer_weights = args.content_layer_weights
        self.style_imgs_weights = args.style_imgs_weights

        self.optimizer = args.optimizer
        self.model_weights = args.model_weights
        self.pooling_type = args.pooling_type

        self.img_output_dir = args.img_output_dir
        self.img_name = args.img_name
        self.content_img_dir = args.content_img_dir
        self.style_imgs_dir = args.style_imgs_dir

        self.verbose = args.verbose
        self.write_iterations_adam = args.write_iterations_adam
        self.max_size = args.max_size
        self.max_iterations = args.max_iterations
        self.print_iterations = args.print_iterations
        self.learning_rate = args.learning_rate

        self._build_graph()

    def _initialize_images(self):
        self.content_img = self.get_content_image()
        self.style_imgs = self.get_style_images()

        # get initial image to compute on the network
        def get_init_img(init_img_type, content_img):
            if init_img_type == 'content':
                return content_img
        self.init_img = get_init_img(self.init_img_type, self.content_img)

    def _build_graph(self):
        # build model
        self.net = vgg.VGG19(model_weights=self.model_weights,
                             pooling_type=self.pooling_type,
                             verbose=self.verbose)

        self._initialize_images()
        self.net = self.net.build_model(self.content_img)

        style_loss = self.sum_style_loss()
        content_loss = self.sum_content_loss()
        # total variation denoising
        tv_loss = tf.image.total_variation(self.net['input'])

        alpha = self.content_weight
        beta = self.style_weight
        theta = self.tv_weight

        # linear combination between the loss components
        self.total_loss = alpha * content_loss + beta * style_loss + theta * tv_loss

    def update(self):
        optimizer = self.get_optimizer()

        if self.optimizer == 'adam':
            self.minimize_with_adam(optimizer)
        elif self.optimizer == 'lbfgs':
            self.minimize_with_lbfgs(optimizer)

        output_img = self.sess.run(self.net['input'])
        return output_img, self.content_img, self.style_imgs, self.init_img

    def get_content_image(self):
        path = os.path.join(self.content_img_dir, self.content_img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        check_image(img, path)
        img = img.astype(np.float32)
        img = check_resize(img, self.max_size)
        img = preprocess(img)
        return img

    def get_style_images(self):
        _, ch, cw, cd = self.content_img.shape
        style_imgs = []
        for style_fn in self.style_imgs:
            path = os.path.join(self.style_imgs_dir, style_fn)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            check_image(img, path)
            img = img.astype(np.float32)
            img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
            img = preprocess(img)
            style_imgs.append(img)
        return style_imgs

    def sum_style_loss(self):
        """
        Computes the weighted sum of the means losses of each style image

        Returns:
        - total_style_loss: Value of the total style loss
        """
        total_style_loss = sum([self._mean_loss(img, lambda a,x: style_layer_loss(a,x), self.style_layers, self.style_layer_weights) *
                                self.style_imgs_weights for img in self.style_imgs])
        total_style_loss /= float(len(self.style_imgs))
        return total_style_loss

    def sum_content_loss(self):
        """
        Computes the mean loss of the content image

        Returns:
        - content_loss: Value of the content loss
        """
        content_loss = self._mean_loss(self.content_img, lambda p,x: content_layer_loss(p,x), self.content_layers, self.content_layer_weights)
        return content_loss

    def _mean_loss(self, img, func, layers, layer_weights):
        """
        Computes the mean loss of all the returned values from the loss function
        using the original image, layers, and layers weights 

        Inputs:
        - img: Original image
        - func: Function to be used to compute the loss
        - layers: Layers used for the content or style image
        - layers_weights: Weighting factors of the contribution of each layer to the loss

        Returns:
        - loss: Value of the mean loss
        """
        self.sess.run(self.net['input'].assign(img))
        loss = 0
        for layer in layers:
            a = self.sess.run(self.net[layer])
            x = self.net[layer]
            a = tf.convert_to_tensor(a)
            loss += func(a,x) * layer_weights
        loss /= float(len(layers))
        return loss

    def minimize_with_lbfgs(self, optimizer):
        if self.verbose: 
            print('\nMINIMIZING LOSS USING: L-BFGS OPTIMIZER')
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.sess.run(self.net['input'].assign(self.init_img))
        optimizer.minimize(self.sess)

    def minimize_with_adam(self, optimizer):
        if self.verbose: 
            print('\nMINIMIZING LOSS USING: ADAM OPTIMIZER')
        train_op = optimizer.minimize(self.total_loss)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.sess.run(self.net['input'].assign(self.init_img))

        if self.write_iterations_adam:
            out_dir = os.path.join(self.img_output_dir, self.img_name, timestr.get_time())
            maybe_make_directory(out_dir)

        for iterations in range(self.max_iterations):
            self.sess.run(train_op)

            # write image at every iteration
            if self.write_iterations_adam:
                img_path = os.path.join(out_dir, self.img_name+str(iterations)+'.png')
                output_img = self.sess.run(self.net['input'])
                write_image(img_path, output_img)
            if iterations % self.print_iterations == 0 and self.verbose:
                curr_loss = self.total_loss.eval()
                print("At iterate {}\tf= {}".format(iterations, curr_loss[0]))

    def get_optimizer(self):
        print_iterations = self.print_iterations if self.verbose else 0
        if self.optimizer == 'lbfgs':
            return tf.contrib.opt.ScipyOptimizerInterface(
                self.total_loss, method='L-BFGS-B', options={'maxiter': self.max_iterations,
                                                             'disp': print_iterations})
        elif self.optimizer == 'adam':
            return tf.train.AdamOptimizer(self.learning_rate)

def content_layer_loss(p, x):
    """
    Computes the content layer loss

    Inputs:
    - p: 4D tensor of the original image
    - x: 4D tensor of the generated image

    Returns:
    - loss: Value of the content layer loss
    """
    _, h, w, d = p.get_shape()  
    M = h.value * w.value       # area of the feature map
    N = d.value                 # depth

    # content loss function
    K = 1. / (2. * N**0.5 * M**0.5)

    # mean squared distance between the two feature representations
    loss = K * tf.reduce_sum(tf.pow((x - p), 2))
    return loss

def style_layer_loss(a, x):
    """
    Computes the style layer loss

    Inputs:
    - a: 4D tensor of the original image
    - x: 4D tensor of the generated image

    Returns:
    - loss: Value of the style layer loss
    """
    _, h, w, d = a.get_shape() 
    M = h.value * w.value       # area of the feature map
    N = d.value                 # depth

    A = gram_matrix(a)
    G = gram_matrix(x)

    # style loss function   
    K = 1. / (4 * N**2 * M**2)

    # mean squared distance between the Gram matrices of style and generated images
    loss =  K * tf.reduce_sum(tf.pow((G - A), 2))
    return loss

def gram_matrix(tensor):
    """
    Computes the inner products of the vectorized feature maps at a layer.
    This captures feature correlations activations corresponding to texture information

    Input:
    - tensor: 4D Tensor 

    Returns:
    - G: Tensor of shape NxN where N equals to the depth of the inputted tensor
    """
    _,_,_,N = tensor.get_shape()

    # Reshape into a 2D tensor
    V = tf.reshape(tensor, shape=[-1, int(N)])
    G = tf.matmul(tf.transpose(V), V)
    return G