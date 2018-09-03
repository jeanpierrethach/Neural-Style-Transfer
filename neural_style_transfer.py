import tensorflow as tf

import time
import os
import errno
import argparse

import style_transfer
from utils import maybe_make_directory, normalize, write_image_output, check_model

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--meta_name', type=str,
                        default='meta_data.txt',
                        help='Configuration file output')

    parser.add_argument('--content_img', type=str,
                        required=True,
                        help='Filename of the content image (example: lion.jpg)')

    parser.add_argument('--content_img_dir', type=str,
                        default='./image_input',
                        help='Directory path to the content image. (default: %(default)s)')

    parser.add_argument('--style_imgs', nargs='+', type=str,
                        required=True,
                        help='Filenames of the style images (example: starry-night.jpg)')

    parser.add_argument('--style_imgs_weights', nargs='+', type=float,
                        default=[1.0],
                        help='Interpolation weights of each of the style images. (example: 0.5 0.5)')

    parser.add_argument('--style_imgs_dir', type=str,
                        default='./styles',
                        help='Directory path to the style images. (default: %(default)s)')

    parser.add_argument('--init_img_type', type=str,
                        default='content',
                        help='Image used to initialize the network. (default: %(default)s)')

    parser.add_argument('--max_size', type=int,
                        default=1920,
                        help='Maximum width or height of the input images. (default: %(default)s)')

    parser.add_argument('--content_weight', type=float,
                        default=5e0,
                        help='Weight for the content loss function. (default: %(default)s)')

    parser.add_argument('--style_weight', type=float,
                        default=1e4,
                        help='Weight for the style loss function. (default: %(default)s)')

    parser.add_argument('--tv_weight', type=float,
                        default=1e-3,
                        help='Weight for the total variational loss function. Set small (e.g. 1e-3). (default: %(default)s)')

    parser.add_argument('--content_layers', nargs='+', type=str,
                        default=['conv4_2'],
                        help='VGG19 layers used for the content image. (default: %(default)s)')

    parser.add_argument('--style_layers', nargs='+', type=str,
                        default=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
                        help='VGG19 layers used for the style image. (default: %(default)s)')

    parser.add_argument('--content_layer_weights', nargs='+', type=float,
                        default=[1.0],
                        help='Contributions (weights) of each content layer to loss. (default: %(default)s)')

    parser.add_argument('--style_layer_weights', nargs='+', type=float,
                        default=[0.2, 0.2, 0.2, 0.2, 0.2],
                        help='Contributions (weights) of each style layer to loss. (default: %(default)s)')

    parser.add_argument('--model_weights', type=str,
                        default='imagenet-vgg-verydeep-19.mat',
                        help='Weights and biases of the VGG-19 network.')

    parser.add_argument('--pooling_type', type=str,
                        default='avg', choices=['avg', 'max'],
                        help='Type of pooling in convolutional neural network. (default: %(default)s)')

    parser.add_argument('--device', type=str,
                        default='/gpu:0', choices=['/gpu:0', '/cpu:0'],
                        help='GPU or CPU mode.  GPU mode requires NVIDIA CUDA. (default|recommended: %(default)s)')

    parser.add_argument('--img_output_dir', type=str,
                        default='./image_output',
                        help='Relative or absolute directory path to output image and data.')

    parser.add_argument('--img_name', type=str,
                        default='result',
                        help='Filename of the output image.')

    parser.add_argument('--verbose', action='store_true',
                        help='Boolean flag indicating if statements should be printed to the console.')

    parser.add_argument('--write_iterations_adam', action='store_true',
                        help='Boolean flag indicating if output images should be written in every iteration under the Adam optimizer.')

    parser.add_argument('--optimizer', type=str,
                        default='lbfgs', choices=['lbfgs', 'adam'],
                        help='Loss minimization optimizer.  L-BFGS gives better results.  Adam uses less memory. (default|recommended: %(default)s)')

    parser.add_argument('--learning_rate', type=float,
                        default=1e0,
                        help='Learning rate parameter for the Adam optimizer. (default: %(default)s)')

    parser.add_argument('--max_iterations', type=int,
                        default=300,
                        help='Max number of iterations for the Adam or L-BFGS optimizer. (default: %(default)s)')

    parser.add_argument('--print_iterations', type=int,
                        default=5,
                        help='Number of iterations between optimizer print statements. (default: %(default)s)')

    args = parser.parse_args()

    if args.write_iterations_adam and args.optimizer != 'adam':
        parser.error('The optimizer argument should be adam')

    # normalize weights
    args.style_layer_weights   = normalize(args.style_layer_weights)
    args.content_layer_weights = normalize(args.content_layer_weights)
    args.style_imgs_weights    = normalize(args.style_imgs_weights)

    maybe_make_directory(args.img_output_dir)
    check_model(args.model_weights)
    return args

def render_single_image():
    with tf.Graph().as_default():
        print('\n---- RENDERING SINGLE IMAGE ----\n')
        start_time = time.time()
        stylize()
        end_time = time.time()
        print('Single image elapsed time: {}s'.format(end_time - start_time))

def stylize():
    args = parse_args()
    with tf.device(args.device), tf.Session() as sess:
        st = style_transfer.StyleTransfer(args, sess=sess)

        output_img, content_img, style_imgs, init_img = st.update()
        write_image_output(args, output_img, content_img, style_imgs, init_img)

def main():
    render_single_image()

if __name__ == '__main__':
    main()