# Neural-Style-Transfer


A tensorflow implementation of style transfer (neural style) described in the papers:
* [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576v2.pdf) : *submitted version*
* [Image Style Transfer Using Convolutional Neural Networks](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) : *published version*  
by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge

## Setup
#### Dependencies:
* [tensorflow](https://github.com/tensorflow/tensorflow)
* [opencv](http://opencv.org/downloads.html)


#### Optional (but recommended) dependencies:
* [CUDA](https://developer.nvidia.com/cuda-downloads) 8.0+
* [cuDNN](https://developer.nvidia.com/cudnn) 5.0+


#### After installing the dependencies: 
* Download the [VGG-19 model weights](http://www.vlfeat.org/matconvnet/pretrained/) (see the "VGG-VD models from the *Very Deep Convolutional Networks for Large-Scale Visual Recognition* project" section). More info about the VGG-19 network can be found [here](http://www.robots.ox.ac.uk/~vgg/research/very_deep/).
* After downloading, copy the weights file `imagenet-vgg-verydeep-19.mat` to the **project directory**.

## Usage
### Basic Usage

#### Single Image
1. Copy 1 content image to the default image content directory `./image_input`
2. Copy 1 or more style images to the default style directory `./styles`

*Note*: Supported image formats include: `.png`, `.jpg`, `.ppm`, `.pgm`

3. Run the command with specific arguments:
```
python neural_style_transfer.py <arguments>
```
*Example*:
```
python neural_style_transfer.py --content_img c1.jpg \
                                --style_imgs s1.jpg \
                                --max_size 1024 \
                                --max_iterations 100 \
                                --device /cpu:0 \
                                --verbose;
```

To use multiple style images, pass a *space-separated* list of the image names and image weights like this:

`--style_imgs s1.jpg s2.jpg --style_imgs_weights 0.5 0.5`

#### Arguments
* `--meta_name` : Filename of the configuration settings
* `--content_img`: Filename of the content image. *Example*: `c1.jpg`
* `--content_img_dir`: Relative or absolute directory path to the content image. *Default*: `./image_input`
* `--style_imgs`: Filenames of the style images. To use multiple style images, pass a *space-separated* list.  *Example*: `--style_imgs s1.jpg`
* `--style_imgs_weights`: The blending weights for each style image.  *Default*: `1.0` (assumes only 1 style image)
* `--style_imgs_dir`: Relative or absolute directory path to the style images. *Default*: `./styles`
* `--init_img_type`: Image used to initialize the network. *Default*: `content`
* `--max_size`: Maximum width or height of the input images. *Default*: `1920`
* `--content_weight`: Weight for the content loss function. *Default*: `5e0`
* `--style_weight`: Weight for the style loss function. *Default*: `1e4`
* `--tv_weight`: Weight for the total variational loss function. *Default*: `1e-3`
* `--content_layers`: *Space-separated* VGG-19 layer names used for the content image. *Default*: `conv4_2`
* `--style_layers`: *Space-separated* VGG-19 layer names used for the style image. *Default*: `relu1_1 relu2_1 relu3_1 relu4_1 relu5_1`
* `--content_layer_weights`: *Space-separated* weights of each content layer to the content loss. *Default*: `1.0`
* `--style_layer_weights`: *Space-separated* weights of each style layer to loss. *Default*: `0.2 0.2 0.2 0.2 0.2`
* `--model_weights`: Weights and biases of the VGG-19 network.  Download [here](http://www.vlfeat.org/matconvnet/pretrained/). *Default*:`imagenet-vgg-verydeep-19.mat`
* `--pooling_type`: Type of pooling in convolutional neural network. *Choices*: `avg`, `max`. *Default*: `avg`
* `--device`: GPU or CPU device.  GPU mode highly recommended but requires NVIDIA CUDA. *Choices*: `/gpu:0` `/cpu:0`. *Default*: `/gpu:0`
* `--img_output_dir`: Directory to write output to.  *Default*: `./image_output`
* `--img_name`: Filename of the output image. *Default*: `result`
* `--verbose`: Boolean flag indicating if statements should be printed to the console.
* `--write_iterations_adam`: Boolean flag indicating if output images should be writed in every iteration under the Adam optimizer.

#### Optimization Arguments
* `--optimizer`: Loss minimization optimizer.  L-BFGS gives better results.  Adam uses less memory. *Choices*: `lbfgs`, `adam`. *Default*: `lbfgs`
* `--learning_rate`: Learning-rate parameter for the Adam optimizer. *Default*: `1e0`  
* `--max_iterations`: Max number of iterations for the Adam or L-BFGS optimizer. *Default*: `1000`
* `--print_iterations`: Number of iterations between optimizer print statements. *Default*: `50`

## References

This implementation is based on the project:

* https://github.com/cysmith/neural-style-tf