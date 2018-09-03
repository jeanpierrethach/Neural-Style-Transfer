import numpy as np
import os
import errno
import cv2
import timestr

def normalize(weights):
    denom = sum(weights)
    if denom > 0.:
        return [float(w) / denom for w in weights]
    return [0.] * len(weights)

def check_model(model_path):
    if not os.path.exists(model_path):
        raise OSError(errno.ENOENT, "Missing model file in root project", model_path)

def check_image(img, path):
    if img is None:
        raise OSError(errno.ENOENT, "No such file", path)

def maybe_make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def check_resize(img, max_size):
    h, w, d = img.shape
    mx = max_size

    if h > w and h > mx:
        w = (float(mx) / float(h)) * w
        img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
    if w > mx:
        h = (float(mx) / float(w)) * h
        img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)
    return img

def preprocess(img):
    mean_pixel = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    # bgr to rgb
    img = img[...,::-1]
    # shape (h, w, d) to (1, h, w, d)
    img = img[np.newaxis,:,:,:]
    img -= mean_pixel
    return img

def postprocess(img):
    mean_pixel = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    img += mean_pixel
    # shape (1, h, w, d) to (h, w, d)
    img = np.clip(img[0], 0, 255).astype('uint8')
    # rgb to bgr
    img = img[...,::-1]
    return img

def write_image(path, img):
    img = postprocess(img)
    cv2.imwrite(path, img)

def write_image_output(args, output_img, content_img, style_imgs, init_img):
    out_dir = os.path.join(args.img_output_dir, args.img_name, timestr.get_time())
    maybe_make_directory(out_dir)
    img_path = os.path.join(out_dir, args.img_name+'.png')
    content_path = os.path.join(out_dir, 'content.png')
    init_path = os.path.join(out_dir, 'init.png')

    write_image(img_path, output_img)
    write_image(content_path, content_img)
    write_image(init_path, init_img)

    for idx, style_img in enumerate(style_imgs):
        path = os.path.join(out_dir, 'style_'+str(idx)+'.png')
        write_image(path, style_img)

    # save the configuration settings
    out_file = os.path.join(out_dir, args.meta_name)
    write_metadata(out_file, args)

def write_metadata(path, args):
    with open(path , 'w') as f:
        f.write('image_name: {}\n'.format(args.img_name))
        f.write('content: {}\n'.format(args.content_img))

        for idx, (style_img, weight) in enumerate(zip(args.style_imgs, args.style_imgs_weights)):
            f.write('styles['+str(idx)+']: {} * {}\n'.format(weight, style_img))
        f.write('init_type: {}\n'.format(args.init_img_type))
        f.write('content_weight: {}\n'.format(args.content_weight))
        f.write('style_weight: {}\n'.format(args.style_weight))
        f.write('tv_weight: {}\n'.format(args.tv_weight))
        f.write('content_layers: {}\n'.format(args.content_layers))
        f.write('style_layers: {}\n'.format(args.style_layers))
        f.write('optimizer_type: {}\n'.format(args.optimizer))
        f.write('max_iterations: {}\n'.format(args.max_iterations))
        f.write('max_size: {}\n'.format(args.max_size))