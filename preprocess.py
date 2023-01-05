import tarfile
import os
import shutil
import random
import scipy.io
import numpy as np
import cv2 as cv

def extract_tar(tar_dst, tar_src):
    if not os.path.exists(tar_dst):
        with tarfile.open(tar_src, "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner) 
                
            
            safe_extract(tar)
    print("Extract tar " + tar_src + " done")

def create_folder(folder):
    if os.path.exists(folder) == False:
        os.makedirs(folder)
    print("Create folder - " + folder + " done")

def save_train_data(fnames, labels, bboxes, margin=0):
    print("Saving train")
    src_folder = 'cars_train'
    num_samples = len(fnames)
    print("num_samples = " + str(num_samples))
    train_split = 0.8
    num_train = int(round(num_samples * train_split))
    train_idx = random.sample(range(num_samples), num_train)

    for i in range(num_samples):
        label = labels[i]
        fname = fnames[i]
        (x1, y1, x2, y2) = bboxes[i]

        src_path = os.path.join(src_folder, fname)
        src_image = cv.imread(src_path)
        height, width = src_image.shape[:2]
        
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)

        if i in train_idx:
            dst_folder = 'data/train'
        else:
            dst_folder = 'data/valid'

        dst_path = os.path.join(dst_folder, label)
        if os.path.exists(dst_path) == False:
            os.makedirs(dst_path)
        dst_path = os.path.join(dst_path, fname)

        crop_image = src_image[y1:y2, x1:x2]
        dst_img = cv.resize(src=crop_image, dsize=(img_height, img_width))
        cv.imwrite(dst_path, dst_img)
    print("Saving train done")

def save_test_data(fnames, bboxes, margin=0):
    print("Saving test")
    src_folder = 'cars_test'
    dst_folder = 'data/test/images'
    num_samples = len(fnames)

    for i in range(num_samples):
        fname = fnames[i]
        (x1, y1, x2, y2) = bboxes[i]
        
        src_path = os.path.join(src_folder, fname)
        src_image = cv.imread(src_path)
        height, width = src_image.shape[:2]
        
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)

        dst_path = os.path.join(dst_folder, fname)
        crop_image = src_image[y1:y2, x1:x2]
        dst_img = cv.resize(src=crop_image, dsize=(img_height, img_width))
        cv.imwrite(dst_path, dst_img)
    print("Saving test done")

def process_train_data():
    print("Processing train")
    annotations = scipy.io.loadmat('devkit/cars_train_annos')
    annotations = annotations['annotations']
    annotations = np.transpose(annotations)

    fnames = []
    bboxes = []
    labels = []

    for annotation in annotations:
        bbox_x1 = annotation[0][0][0][0]
        bbox_y1 = annotation[0][1][0][0]
        bbox_x2 = annotation[0][2][0][0]
        bbox_y2 = annotation[0][3][0][0]
        labels.append('%04d' % (annotation[0][4][0][0],)) # Padding
        fname = annotation[0][5][0]
        bboxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        fnames.append(fname)

    save_train_data(fnames, labels, bboxes)
    print("Processing train done")

def process_test_data():
    print("Processing test")
    cars_annos = scipy.io.loadmat('devkit/cars_test_annos')
    annotations = cars_annos['annotations']
    annotations = np.transpose(annotations)

    fnames = []
    bboxes = []

    for annotation in annotations:
        bbox_x1 = annotation[0][0][0][0]
        bbox_y1 = annotation[0][1][0][0]
        bbox_x2 = annotation[0][2][0][0]
        bbox_y2 = annotation[0][3][0][0]
        fname = annotation[0][4][0]
        bboxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        fnames.append(fname)

    save_test_data(fnames, bboxes)
    print("Processing test done")

def process_test_labels():
    print("Creating test labels")
    mat = scipy.io.loadmat('cars_test_annos_withlabels')
    labels = []
    for i in range(8041):
        labels.append(str(mat['annotations'][0][i][4][0][0]))
    outF = open("test_labels.txt", "w")
    for line in labels:
        outF.write(str(line))
        outF.write("\n")
    outF.close()

if __name__ == '__main__':
    margin = 0
    img_width, img_height = 224, 224
    
    extract_tar('car_devkit', 'car_devkit.tgz')
    extract_tar('cars_train', 'cars_train.tgz')
    extract_tar('cars_test', 'cars_test.tgz')

    create_folder('data/train')
    create_folder('data/valid')
    create_folder('data/test')
    create_folder('data/test/images')
    create_folder('data/custom_test/images')

    process_train_data()
    process_test_data()
    process_test_labels()

    # Clean original data after shifting
    shutil.rmtree('cars_train')
    shutil.rmtree('cars_test')