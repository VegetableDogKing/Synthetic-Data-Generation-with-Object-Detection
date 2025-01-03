#!/usr/bin/env python2

from PIL import Image
import argparse
import os
import csv

OUT_LABELS_DIR = "labels"

CLAZZ_NUMBERS = {
            "bswitch": 0, 
            "grbtn": 1, 
            "gswitch": 2, 
            "rrockbtn": 3, 
            "rsrbtn": 4, 
            "rswitch": 5, 
            "rypushbtn": 6, 
            "ybtn": 7, 
            "ysroundbtn": 8,
            "handle": 9,
            "valve": 10
        }

def getSampleId(path):
    basename = os.path.basename(path)
    return os.path.splitext(basename)[0]

def resolveClazzNumberOrNone(clazz, use_dont_care):
    if clazz == "bswitch":
        return CLAZZ_NUMBERS["bswitch"]
    elif clazz == "grbtn":
        return CLAZZ_NUMBERS["grbtn"]
    elif clazz == "gswitch":
        return CLAZZ_NUMBERS["gswitch"]
    elif clazz == "rrockbtn":
        return CLAZZ_NUMBERS["rrockbtn"]
    elif clazz == "rsrbtn":
        return CLAZZ_NUMBERS["rsrbtn"]
    elif clazz == "rswitch":
        return CLAZZ_NUMBERS["rswitch"]
    elif clazz == "rypushbtn":
        return CLAZZ_NUMBERS["rypushbtn"]
    elif clazz == "ybtn":
        return CLAZZ_NUMBERS["ybtn"]
    elif clazz == "ysroundbtn":
        return CLAZZ_NUMBERS["ysroundbtn"]
    elif clazz == "handle":
        return CLAZZ_NUMBERS["handle"]
    elif clazz == "valve":
        return CLAZZ_NUMBERS["valve"]
    return None

def convertToYoloBBox(bbox, size):
    # Yolo uses bounding bbox coordinates and size relative to the image size.
    # This is taken from https://pjreddie.com/media/files/voc_label.py .
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (bbox[0] + bbox[1]) / 2.0
    y = (bbox[2] + bbox[3]) / 2.0
    w = bbox[1] - bbox[0]
    h = bbox[3] - bbox[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def readRealImageSize(img_path):
    # This loads the whole sample image and returns its size.
    return Image.open(img_path).size

def readFixedImageSize():
    # This is not exact for all images but most (and it should be faster).
    return (1242, 375)

def parseSample(lbl_path, img_path, use_dont_care):
    with open(lbl_path) as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=["type", "truncated", "occluded", "alpha", "bbox2_left", "bbox2_top", "bbox2_right", "bbox2_bottom", "bbox3_height", "bbox3_width", "bbox3_length", "bbox3_x", "bbox3_y", "bbox3_z", "bbox3_yaw"], delimiter=" ")
        yolo_labels = []
        for row in reader:
            clazz_number = resolveClazzNumberOrNone(row["type"], use_dont_care)
            if clazz_number is not None:
                size = readRealImageSize(img_path)
                #size = readFixedImageSize()
                # Image coordinate is in the top left corner.
                bbox = (
                        float(row["bbox2_left"]),
                        float(row["bbox2_right"]),
                        float(row["bbox2_top"]),
                        float(row["bbox2_bottom"])
                       )
                yolo_bbox = convertToYoloBBox(bbox, size)
                # Yolo expects the labels in the form:
                # <object-class> <x> <y> <width> <height>.
                yolo_label = (clazz_number,) + yolo_bbox
                yolo_labels.append(yolo_label)
    return yolo_labels

def parseArguments():
    parser = argparse.ArgumentParser(description="Generates labels for training darknet on KITTI.")
    parser.add_argument("--label_dir", default="button_data/Camera", help="data_object_label_2/training/label_2 directory; can be downloaded from KITTI.")
    parser.add_argument("--image_2_dir", default="button_data/Camera/rgb", help="data_object_image_2/training/image_2 directory; can be downloaded from KITTI.")
    parser.add_argument("--training-samples", type=int, default=0.8, help="percentage of the samples to be used for training between 0.0 and 1.0.")
    parser.add_argument("--use-dont-care", action="store_true", help="do not ignore 'DontCare' labels.")
    args = parser.parse_args()
    if args.training_samples < 0 or args.training_samples > 1:
        print("Invalid argument {} for --training-samples. Expected a percentage value between 0.0 and 1.0.")
        exit(-1)
    return args

def main():
    args = parseArguments()

    if not os.path.exists(OUT_LABELS_DIR):
        os.makedirs(OUT_LABELS_DIR)

    print("Generating darknet labels...")
    sample_img_pathes = []
    print(args.label_dir)
    for dir_path, sub_dirs, files in os.walk(args.label_dir):
        for file_name in files:
            if file_name.endswith(".txt"):
                lbl_path = os.path.join(dir_path, file_name)
                sample_id = getSampleId(lbl_path)
                img_path = os.path.join(args.image_2_dir, "{}.png".format(sample_id))
                sample_img_pathes.append(img_path)
                yolo_labels = parseSample(lbl_path, img_path, args.use_dont_care)
                with open(os.path.join(OUT_LABELS_DIR, "{}.txt".format(sample_id)), "w") as yolo_label_file:
                    for lbl in yolo_labels:
                        yolo_label_file.write("{} {} {} {} {}\n".format(*lbl))

    print("Writing training and test sample ids...")
    first_test_sample_index = int(args.training_samples * len(sample_img_pathes))
    with open("kitti_train.txt", "w") as train_file:
        for sample_index in range(first_test_sample_index):
            train_file.write("{}\n".format(sample_img_pathes[sample_index]))
    with open("kitti_test.txt", "w") as test_file:
        for sample_index in range(first_test_sample_index, len(sample_img_pathes)):
            test_file.write("{}\n".format(sample_img_pathes[sample_index]))

if __name__ == "__main__":
    main()
