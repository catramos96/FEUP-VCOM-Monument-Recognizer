from keras.preprocessing import image
from xml.dom import minidom
import os
import os.path
import sys


def get_data_info(base_directory, class_name, class_index):

    images_dir = "{}/images/{}".format(base_directory, class_name)
    annotations_dir = "{}/annotations/{}".format(base_directory, class_name)
    info_file_path = "{}/{}_information.txt".format(base_dir, class_name)

    f = open(info_file_path, "w+")

    first = True

    # iterate files in images fot the class

    for image_name in os.listdir(images_dir):

        index_p = image_name.rfind('.')
        base_name = image_name[:index_p]
        annot_name = "{}.xml".format(base_name)

        # check if files are valid

        if os.path.isfile(os.path.join(images_dir, image_name)) and os.path.isfile(os.path.join(annotations_dir, annot_name)):

            # get paths to annotations and images

            image_path = "{}/images/{}/{}".format(
                base_directory, class_name, image_name)
            annot_path = "{}/annotations/{}/{}".format(
                base_directory, class_name, annot_name)

            # get bounding box from xml

            xmldoc = minidom.parse(annot_path)
            xmin = int(round(float(xmldoc.getElementsByTagName(
                'xmin')[0].firstChild.nodeValue), 0))
            ymin = int(round(float(xmldoc.getElementsByTagName(
                'ymin')[0].firstChild.nodeValue), 0))
            xmax = int(round(float(xmldoc.getElementsByTagName(
                'xmax')[0].firstChild.nodeValue), 0))
            ymax = int(round(float(xmldoc.getElementsByTagName(
                'ymax')[0].firstChild.nodeValue), 0))

            endline = ""

            if(not first):
                endline = "\n"
            else:
                first = False

            info = "{}{} {},{},{},{},{}".format(
                endline, image_path, xmin, ymin, xmax, ymax, class_index)

            f.write(info)

    f.close()

    return info_file_path


if __name__ == "__main__":

    if len(sys.argv) <= 1:
        print("Specify the path for \"data\"")
        exit()

    base_dir = sys.argv[1]
    images_dir = "{}/images".format(base_dir)

    p = open("data_resources.py", "w+")

    classes_code = "classes = [\""
    info_code = "data_info = [\""

    first = True
    n_classes = 0

    # iterate directory of images
    for class_name in os.listdir(images_dir):

        # check for directories = classes
        if(os.path.isdir(os.path.join(images_dir, class_name))):

            info_path = get_data_info(base_dir, class_name, n_classes)
            n_classes += 1

            if(first):
                first = False
            else:
                classes_code += "\", \""
                info_code += "\", \""

            classes_code += class_name
            info_code += info_path

    classes_code += "\"]"
    info_code += "\"]"

    # save on a python file the classes and path to the info abour each class data
    p.write(classes_code)
    p.write("\n\n")
    p.write(info_code)

    p.close()
