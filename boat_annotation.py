import json
import random
import time
import os

start = time.time()
wd = os.getcwd()

def read_xywhi():
    with open('vt_fullInterpolated.json') as json_data:
        d = json.load(json_data)
        json_data.close()
        Xmin = []
        Ymin = []
        Xmax = []
        Ymax = []
        Class = []
        bateau_class = read_classes("./model_data/boat_classes.txt")
        ################################################################################################################
        with open('vt_fullInterpolated.json') as json_data:
            d = json.load(json_data)
            json_data.close()
            num = []
            for i in range(len(d['frames'])):
                # print(type(d['frames'][i]['num']))
                num.append(d['frames'][i]['num'])
        number_image = len(d['frames'])
        number_of_trainset = round(number_image*0.8)
        print(number_of_trainset)
        number_of_valiset = round((number_image-number_of_trainset)*0.6)
        print(number_of_valiset)
        number_of_testset = number_image - number_of_trainset - number_of_valiset
        print(number_of_testset)
        train_num = random.sample(population=num, k=number_of_trainset)
        num1 = list(set(num) - set(train_num))
        valid_num = random.sample(population=num1, k=number_of_valiset)
        num2 = list(set(num1) - set(valid_num))
        test_num = random.sample(population=num2, k=number_of_testset)
        ################################################################################################################
        file_train = open('train_boat.txt', 'w')
        file_vali = open('valid_boat.txt', 'w')
        file_test = open('test_boat.txt', 'w')
        for img in range(number_image):
            if len((d['frames'])[img]['annotations']) == 0:
                image_name = '0000000.ppm'
                c = list(image_name)
                c[7 - len(str(img)):7] = str(img)
                image_name = ''.join(c)
                if d['frames'][img]['num'] in train_num:
                    file_train.write('%s/dataset/%s' %(wd, image_name))  # TODO save dataset in folder dataset
                    file_train.write('\n')
                elif d['frames'][img]['num'] in valid_num:
                    file_train.write('%s/dataset/%s' % (wd, image_name))
                    file_vali.write('\n')
                elif d['frames'][img]['num'] in test_num:
                    file_train.write('%s/dataset/%s' % (wd, image_name))
                    file_test.write('\n')
            else:
                for obj in range(len((d['frames'])[img]['annotations'])):
                    id = d['frames'][img]['annotations'][obj]['id']  # string
                    type = read_idx(id)
                    index = bateau_class.index(type)
                    Class.append(index)
                    xmin = d['frames'][img]['annotations'][obj]['x']  # int
                    Xmin.append(xmin)
                    ymin = d['frames'][img]['annotations'][obj]['y']
                    Ymin.append(ymin)
                    w = d['frames'][img]['annotations'][obj]['width']
                    h = d['frames'][img]['annotations'][obj]['height']
                    xmax = xmin + w
                    Xmax.append(xmax)
                    ymax = ymin + h
                    Ymax.append(ymax)
                    if obj == 0:
                        if d['frames'][img]['num'] in train_num:
                            file_train.write('%s/dataset/%s ' % (wd, image_name))
                            file_train.write(str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(index))
                        elif d['frames'][img]['num'] in valid_num:
                            file_train.write('%s/dataset/%s ' % (wd, image_name))
                            file_vali.write(str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(index))
                        elif d['frames'][img]['num'] in test_num:
                            file_train.write('%s/dataset/%s ' % (wd, image_name))
                            file_test.write(str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(index))
                    else:
                        if d['frames'][img]['num'] in train_num:
                            file_train.write('%s/dataset/%s ' % (wd, image_name))
                            file_train.write(' ' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(index))
                        elif d['frames'][img]['num'] in valid_num:
                            file_train.write('%s/dataset/%s ' % (wd, image_name))
                            file_vali.write(' ' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(index))
                        elif d['frames'][img]['num'] in test_num:
                            file_train.write('%s/dataset/%s ' % (wd, image_name))
                            file_test.write(' ' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(index))
                if d['frames'][img]['num'] in train_num:
                    file_train.write('\n')
                elif d['frames'][img]['num'] in valid_num:
                    file_vali.write('\n')
                elif d['frames'][img]['num'] in test_num:
                    file_test.write('\n')
        file_train.close()
        file_vali.close()
        file_test.close()
    return Xmin, Ymin, Xmax, Ymax, Class


def read_idx(id):
    with open('vt_boatstype.json') as json_data:
        d = json.load(json_data)
        json_data.close()
        id = str(id)
        type = d[id]['type']
    return type


def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


a, b, c, d, e = read_xywhi()

print(time.time()-start)


