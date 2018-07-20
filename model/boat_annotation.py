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
        bateau_class = read_classes("boat_classes.txt")
        ################################################################################################################
        with open('vt_fullInterpolated.json') as json_data:
            d = json.load(json_data)
            json_data.close()
            num = []
            for i in range(len(d['frames'])):
                # print(type(d['frames'][i]['num']))
                num.append(d['frames'][i]['num'])
        number_image = len(d['frames'])
        number_of_trainset = round(number_image*0.7)
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
        file_train = open('boat_train.txt', 'w')
        file_valid = open('boat_valid.txt', 'w')
        file_test = open('boat_test.txt', 'w')
        for img in range(number_image):
            image_name = '0000000.ppm'
            c = list(image_name)
            c[7 - len(str(img+1)):7] = str(img+1)
            image_name = ''.join(c)
            if len((d['frames'])[img]['annotations']) == 0:
                hihihi=1
                #if d['frames'][img]['num'] in train_num:
                #    file_train.write('%s/dataset/%s' %(wd, image_name))  # TODO save dataset in folder dataset
                #    file_train.write('\n')
                #elif d['frames'][img]['num'] in valid_num:
                #    file_valid.write('%s/dataset/%s' % (wd, image_name))
                #    file_valid.write('\n')
                #elif d['frames'][img]['num'] in test_num:
                #    file_test.write('%s/dataset/%s' % (wd, image_name))
                #    file_test.write('\n')
            else:
                for obj in range(len((d['frames'])[img]['annotations'])):
                    id = d['frames'][img]['annotations'][obj]['id']  # string
                    type = read_idx(id)
                    index = bateau_class.index(type)
                    xmin = d['frames'][img]['annotations'][obj]['x']  # int
                    ymin = d['frames'][img]['annotations'][obj]['y']
                    w = d['frames'][img]['annotations'][obj]['width']
                    h = d['frames'][img]['annotations'][obj]['height']
                    xmax = xmin + w
                    ymax = ymin + h
                    if obj == 0:
                        if d['frames'][img]['num'] in train_num:
                            file_train.write('%s/dataset/%s ' % (wd, image_name))
                            file_train.write(str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(index))
                        elif d['frames'][img]['num'] in valid_num:
                            file_valid.write('%s/dataset/%s ' % (wd, image_name))
                            file_valid.write(str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(index))
                        elif d['frames'][img]['num'] in test_num:
                            file_test.write('%s/dataset/%s ' % (wd, image_name))
                            file_test.write(str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(index))
                    else:
                        if d['frames'][img]['num'] in train_num:
                            file_train.write(' ' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(index))
                        elif d['frames'][img]['num'] in valid_num:
                            file_valid.write(' ' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(index))
                        elif d['frames'][img]['num'] in test_num:
                            file_test.write(' ' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(index))
                if d['frames'][img]['num'] in train_num:
                    file_train.write('\n')
                elif d['frames'][img]['num'] in valid_num:
                    file_valid.write('\n')
                elif d['frames'][img]['num'] in test_num:
                    file_test.write('\n')
        file_train.close()
        file_valid.close()
        file_test.close()


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

read_xywhi()

print(time.time()-start)


