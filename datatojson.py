# -*- coding: utf-8 -*-
import os
import json

dir = "f:/wider_face_split/wider_face_train"

for filelist in os.walk(dir):
    for file_dir in filelist[1]:
        for subfilelist in os.walk(''.join([dir, "/", file_dir])):
            for file in subfilelist[2]:
                if file.endswith(".txt"):
                    with open(''.join([dir, "/", file_dir, "/", file]), 'r') as txtfile:
                        dict = {}
                        for i, line in enumerate(txtfile.readlines()):
                            num = line.split(' ')
                            dict[i] = {
                                'x': [int(num[0]), int(num[0]) + int(num[2]), int(num[0]) + int(num[2]), int(num[0])],
                                'y': [int(num[1]), int(num[1]), int(num[1]) + int(num[3]), int(num[1]) + int(num[3])]
                            }
                        with open(''.join([dir, "/", file_dir, "/", file[:-4], ".json"]), 'w+') as jsonfile:
                            jsonfile.write(json.dumps(dict))
                        jsonfile.close()
                    txtfile.close()
