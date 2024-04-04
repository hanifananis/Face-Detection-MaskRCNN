# -*- coding: utf-8 -*-
import os
import json
dir = "f:/wider_face_split/wider_face_train"

for filelist in os.walk(dir):
    for file_dir in filelist[1]:
        for subfilelist in os.walk(''.join([dir, "/", file_dir])):
            for file in subfilelist[2]:
                if file.endswith(".json"):
                    jsonfile = json.load(open(''.join([dir, "/", file_dir, "/", file])))
                    print(file, "len:", len(jsonfile))
