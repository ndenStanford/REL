import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

filename = 'data/wiki_2014/basic_data/wiki_name_id_map.txt'
f = open(filename, "r")

for i in range(20):
    print(f.readline())