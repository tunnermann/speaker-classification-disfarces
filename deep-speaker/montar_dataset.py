import os

shpfiles = []
for dirpath, subdirs, files in os.walk('../data/'):
    for x in files:
        if x.endswith(".wav"):
            shpfiles.append(os.path.join(dirpath, x))


with open('dataset.txt', 'w') as f_out:
    for file_path in shpfiles:
        tokens = file_path.split('/')
        locutor_disfarce = tokens[3].split('_')
        f_out.write(file_path + '|' + locutor_disfarce[0] + '|' + locutor_disfarce[1]+'\n')

