__author__ = 'oli'

# Names for coding as used in the second photo-taking session
def getNamesFromFileName(name):
    import re
    match = re.search(r'(\w+)-+(\d)-+(\d+)\.png', name)
    if (match):
        nameP = match.group(1)
        batch = match.group(2)
        number = match.group(3)
        return [nameP, int(batch), int(number)]


def read_images_2(path, sz=None, useBatch=-1, maxNum=100000000, loadFaces=True):
    # print 'This is the file reader read_images_2()'
    # print 'path: ', path
    # print 'size: ', sz
    # print 'useBatch: ', useBatch
    # print 'maxNum ', maxNum
    # print 'loadFaces ', loadFaces
    c = 0
    y, block, names, retFN = [], [], [], []
    import os
    for dirname, dirnames, filename in os.walk(path):
        if '.svn' in dirnames:
            dirnames.remove('.svn')
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if (filename != ".svn"):
                        fn = os.path.join(subject_path, filename)
                        # print("Readding Filename " + fn)
                        #print("----------", str(filename))
                        ret = getNamesFromFileName(filename)
                        if (ret is not None):
                            #print 'file name: ', str(filename) + '\t ret: ' + str(ret)
                            (name, batch, number) = ret
                            if (useBatch < 0 or useBatch == int(batch) and number < maxNum):
                                print(str(name) + "|" + str(batch) + "|" + str(number))
                                retFN.append(fn)
                                #name = string.split(subject_path, "/")[-1] + "_" + filename
                                #print(name)
                                if not name in names and len(names) > 0:
                                    c = c + 1
                                y.append(c)
                                names.append(name)
                                block.append(batch)
                except IOError as e:
                    print "I/O error({0}): {1}".format(e.errno, e.strerror)
    c = c + 1
    print("Read " + str(c) + " Files")
    print 'length of (names): ', len(names)
    return [y, block, names, retFN]