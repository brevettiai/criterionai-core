import os
import shutil

def movedir(srcdir, targetdir):
    for file in os.listdir(srcdir):
        shutil.move(os.path.join(srcdir, file), os.path.join(targetdir, file))

def join(*paths):
    sp = paths[0].split('://', 1)
    if len(sp) == 2:
        # Found protocol
        pathsep = '/'
        paths = list(paths)
        for i, p in enumerate(paths[:-1]):
            if p[-1] == pathsep:
                paths[i] = p[:-1]
        return pathsep.join(paths)
    else:
        return os.path.join(*paths)