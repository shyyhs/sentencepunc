import os
import sys
import glob
import random

with open(sys.argv[1],"r") as f1:
    with open(sys.argv[2],"w") as f2:
        lines = f1.readlines()
        totlen = len(lines)
        now = 0
        while (now < totlen):
            linenum = random.randint(1,3)
            wline = ' '.join(lines[now:now+linenum]).replace('\n','') + '\n'
            f2.write(wline)
            now += linenum
