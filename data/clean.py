import os
import sys
import glob


with open(sys.argv[1],"r") as f1:
    with open(sys.argv[2],"w") as f2:
        lines = f1.readlines()
        for line in lines:
            line = line.strip()
            if (len(line)==0): 
                print (123)
                continue
            if (line[-1]!='。'): line=line+"。" 
            seg = line.split('。')[0]
            wline = seg.strip().replace('，','').replace('、','') + ' 。\n'
            f2.write(wline)
