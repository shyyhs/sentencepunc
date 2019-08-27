import os
import sys

with open(sys.argv[1], "r") as f1:
    with open(sys.argv[1] + ".nopunc", "w") as f2:
        with open(sys.argv[1] + ".nopunc.num", "w") as f3:
            lines = f1.readlines()
            for line in lines:
                words = line.strip().split()
                flag = 0
                for word in words:
                    if (word!='。'):
                        f2.write(word + ' ')
                        if (flag!=0):
                            f3.write('0 ')
                        else:
                            flag = 1
                    else:
                        f3.write('1 ')
                        flag = 0
                f2.write('\n')
                f3.write('\n')
                    
                    
