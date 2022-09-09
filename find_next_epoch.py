import os
import sys
from vcom_python import util_file as vfile

if __name__ == "__main__":
    ChekPointPath="/apdcephfs_cq2/share_1630463/portrait_matting_cache/"
    pths = vfile.get_files_recursively(ChekPointPath, ".pth", 3)
    pths = set(map(lambda pth: os.path.split(pth)[1], pths))
    print(pths)
    epoch = 0;
    for i in range(0, 100):
        epoch_file = "epoch-" + str(i) + ".pth"
        # print(epoch_file)
        if not epoch_file in pths:
            epoch = i;
            break;

    sys.exit(epoch)

