import os


def GetFileNameSplit(filePath):
    bn = os.path.basename(filePath)
    arr = os.path.splitext(bn)
    pre = arr[0]
    ext = arr[-1]
    return bn, pre, ext