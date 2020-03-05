import math
import os

def parsing(dataPath):
    """
    Args:
        dataPath (string) : path of dataset along with images, cache.txt
    Return: 
        dataInfo (list) : list of information of dataset [imagePath, score, magnification]
    Example::
        >>> testData = parsing('./dataset')
    """
    f = open(os.path.join(dataPath, 'cache.txt'), 'r')
    data = f.read().split('\n')
    idx1, idx2, _ = map(int, data.pop(0).split('\t'))
    data = data[idx1+idx2:-1]

    testData = []
    for dat in data:
        imgname, target, mag = dat.split('\t')[:3]
        imgname = os.path.join(dataPath, imgname)
        testData.append([imgname, target, mag])
    f.close()
    return testData

def get_sharpness(dataInfo, sharpness_fn):
    """
    Args:
        dataInfo (list) : list of information of dataset [imagePath, score, magnification]
        sharpness_fn (func) : one of the sharpness functions
    Return:
        sharp_data (dict) : hashmap (key:imagePath, value:sharpness_value)
    Example:
        >>> sharp_data = get_sharpness(dataInfo, sharpness_fn)
    """
    min_value, max_value = float('inf'), -float('inf')
    sharp_data = {}
    for info in dataInfo:
        imgname = info[0]
        sharpness_value = sharpness_fn(imgname, crop_size=(240, 320))
        sharp_data[imgname] = sharpness_value
    return sharp_data


def printResult(dataInfo, sharpnessInfo):
    """
    Args:
        dataInfo (list) : list of information of dataset [imagePath, score, magnification]
        sharpnessInfo (dict) : hashmap (key:imagePath, value:sharpness_value)
    Print:
        *RMSE : root mean square error with total dataset and per magnification [500, 1000, 2000, 5000, 10000]
    Example:
        >>> printResult(dataInfo, sharpnessInfo)
    """
    total_sharpness_value = sum(sharpnessInfo.values())
    max_imgname = max(sharpnessInfo.keys(), key=lambda x:sharpnessInfo[x])
    min_imgname = min(sharpnessInfo.keys(), key=lambda x:sharpnessInfo[x])
    max_value = sharpnessInfo[max_imgname]
    min_value = sharpnessInfo[min_imgname]
    meanValue = int(total_sharpness_value/len(dataInfo))

    dataInfo.sort()
    err_per_mag = {'500':[0,0], '1000':[0,0], '2000':[0,0], '5000':[0,0], '10000':[0,0]}
    total_error = 0
    for imgname, target, mag in dataInfo:
        score = sharpnessInfo[imgname] * (10/(max_value-min_value))
        error = math.sqrt((float(target)-float(score))**2)
        total_error += error
        err_per_mag[mag][0] += 1
        err_per_mag[mag][1] += error
        imgname = os.path.normpath('/'+imgname)
        print(f'{imgname}\t{score:.2f}')
    print(f'mean = {meanValue}')
    print(f'min = {min_imgname} : {min_value}')
    print(f'max = {max_imgname} : {max_value}')
    print('RMSE = {}'.format(total_error/len(dataInfo)))
    print('500 RMSE = {}'.format(err_per_mag['500'][1]/err_per_mag['500'][0]))
    print('1000 RMSE = {}'.format(err_per_mag['1000'][1]/err_per_mag['1000'][0]))
    print('2000 RMSE = {}'.format(err_per_mag['2000'][1]/err_per_mag['2000'][0]))
    print('5000 RMSE = {}'.format(err_per_mag['5000'][1]/err_per_mag['5000'][0]))
    print('10000 RMSE = {}'.format(err_per_mag['10000'][1]/err_per_mag['10000'][0]))
