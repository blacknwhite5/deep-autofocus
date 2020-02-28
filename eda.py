import glob
import sys
import os

txts = glob.glob(os.path.join(sys.argv[1], '*.txt'))

mags = []
for txt in txts:
    with open(txt, 'r') as r:
        data = r.read().split('\n')
    for content in data:
        if 'MAG' in content and 'IMAGETYPE' not in content:
            mags.append(int(content.split(',')[-1]))

print('magnitude :', sorted(list(set(mags))))
print('minimum mag', min(mags))
print('maximum mag', max(mags))
