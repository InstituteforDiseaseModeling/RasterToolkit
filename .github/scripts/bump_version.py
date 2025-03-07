import os

from datetime import datetime

VNS = '__version__'
VDS = '__versiondate__'
EQS = ' = "'
NLS = '"\n'


def bump_version():

    fname = os.path.join('rastertoolkit', 'version.py')

    with open(fname) as fid01:
        flines = fid01.readlines()

    with open(fname, 'w') as fid02:
        for lval in flines:
            if (lval.startswith(VDS)):
                dval = datetime.today().strftime('%Y-%m-%d')
                nline = VDS + EQS + dval + NLS
                fid02.write(nline)
            elif (lval.startswith(VNS)):
                ver = lval.split('"')[1]
                vnum = ver.split('.')
                nver = vnum[0] + '.' + vnum[1] + '.'
                nver = nver + str(int(vnum[2]) + 1)
                nline = VNS + EQS + nver + NLS
                fid02.write(nline)
            else:
                fid02.write(lval)

    return None


if __name__ == '__main__':
    bump_version()
