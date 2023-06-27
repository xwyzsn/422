import re
import pandas as pd
ENG_REG = re.compile('english: (.+)')
ZH_REG = re.compile('mandarin: (.+)')
def read_data(path:str='./data/data.txt'):
    res = []
    with open(path ,mode='r') as f :
        lines = f.readlines()
        d = {}
        for line in lines:
            if line.startswith("english"):
                eng = re.search(ENG_REG,line).groups()[-1]
                d['eng'] = eng.rstrip('\n')
            elif line.startswith("mandarin"):
                zh = re.search(ZH_REG,line).groups()[-1]
                d['zh'] = zh.rstrip('\n')
            elif line.startswith("--"):
                res.append(d.copy())
                d.clear()
            else:continue
    return res


if __name__ == '__main__':
    data = read_data()
    df = pd.DataFrame(data)
    df.to_csv('./data.csv')



