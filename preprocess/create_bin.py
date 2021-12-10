import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from option import args
from data.div2k import DIV2K

if __name__ == '__main__':
    args.ext = "sep+reset"
    args.data_range = "1-800/801-810"
    args.scale = "2"
    print(args.dir_data)
    dataset = DIV2K(args)
