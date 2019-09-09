import os
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-sl","--src_length",help="truncate length for source article",default=600)
    parser.add_argument("-tl","--tgt_length",help="truncate length for target summary",default=70)
    parser.add_argument("-d","--dataset",help="dataset name",default="cnndm")

    args = parser.parse_args()

    SRC_LENGTH = int(args.src_length)
    TGT_LENGTH = int(args.tgt_length)
    DATASET = str(args.dataset)

    for f in os.listdir("./" + DATASET + "/bpe-output"):
        if "src" in f:
            LENGTH = SRC_LENGTH
        elif "tgt" in f:
            LENGTH = TGT_LENGTH
        with open("./" + DATASET + "/bpe-output/"+f,"r") as fr:
            fw = open("./" + DATASET + "/bpe-truncate/"+f,"w")
            print(f)
            for line in tqdm(fr):
                truncated = " ".join(line.split(" ")[:LENGTH])
                fw.write(truncated)
                if ord(truncated[-1]) != 10:
                    fw.write("\n")

if __name__ == "__main__":
    main()