from CSA_SR import *
from utils import *
import json

vovab_size = len(word_counts)
BATCH_SIZE = 10

if __name__ == "__main__":
    torch.cuda.set_device(0)
    csa_sr = CSA_SR(vocab_size=vovab_size, batch_size=BATCH_SIZE)
    csa_sr = csa_sr.cuda()
    csa_sr.load_state_dict(torch.load("Data/epoch_49.pkl"))
    csa_sr.eval()
    val_result = {}
    for idx in range(0, 670, BATCH_SIZE):
        video, caption, cap_mask, vid, tag, linear = fetch_val_data_orderly(idx, batch_size=BATCH_SIZE)
        video, tag, linear = torch.FloatTensor(video).cuda(), torch.FloatTensor(tag).cuda(), torch.FloatTensor(linear).cuda(),

        cap_out = csa_sr(video, tag, linear)

        captions = []
        for tensor in cap_out:
            captions.append(tensor.tolist())

        captions = [[row[i] for row in captions] for i in range(len(captions[0]))]

        print('............................\nGT Caption:\n')
        print_in_english(captions)
        print('............................\nLABEL Caption:\n')
        print_in_english(caption)
        val_result = save_val_result(vid, captions, val_result)
    with open("results.json", "a+") as f:
        json.dump(val_result, f)

