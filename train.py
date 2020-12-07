from CSA_SR import *
from utils import *
from focalloss import *


EPOCH = 50
nIter = 1200
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
vovab_size = len(word_counts)
lamda = 0.2


# save training log
def write_txt(epoch, iteration, loss):
    with open("training_log.txt", 'a+') as f:
        f.write("Epoch:[ %d ]\t Iteration:[ %d ]\t loss:[ %f ]\n" % (epoch, iteration, loss))


if __name__ == "__main__":
    # torch.cuda.set_device(1)
    pkl_file = False
    csa_sr = CSA_SR(vocab_size=vovab_size, batch_size=BATCH_SIZE)
    if pkl_file:
        csa_sr.load_state_dict(torch.load("Data/epoch_0.pkl"))
    csa_sr = csa_sr.cuda()
    loss_func = nn.CrossEntropyLoss()
    loss_reconstructor = torch.nn.MSELoss()
#     loss_func = FocalLoss(2, 0.25)
    optimizer = torch.optim.Adam(csa_sr.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCH):
        if epoch == 40:
            optimizer = torch.optim.Adam(csa_sr.parameters(), lr=0.0001)
        for i in range(nIter):
            video, caption, cap_mask, tag, linear = fetch_train_data(BATCH_SIZE)
            video, caption, cap_mask, tag, linear = torch.FloatTensor(video).cuda(), torch.LongTensor(caption).cuda(), \
                                       torch.FloatTensor(cap_mask).cuda(), torch.FloatTensor(tag).cuda(), \
                torch.FloatTensor(linear).cuda()

            linear = linear.sum(1)/linear.shape[1]
#             cap_out = csa_sr(video, tag, linear, caption)
            cap_out, sem_out = csa_sr(video, tag, caption)
    
            cap_labels = caption[:, 1:].contiguous().view(-1)  # size [batch_size, 79]
            cap_mask = cap_mask[:, 1:].contiguous().view(-1)  # size [batch_size, 79]
            logit_loss = loss_func(cap_out, cap_labels)
            masked_loss = logit_loss * cap_mask
            
            loss_word = torch.sum(masked_loss) / torch.sum(cap_mask)
            loss_sem = loss_reconstructor(sem_out, tag)

            loss = loss_word + lamda * loss_sem

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                # print("Epoch: %d  iteration: %d , loss: %f" % (epoch, i, loss))
                write_txt(epoch, i, loss)
            if i % 1199 == 0:
                torch.save(csa_sr.state_dict(), "Data/epoch_"+str(epoch)+".pkl")
                print("Epoch: %d iter: %d save successed!" % (epoch, i))
