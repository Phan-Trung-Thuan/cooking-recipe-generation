import matplotlib.pyplot as plt

text = '''Epoch:  1, Train Loss: 0.611272, Val Loss: 0.501793
Epoch:  2, Train Loss: 0.576085, Val Loss: 0.476603
Epoch:  3, Train Loss: 0.562478, Val Loss: 0.466494
Epoch:  4, Train Loss: 0.551071, Val Loss: 0.461554
Epoch:  5, Train Loss: 0.547335, Val Loss: 0.457554
Epoch:  6, Train Loss: 0.542338, Val Loss: 0.454835
Epoch:  7, Train Loss: 0.534591, Val Loss: 0.452678
Epoch:  8, Train Loss: 0.536167, Val Loss: 0.451148
Epoch:  9, Train Loss: 0.497563, Val Loss: 0.425833
Epoch: 10, Train Loss: 0.498166, Val Loss: 0.426147
Epoch: 11, Train Loss: 0.498955, Val Loss: 0.426520
Epoch: 12, Train Loss: 0.498700, Val Loss: 0.426365
Epoch: 13, Train Loss: 0.532344, Val Loss: 0.449168
Epoch: 14, Train Loss: 0.531730, Val Loss: 0.448113
Epoch: 15, Train Loss: 0.529223, Val Loss: 0.445977
Epoch: 16, Train Loss: 0.527232, Val Loss: 0.446489
Epoch: 17, Train Loss: 0.515771, Val Loss: 0.439053
Epoch: 18, Train Loss: 0.513480, Val Loss: 0.436738
Epoch: 19, Train Loss: 0.510182, Val Loss: 0.435502
Epoch: 20, Train Loss: 0.510072, Val Loss: 0.435354'''

text = text.split('\n')

# print(text)

train_loss = []
val_loss = []
for item in text:
    item = item.split(',')
    train_loss.append(item[1][item[1].find(': ') + 2:])
    val_loss.append(item[2][item[2].find(': ') + 2:])

train_loss = [float(item) for item in train_loss]
val_loss = [float(item) for item in val_loss]

plt.plot([i for i in range(1, 21)], train_loss, label='train loss')
plt.plot([i for i in range(1, 21)], val_loss, label='val loss')

plt.xlabel("epochs") 
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
plt.ylabel("cross entropy loss")

plt.legend()
plt.show()