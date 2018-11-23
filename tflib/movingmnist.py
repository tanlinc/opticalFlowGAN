import numpy as np
import os

def movingmnist_generator(data, batch_size):
    num_seqs = data.shape[1]
    data = np.reshape(data, [20, num_seqs, 4096])

    def get_epoch():
        NUM_BATCHES = np.round(num_seqs/batch_size).astype('uint8') # for whole 10000, b=50: 200

        while True:
            np.random.shuffle(data) # TODO: check how it is shuffled, certain dim..
            for seq_ctr in range (0, NUM_BATCHES):
                for frame in range(0,18): 
                    batch = data[frame:(frame+3), seq_ctr*batch_size : (seq_ctr+1)*batch_size,:]
                    yield np.copy(batch)

    return get_epoch            

def load(batch_size = 50, test_batch_size = 50): 
    filepath = '/home/linkermann/Desktop/MA/data/movingMNIST/mnist_test_seq.npy'

    if not os.path.isfile(filepath):
        print("Couldn't find movingMNIST dataset")

    dat = np.load(filepath) 
    # print(dat.shape) = (20, 10000, 64, 64)
    # TODO: generate/load other files as dev and test data!
    train_data, dev_data, test_data = dat[:,0:7000,:], dat[:,7000:8000,:], dat[:,8000:10000,:]

    return (
        movingmnist_generator(train_data, batch_size), 
        movingmnist_generator(dev_data, test_batch_size), 
        movingmnist_generator(test_data, test_batch_size)
    )

if __name__ == '__main__':
    train_gen, dev_gen, test_gen = load(50, 50)	# load sets into global vars
    gen = train_gen()		# init iterator for training set
    
    data = next(gen)
    print(data.shape) # (3, 50, 4096)
