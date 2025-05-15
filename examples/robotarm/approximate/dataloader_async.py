import queue
from torch.utils.data import Dataset
import threading
import jax
import jax.numpy as jnp
import numpy as np
import h5py
import tqdm

    
class JaxDataLoader:
    def __init__(self, dataset, rng_key, batch_size=1, shuffle=False, prefetch_size=100):
        self.dataset = dataset
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        self.rng_key = rng_key
        
    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self):
        if self.shuffle:
            self.rng_key, key = jax.random.split(self.rng_key)
            self.indices = np.array(jax.random.permutation(key, len(self.dataset)))
        else:
            self.indices = np.arange(len(self.dataset))

        self.idx = 0
        self.max_idx = len(self.indices)

        def batch_generator():
            while self.idx < self.max_idx:
                batch_idx = self.indices[self.idx:self.idx+self.batch_size]
                batch = [self.dataset[i] for i in batch_idx]

                batch_X = np.stack([b["X"] for b in batch])
                batch_U = np.stack([b["U"] for b in batch])
                batch_Y = np.stack([b["Y"] for b in batch])

                self.idx += self.batch_size
                yield batch_X, batch_U, batch_Y

        return self._prefetch(batch_generator(), buffer_size=self.prefetch_size)

    def _prefetch(self, generator, buffer_size):
        """Asynchronously prefetch batches using a thread-safe queue."""
        q = queue.Queue(maxsize=buffer_size)

        def producer():
            for item in generator:
                item = jax.device_put(item)  # move to device here (fasten training)
                q.put(item)
            q.put(None)  # end signal

        threading.Thread(target=producer, daemon=True).start()

        while True:
            item = q.get()
            if item is None:
                break
            yield item