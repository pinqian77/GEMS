import numpy as np

class SimHash(object):
    """Hash-based count bonus for exploration.
    Tang, H., Houthooft, R., Foote, D., Stooke, A., Chen, X., Duan, Y., Schulman, J., De Turck, F., and Abbeel, P. (2017).
    #Exploration: A study of count-based exploration for deep reinforcement learning.
    In Advances in Neural Information Processing Systems (NIPS)
    """
    
    def __init__(self, dim_key=128, obs_processed_flat_dim=None, bucket_sizes=None):
        # Hashing function: SimHash
        if bucket_sizes is None:
            # Large prime numbers
            bucket_sizes = [999931, 999953, 999959, 999961, 999979, 999983]
        mods_list = []
        for bucket_size in bucket_sizes:
            mod = 1
            mods = []
            for _ in range(dim_key):
                mods.append(mod)
                mod = (mod * 2) % bucket_size
            mods_list.append(mods)
        self.bucket_sizes = np.asarray(bucket_sizes)
        self.mods_list = np.asarray(mods_list).T
        self.tables = np.zeros((len(bucket_sizes), np.max(bucket_sizes)))
        self.projection_matrix = np.random.normal(size=(obs_processed_flat_dim, dim_key))


    def compute_keys(self, obss):
        binaries = np.sign(np.asarray(obss).dot(self.projection_matrix))
        keys = np.cast['int'](binaries.dot(self.mods_list)) % self.bucket_sizes
        return keys