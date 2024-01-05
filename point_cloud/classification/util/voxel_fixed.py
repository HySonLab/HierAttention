import numpy as np

def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr

class Voxel:
    def __init__(self, divide=4, depth=4):
        self.divide = divide
        self.depth = depth
    
    def debug(self, x):
        y = np.unique(x[:, -2], return_counts = True)[1]
        print("HELLO", y[y>1])

    def get_mappings(self, points):
        # Initialization         
        points = np.array(points) # Convert point to array
        original_points = points.copy() # Clone points to return later
        n = points.shape[0] # Get num points
        min_val = points.min(axis = 0) - 1e-6# Min value 
        max_val = points.max(axis = 0) + 1e-6 # Max value
        steps = np.concatenate([(max_val - min_val)/self.divide**i for i in range(self.depth)], axis = 0) # Calculate step -> (3*depths)
        points = np.tile(points - min_val,(1, self.depth)) # Duplicate points to perform voxelization for different steps - [x, y, z, x, y, z, ..., x, y, z] -> (n, 3*depths)        
        
        # Voxelization        
        coords = np.trunc(points/steps)
        coords = coords.reshape([-1, self.depth, 3]).reshape([-1, 3], order = 'F') # Calculate coord of a point -> (n, 3*depths) -> (n * depth, 3)
        depths = np.arange(self.depth + 1).repeat(n)[..., None] # Initialize depth [0,0,0 ... 1,1,1 ... 2,2,2 ... depth, depth, depth] -> (n * (depth + 1), 1)
        coords = np.concatenate([coords, depths[:n*self.depth]], axis = -1).astype(np.int32) # Concat
        
        # Build tree        
        _, virtuals_idx, coords_idx = np.unique(fnv_hash_vec(coords), return_index=True, return_inverse=True) # Get index of virtuals points and all points
        
        # Calculate position of virtual nodes        
        virtuals = coords[virtuals_idx] # Get virtual nodes
        steps_size = steps.reshape(self.depth, -1)[virtuals[:, -1]] # Get xyz step size of virtual nodes
        virtual_points = virtuals[:, :3] * steps_size + steps_size/2  # Calculate position of virtual points

        # Create table that contain [loc voxel of a point at step size 1, loc voxel of a point at step 2, ..., idx of a point]         
        coords_idx = coords_idx.reshape(n, self.depth, order = 'F') # Create table of idx of vitual node
        coords_idx += n # Index of virtual nodes begin at n rather than 0
        coords_idx = np.concatenate([coords_idx, np.arange(n)[:, None]], axis = -1) # Concat real nodes
        coords_idx_depth = np.dstack([coords_idx, depths.reshape(-1, self.depth + 1, order = 'F')]) # Concat depth infomation
        
        # # Debug
        # self.debug(coords_idx)

        # Convert table to mappings         
        mappings = [] 
        
        for i in range(self.depth): # Find unique pair of two columns [i, i+1]
            q = i
            k = i + 1
            if q == 0: # In the first layer, all node are the children of the root node
                query = np.where(virtuals[:, -1] == 0)[0] + n
                key = np.where(virtuals[:, -1] == 1)[0] + n 
                mappings.append(np.vstack([query.repeat(key.shape), np.zeros_like(key), key, np.ones_like(key)]).T)
            elif q == self.depth - 1: # In the last layer, no duplicate pairs
                mappings.append(coords_idx_depth[:, [q, k]].reshape(-1, 4))
            else: # Check and remove duplicates
                coords_idx_depth_layer = coords_idx_depth[:, [q, k]].reshape(-1, 4)
                mappings.append(coords_idx_depth_layer[np.unique(fnv_hash_vec(coords_idx_depth_layer[:, [0, 2]]), return_index=True)[1]])
                
        # Gather results
        mappings = np.concatenate(mappings, axis = 0)
        points = np.concatenate([original_points, virtual_points], axis = 0)
        mappings = mappings[:, [0,2,1,3]]

        return mappings, points

if __name__ == "__main__":
    voxel = Voxel()
    points = np.loadtxt("dataset/modelnet40_normal_resampled/sink/sink_0078.txt", delimiter=',')[:1024,:3]
    a = voxel.get_mappings(points)