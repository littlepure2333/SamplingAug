''' 
This script can create a sparse index_psnr list (ascending) by by throwing darts
index is the (u,v)th patch (u - y axis, v - x axis)
low psnr indicates difficult sample, which is more worth training
'''

import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import time
import random
import pickle
import numpy as np
from tqdm import tqdm
from option import args


def is_bin_file(filename, patch_size):
    return any(filename.endswith(ext) for ext in ['_ii_map_p{}.pt'.format(patch_size)])

def samplePatchesProg(img_dim, patch_size, n_samples, maxiter=5):
    # Sample patches using dart throwing (works well for sparse/non-overlapping patches)

    # estimate each sample patch area
    full_area = float(img_dim[0] * img_dim[1])
    sample_area = full_area / n_samples

    # get corresponding dart throwing radius
    radius = np.sqrt(sample_area / np.pi)
    minsqrdist = (2 * radius) ** 2

    # compute the distance to the closest patch
    def get_sqrdist(x, y, patches):
        if len(patches) == 0:
            return np.infty
        dist = patches - [x, y]
        return np.sum(dist ** 2, axis=1).min()

    # perform dart throwing, progressively reducing the radius
    rate = 0.96
    patches = np.zeros((n_samples, 2), dtype=int)
    xmin, xmax = 0, img_dim[1] - patch_size[1] - 1
    ymin, ymax = 0, img_dim[0] - patch_size[0] - 1
    for patch in tqdm(range(n_samples)):
        done = False
        while not done:
            for i in range(maxiter):
                x = random.randint(xmin, xmax)
                y = random.randint(ymin, ymax)
                sqrdist = get_sqrdist(x, y, patches[:patch, :])
                if sqrdist > minsqrdist:
                    patches[patch, :] = [x, y]
                    done = True
                    break
            if not done:
                radius *= rate
                minsqrdist = (2 * radius) ** 2

    return patches

def prunePatches(shape, patches, patchsize, imp):
    pruned = np.empty_like(patches)

    # Generate a set of regions tiling the image using snake ordering.
    def get_regions_list(shape, step):
        regions = []
        for y in range(0, shape[0], step):
            if y // step % 2 == 0:
                xrange = range(0, shape[1], step)
            else:
                xrange = reversed(range(0, shape[1], step))
            for x in xrange:
                regions.append((x, x + step, y, y + step))
        return regions

    # Split 'patches' in current and remaining sets, where 'cur' holds the
    # patches in the requested region, and 'rem' holds the remaining patches.
    def split_patches(patches, region):
        cur = np.empty_like(patches)
        rem = np.empty_like(patches)
        ccount, rcount = 0, 0
        for i in range(patches.shape[0]):
            x, y = patches[i, 0], patches[i, 1]
            if region[0] <= x < region[1] and region[2] <= y < region[3]:
                cur[ccount, :] = [x, y]
                ccount += 1
            else:
                rem[rcount, :] = [x, y]
                rcount += 1
        return cur[:ccount, :], rem[:rcount, :]

    # Process all patches, region by region, pruning them randomly according to
    # their importance value, ie. patches with low importance have a higher
    # chance of getting pruned. To offset the impact of the binary pruning
    # decision, we propagate the discretization error and take it into account
    # when pruning.
    rem = np.copy(patches)
    count, error = 0, 0
    for region in get_regions_list(shape, 4 * patchsize):
        cur, rem = split_patches(rem, region)
        for i in range(cur.shape[0]):
            x, y = cur[i, 0], cur[i, 1]
            if imp[y, x] - error > random.random():
                pruned[count, :] = [x, y]
                count += 1
                error += 1 - imp[y, x]
            else:
                error += 0 - imp[y, x]

    return pruned[:count, :]

def gen_psnr_darts(psnr_map_file, n_samples = 1000):
    # get psnr map
    with open(psnr_map_file, 'rb') as _f:
        psnr_map = pickle.load(_f)
    
    # get patches
    img_dim_lr = (psnr_map.shape[0]+patch_size_lr-1, psnr_map.shape[1]+patch_size_lr-1)
    # print(img_dim_lr, flush=True)

    darts_list = samplePatchesProg(img_dim_lr, (patch_size_lr,patch_size_lr), n_samples)
    # darts_uvp_list_lr = np.array(darts_list) # [x,y]
    darts_uv_list_lr = darts_list[...,::-1] # [x,y] -> [y,x]
    darts_p_list_lr = psnr_map[darts_uv_list_lr[:,0],darts_uv_list_lr[:,1]]
    darts_uvp_list_lr = np.hstack((darts_uv_list_lr,darts_p_list_lr.reshape(-1,1)))
    
    # sort index
    sort_index = np.argsort(darts_uvp_list_lr[:,-1])
    darts_uvp_list_lr = darts_uvp_list_lr[sort_index]

    psnr_file = psnr_map_file.replace("map", "list_darts")
    print("saving {}".format(psnr_file))
    with open(psnr_file, 'wb') as _f:
        pickle.dump(darts_uvp_list_lr, _f)


if __name__ == '__main__':
    ################## settings
    scale = 2
    patch_size_hr = 192
    patch_size_lr = patch_size_hr // scale
    threshold = 0.05
    ii_map_dir = os.path.join(args.dir_data, "DIV2K/bin/DIV2K_train_LR_bicubic/X{}/".format(scale))
    ##################

    all_files = os.listdir(ii_map_dir)
    files = []
    for f in all_files:
        if is_bin_file(f,patch_size_hr):
            files.append(f)
    files.sort()
    print('number of files:', len(files))
    t_start = time.time()
    for i, file in enumerate(files):
        # if i > 3: break
        # if i < 598: continue
        tic = time.time()

        print("[{}/{}] processing [{}]...".format(i, len(files), file))
        psnr_map_file = os.path.join(ii_map_dir, file)

        gen_psnr_darts(psnr_map_file)

        toc = time.time()
        print("time: {}".format(toc - tic))
    print("total time: {}".format(toc - t_start))