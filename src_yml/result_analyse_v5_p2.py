# %%
import numpy as np
import pandas as pd

# %%
labels_valid = pd.read_csv('tmp/labels_valid.csv')
real_labels = labels_valid.label.values
# %%
lbs_np = np.load('tmp/pred_lbs.npy')
lbs_np.shape
# %%
indexs = {
    'i0': 0,
    'i1': 1,
    'i2': 2,
    'i3': 3,
    'c0': 4,
    'c1': 5,
    'c2': 6,
    'c3': 7,
    'l0': 8,
    'l1': 9,
    'l2': 10,
    'l3': 11,
    'r0': 12,
    'r1': 13,
    'r2': 14,
    'r3': 15,
    'if0': 16,
    'if1': 17,
    'if2': 18,
    'if3': 19,
    'cf0': 20,
    'cf1': 21,
    'cf2': 22,
    'cf3': 23,
    'lf0': 24,
    'lf1': 25,
    'lf2': 26,
    'lf3': 27,
    'rf0': 28,
    'rf1': 29,
    'rf2': 30,
    'rf3': 31,
}
# %%


def tta(crop=True, flip=True, r90_270=True, r180=True):
    mask = [False]*32
    if crop:
        imgs = ['i', 'c', 'l', 'r']
    else:
        imgs = ['i']

    if flip:
        imgs_flip = [i+'f' for i in imgs]
        imgs = imgs+imgs_flip
    imgs_new = []
    for img in imgs:
        imgs_new.append(img+'0')
        if r90_270:
            imgs_new.append(img+'1')
        if r180:
            imgs_new.append(img+'2')
        if r90_270:
            imgs_new.append(img+'3')
    # print(imgs_new)
    for i in imgs_new:
        mask[indexs[i]] = True
    return mask


# %%
results = []
for crop in [True, False]:
    for flip in [True, False]:
        for r90_270 in [True, False]:
            for r180 in [True, False]:
                mask = tta(crop, flip, r90_270, r180)
                preds = []
                for img in lbs_np:
                    pred = np.argmax(img[mask].sum(axis=0))
                    preds.append(pred)
                pred_labels = np.array(preds)
                acc = (real_labels == pred_labels).sum()/real_labels.shape[0]
                results.append([crop, flip, r90_270, r180, sum(mask), acc])
    #             break
    #         break
    #     break
    # break
# %%
res_df = pd.DataFrame(
    results, columns=['crop', 'flip', 'r90_270', 'r180', 'tta', 'acc'])
# %%
res_df.sort_values(by='acc', ascending=False, inplace=True)

# %%
res_df.to_csv('tmp/tta.csv',index=False)

#%%
