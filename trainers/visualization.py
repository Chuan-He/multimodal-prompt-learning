from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
# The default of 1,000 iterations gives fine results, but I'm training for longer just to eke
# out some marginal improvements. NB: This takes almost an hour!
# def compute_prototype(w):
#     tsne = TSNE(random_state=1, n_iter=15000, metric="cosine")

#     embs = tsne.fit_transform(w)
#     df = pd.DataFrame()
#     # Add to dataframe for convenience
#     df['x'] = embs[:, 0]
#     df['y'] = embs[:, 1]

def compute_prototype(model, data_loader):
    model.eval()
    count = 0
    embeddings = []
    embeddings_labels = []
    prototype= {}
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            count += 1
            inputs, labels = data['img'], data['label']
            # wrap them in Variable
            inputs = Variable(inputs.cuda())
            _ = model(inputs)
            embed_feat = model.image_features_pool.mean(dim=0)
            embeddings_labels.append(labels.numpy())
            embeddings.append(embed_feat.cpu().numpy())

    embeddings = np.asarray(embeddings)
    embeddings = np.reshape(
        embeddings, (embeddings.shape[0]*embeddings.shape[1], embeddings.shape[2]))
    embeddings_labels = np.asarray(embeddings_labels)
    embeddings_labels = np.reshape(
        embeddings_labels, embeddings_labels.shape[0]*embeddings_labels.shape[1])
    # labels_set = np.unique(embeddings_labels)
    # class_mean = []
    # class_std = []
    # class_label = []
    # for i in labels_set:
    #     ind_cl = np.where(i == embeddings_labels)[0]
    #     embeddings_tmp = embeddings[ind_cl]
    #     # embeddings_tmp = embeddings_tmp / np.linalg.norm(x=embeddings_tmp, ord=2)
    #     class_label.append(i)
    #     class_mean.append(np.mean(embeddings_tmp, axis=0))
    #     class_std.append(np.std(embeddings_tmp, axis=0))
    # prototype_new = {'class_mean': class_mean, 'class_std': class_std, 'class_label': class_label}
    # prototype = prototype_new
    np.save('./prototypes.npy', embeddings)
    np.save('./labels.npy',embeddings_labels)
    return embeddings

# def compute_prototype(model, data_loader, session):

    # model.eval()
    # count = 0
    # embeddings = []
    # embeddings_labels = []
    # prototype= {}
    # with torch.no_grad():
    #     for i, data in enumerate(data_loader, 0):
    #         count += 1
    #         inputs, labels = data
    #         # wrap them in Variable
    #         inputs = Variable(inputs.cuda())
    #         embed_feat = model.features(inputs)
    #         embeddings_labels.append(labels.numpy())
    #         embeddings.append(embed_feat.cpu().numpy())

    # embeddings = np.asarray(embeddings)
    # embeddings = np.reshape(
    #     embeddings, (embeddings.shape[0]*embeddings.shape[1], embeddings.shape[2]))
    # embeddings_labels = np.asarray(embeddings_labels)
    # embeddings_labels = np.reshape(
    #     embeddings_labels, embeddings_labels.shape[0]*embeddings_labels.shape[1])
    # labels_set = np.unique(embeddings_labels)
    # class_mean = []
    # class_std = []
    # class_label = []
    # for i in labels_set:
    #     ind_cl = np.where(i == embeddings_labels)[0]
    #     embeddings_tmp = embeddings[ind_cl]
    #     # embeddings_tmp = embeddings_tmp / np.linalg.norm(x=embeddings_tmp, ord=2)
    #     class_label.append(i)
    #     class_mean.append(np.mean(embeddings_tmp, axis=0))
    #     class_std.append(np.std(embeddings_tmp, axis=0))
    # # class_mean = class_mean / np.linalg.norm(x=class_mean, ord=2)
    # # class_std = class_std / np.linalg.norm(x=class_std, ord=2)

    # prototype_new = {'class_mean': class_mean, 'class_std': class_std, 'class_label': class_label}
    # if session != 0:
    #     prototype = np.load('./proto/prototypes.npy', allow_pickle=True).item()
    #     prototype['class_mean'].extend(prototype_new['class_mean'][:])
    #     prototype['class_std'].extend(prototype_new['class_std'][:])
    #     prototype['class_label'].extend(prototype_new['class_label'][:])
    # else:
    #     prototype = prototype_new

    # np.save('./proto/prototypes.npy', prototype)
    # return prototype