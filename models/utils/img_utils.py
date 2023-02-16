import torch
import numpy as np
import matplotlib.pyplot as plt

cifar_label_idx_to_name = ["airplane",
"automobile",
"bird",
"cat",
"deer",
"dog",
"frog",
"horse",
"ship",
"truck"]

# def display_imgs(img_lst, label_lst, correct_label_lst):
#     fig = plt.figure(figsize=(20,8))
#     rows = 5 
#     cols = 4
#     for idx in np.arange(1, rows*cols + 1):
#         #ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])
#         #std = np.array([0.229, 0.224, 0.225])
#         #mean = np.array([0.485, 0.456, 0.406])
#         # img = img_lst[idx]
#         # img = img/2 + 0.5
#         # img = np.clip(img, 0, 1)
#         ax = fig.add_subplot(rows, cols, idx)
#         ax.set_title(f"{label_lst[idx]}: x%\n (label: {correct_label_lst[idx]})")
#         plt.imshow(img_lst[idx].transpose(1,2,0))

#     plt.show()


def display_imgs(img_lst, label_lst, correct_label_lst):
    fig = plt.figure(figsize=(10, 6))
    rows = 5 
    cols = 4
    for idx in np.arange(1, rows*cols + 1):
        ax = fig.add_subplot(rows, cols, idx)
        ax.set_title(f"predicted: {cifar_label_idx_to_name[label_lst[idx].squeeze()]} (label: {cifar_label_idx_to_name[correct_label_lst[idx].squeeze()]})",
                     fontdict={'fontsize':8})
        ax.axis('off')
        img = img_lst[idx]
        img = img/2 + 0.5
        img = np.clip(img, 0, 1)
        plt.imshow(img.transpose(1,2,0))
    #plt.subplots
    plt.show()

def display_imgs_gradcam(img_lst, label_lst, correct_label_lst, vis_lst):
    fig = plt.figure(figsize=(10, 6))
    rows = 5 
    cols = 4
    for idx in np.arange(1, rows*cols + 1):
        ax = fig.add_subplot(rows, cols, idx)
        ax.set_title(f"predicted: {cifar_label_idx_to_name[label_lst[idx].squeeze()]} (label: {cifar_label_idx_to_name[correct_label_lst[idx].squeeze()]})",
                     fontdict={'fontsize':8})
        ax.axis('off')
        img = img_lst[idx]
        img = img/2 + 0.5
        img = np.clip(img, 0, 1)
        plt.imshow(img.transpose(1,2,0))
        plt.imshow(vis_lst[idx], alpha=0.5)
    #plt.subplots
    plt.show()



def get_misclassified_imgs(model, test_dataloader):
    incorrect_imgs = []
    incorrect_labels = []
    incorrect_preds = []

    model.eval()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    for img, tgt in test_dataloader:
        img = img.to(device)
        tgt = tgt.to(device)

        output = model(img)
        pred = output.argmax(dim=1, keepdim=True)

        for idx in range(img.shape[0]):
            if tgt[idx] != pred[idx]:
                incorrect_imgs.append(img[idx].cpu().numpy())
                incorrect_preds.append(tgt[idx].cpu().numpy())
                incorrect_labels.append(pred[idx].cpu().numpy())

    return incorrect_imgs, incorrect_labels, incorrect_preds



