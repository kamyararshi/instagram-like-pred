import torch
from torchvision.transforms import Resize

def collate_fn(batch, img_size=(256,256)):
    
    resizer = Resize(img_size)
    datas, imgs, labels = zip(*batch)

    # Resizing images to img_size
    imgs_resized = []
    for sample in batch:
        _, img, _ = sample
        img = resizer(img).to(torch.float32)
        assert img.shape == (3, 256, 256), f"img.shape is {img.shape} but should have been 3, 256,256"
        imgs_resized.append(img)

    # Return eveuthing as tensors
    batch = (torch.stack(datas), torch.stack(imgs_resized), torch.stack(labels).to(torch.long))
    return batch
