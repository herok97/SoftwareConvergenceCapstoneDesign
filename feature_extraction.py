# -*- coding: utf-8 -*-
import h5py
from PIL import Image
from torch.backends import cudnn
from torchvision import transforms, models, datasets
import torch.nn as nn
import torch
from tqdm import tqdm
from time import sleep
class Rescale(object):
    """Rescale a image to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, *output_size):
        self.output_size = output_size

    def __call__(self, image):
        """
        Args:
            image (PIL.Image) : PIL.Image object to rescale
        """
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = image.resize((new_w, new_h), resample=Image.BILINEAR)
        return img


class ResNetFeature(nn.Module):
    def __init__(self, feature='resnet101'):
        """
        Args:
            feature (string): resnet101 or resnet152
        """
        super(ResNetFeature, self).__init__()
        if feature == 'resnet101':
            resnet = models.resnet101(pretrained=True)
        else:
            resnet = models.resnet152(pretrained=True)
        resnet.float()
        resnet.cuda()
        resnet.eval()

        module_list = list(resnet.children())
        self.conv5 = nn.Sequential(*module_list[:-2])
        self.pool5 = module_list[-2]

    def forward(self, x):
        res5c = self.conv5(x)
        pool5 = self.pool5(res5c)
        pool5 = pool5.view(pool5.size(0), -1)
        return res5c, pool5

resnet_transform = transforms.Compose([
        Rescale(224, 224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_vector(image):
    # Create a PyTorch tensor with the transformed image
    t_img = resnet_transform(image)
    # Create a vector of zeros that will hold our feature vector
    my_embedding = torch.zeros(2048)

    # Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.flatten())  # <-- flatten

    # Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # Run the model on our transformed image
    with torch.no_grad():  # <-- no_grad context
        model(t_img.cuda().unsqueeze(0))  # <-- unsqueeze
    # Detach our copy function from the layer
    h.remove()
    # Return the feature vector
    return my_embedding

if __name__ == "__main__":
    cudnn.benchmark = True
    from PIL import Image
    from torchvision.datasets.folder import default_loader

    # GPU 정보
    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)
    device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    print('기기:', device)
    print('graphic name:', torch.cuda.get_device_name())

    model = ResNetFeature()
    layer = model._modules.get('pool5')

    import os
    root_dir = "C:/Users/01079/video_summarization_data/only_video02/"
    save_dir = "C:/Users/01079/video_summarization_data/h5_02/"

    category = ['OVP', 'SumMe', 'TvSum']
    tt = ['train', 'test']

    # 어떤 종류의 동영상인지
    for cate in category:
        # test 인지 train 인지
        for t in tt:
            r_dir = root_dir + '/' + cate + '/' + t + '/'
            s_dir = save_dir + '/' + cate + '/' + t + '/'
            print(r_dir)
            video_list = os.listdir(r_dir)
            # 한 영상에 대한 이미지에 대해서

            for video in video_list:

                # 이미 h5파일 만들었으면 건너 뜀
                if os.path.isfile(os.path.join(s_dir, video + '.h5')):
                    continue

                # 이미지 폴더 접근
                folder = os.path.join(r_dir, video)

                print('folder:', folder)

                images = []
                img_names = [os.path.join(folder, name) for name in os.listdir(folder)]

                with tqdm(total=len(img_names)) as progress_bar:
                    for img_name in img_names:
                        try:
                            img = default_loader(img_name)
                        except:
                            print("Wrong file fomrat, perhaps it was 0 bytes image file")
                            print("i'm gonna duplicate previous image")
                        images.append(get_vector(img))
                        progress_bar.update(1)  # update progress

                # h5 file 생성
                with h5py.File(os.path.join(s_dir, video + '.h5'), 'w') as hf:
                    result = torch.stack(images)
                    hf.create_dataset('pool5', data=result)
                    hf.close()
                    print("h5 file 생성 완료", "result:", result.shape)
                
                # 캐시 메모리 해제
                torch.cuda.empty_cache()
