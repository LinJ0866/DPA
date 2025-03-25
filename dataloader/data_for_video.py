import os
from PIL import Image, ImageEnhance
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
import config as config
import random
import torch

def cv_random_flip(img, label, flow):
    flip_flag = random.randint(0, 1)

    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        flow = flow.transpose(Image.FLIP_LEFT_RIGHT)

    return img, label, flow

def randomCrop(image, label, flow):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)

    return image.crop(random_region), label.crop(random_region), flow.crop(random_region)

def randomRotation(image,label,flow):
    mode = Image.BICUBIC

    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        flow = flow.rotate(random_angle, mode)
    
    return image,label,flow

def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5,15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0,20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0,30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    
    return image

def randomGaussian(image, mean=0.1, sigma=0.35):
    
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        
        return im
    
    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])

    return Image.fromarray(np.uint8(img))

def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])

    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0]-1)  
        randY = random.randint(0, img.shape[1]-1)  

        if random.randint(0,1) == 0:  
            img[randX, randY] = 0  
        else:  
            img[randX, randY] = 255 

    return Image.fromarray(img)  


class SalObjDataset(data.Dataset):
    def __init__(self, dataset_root, dataset, mode='train'):
        self.trainsize = config.TRAIN['img_size']
        
        if dataset == 'rdvs':
            lable_rgb = 'rgb'
            lable_depth = 'Depth'
            lable_gt = 'ground-truth'
            lable_flow = 'FLOW'

            if mode == 'train':
                data_dir = os.path.join(dataset_root, 'RDVS/train')
            else:
                data_dir = os.path.join(dataset_root, 'RDVS/test')
        elif dataset == 'vidsod_100':
            lable_rgb = 'rgb'
            lable_depth = 'depth'
            lable_gt = 'gt'
            lable_flow = 'flow'
            
            if mode == 'train':
                data_dir = os.path.join(dataset_root, 'vidsod_100/train')
            else:
                data_dir = os.path.join(dataset_root, 'vidsod_100/test')
        elif dataset == 'dvisal':
            lable_rgb = 'RGB'
            lable_depth = 'Depth'
            lable_gt = 'GT'
            lable_flow = 'flow'

            data_dir = os.path.join(dataset_root, 'DViSal_dataset/data')

            if mode == 'train':
                dvi_mode = 'train'
            else:
                dvi_mode = 'test_all'
        else:
            raise 'dataset is not support now.'
        
        if dataset == 'dvisal':
            with open(os.path.join(data_dir, '../', dvi_mode+'.txt'), mode='r') as f:
                subsets = set(f.read().splitlines())
        else:
            subsets = os.listdir(data_dir)
        
        self.main_image_dict = {}
        self.main_flow_dict = {}
        self.frames = []
        for video in subsets:
            video_path = os.path.join(data_dir, video)
            rgb_path = os.path.join(video_path, lable_rgb)
            gt_path = os.path.join(video_path, lable_gt)
            flow_path = os.path.join(video_path, lable_flow)
            frames = os.listdir(rgb_path)
            frames = sorted(frames)
            for frame in frames[:-1]:
                data = {}
                data['img_path'] = os.path.join(rgb_path, frame)
                if os.path.isfile(data['img_path']):
                    data['gt_path'] = os.path.join(gt_path, frame.replace('jpg', 'png'))
                    data['flow_path'] = os.path.join(flow_path, frame)
                    data['split'] = video
                    data['dataset'] = dataset
                    if video not in self.main_image_dict:
                        self.main_image_dict[video] = []
                        self.main_flow_dict[video] = []
                    self.main_image_dict[video].append(data['img_path'])
                    self.main_flow_dict[video].append(data['flow_path'])
                    self.frames.append(data)

        
        self.size = len(self.frames)
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        
        self.flows_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        frame = self.frames[index]
        _class = frame['split']

        image = self.rgb_loader(frame['img_path'])
        gt = self.binary_loader(frame['gt_path'])
        flow = self.rgb_loader(frame['flow_path'])
        
        image, gt, flow = cv_random_flip(image, gt, flow)
        image, gt, flow = randomCrop(image, gt, flow)
        image, gt, flow = randomRotation(image, gt, flow)
        
        image = colorEnhance(image)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        flow = self.flows_transform(flow)

        # try:
        ref_image = self.main_image_dict[_class]
        ref_flow = self.main_flow_dict[_class]
        # except:
        #     ref_image = self.sub_image_dict[_class]
        #     ref_flow = self.sub_flow_dict[_class]

        pairs = list(zip(ref_image, ref_flow))
        pairs = random.sample(pairs, 4)
        ref_image, ref_flow = zip(*pairs)

        ref_image_list = []
        ref_flow_list = []

        for i in range(4):
            ref_i = self.rgb_loader(ref_image[i])
            ref_f = self.rgb_loader(ref_flow[i])

            ref_i, _, ref_f = cv_random_flip(ref_i, ref_i, ref_f)
            ref_i, _, ref_f = randomCrop(ref_i, ref_i, ref_f)
            ref_i, _, ref_f = randomRotation(ref_i, ref_i, ref_f)

            ref_i = colorEnhance(ref_i)

            ref_i = self.img_transform(ref_i)
            ref_f = self.flows_transform(ref_f)

            ref_image_list.append(ref_i)
            ref_flow_list.append(ref_f)
        
        ref_images = torch.cat(ref_image_list, dim=0)
        ref_flows = torch.cat(ref_flow_list, dim=0)

        return image, gt, flow, ref_images, ref_flows
    
    def get_image_class_dict(self, images):
        image_class = {}
        for image in images:
            _class = ((image.split('/')[-1]).split('.')[0]).split('_')[0]
            if _class not in image_class:
                image_class[_class] = []
            
            image_class[_class].append(image)
        
        return image_class

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            
            return img.convert('L')

    def resize(self, img, gt, flow):
        assert img.size == gt.size and gt.size == flow.size
        
        w, h = img.size
        
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), flow.resize((w, h), Image.NEAREST)
        else:
            return img, gt, flow

    def __len__(self):
        return self.size


def get_loader(data_root, dataset_, shuffle=True, num_workers=12, pin_memory=False, mode='train'):

    dataset = SalObjDataset(data_root, dataset_, mode)
    if mode == 'test':
        bs = 1
    else:
        bs = config.TRAIN['batch_size']
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=bs,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=True)
    
    return data_loader
    
class SalObjDataset_test(data.Dataset):
    def __init__(self, val):
        self.testsize = config.TRAIN['img_size']
        
        self.images = []
        self.gts = []
        self.flows = []

        folder = os.path.join(config.DATA['data_root'], val)
        valid_list = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
        
        for valid_name in valid_list:
            image_root = os.path.join(config.DATA['data_root'], val, valid_name, "RGB") + "/"
            gt_root = os.path.join(config.DATA['data_root'], val, valid_name, "GT") + "/"
            flow_root = os.path.join(config.DATA['data_root'], val, valid_name, "FLOW") + "/"

            new_images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
            new_gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
            new_flows = [flow_root + f for f in os.listdir(flow_root) if f.endswith('.jpg') or f.endswith('.png')]

            new_images = sorted(new_images)
            new_gts = sorted(new_gts)
            new_flows = sorted(new_flows)

            for i in range(len(new_flows)):
                self.images.append(new_images[i])
                self.gts.append(new_gts[i])
                self.flows.append(new_flows[i])

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.flows = sorted(self.flows)

        self.image_dict = self.get_image_class_dict(self.images)
        self.flow_dict = self.get_image_class_dict(self.flows)
        
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.flows_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)), 
            transforms.ToTensor()])
        
        self.size = len(self.images)
    
    def __getitem__(self, index):
        _class = ((self.images[index].split('/')[-1]).split('.')[0]).split('_')[0]

        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        flow = self.rgb_loader(self.flows[index])

        image = self.transform(image)
        flow = self.flows_transform(flow)

        ref_image = self.image_dict[_class]
        ref_flow = self.flow_dict[_class]

        ref_image_list = []
        ref_flow_list = []

        for i in range(4):
            ref_i = self.rgb_loader(ref_image[i * (len(ref_image) // 4)])
            ref_f = self.rgb_loader(ref_flow[i * (len(ref_flow) // 4)])

            ref_i = self.transform(ref_i)
            ref_f = self.flows_transform(ref_f)

            ref_image_list.append(ref_i)
            ref_flow_list.append(ref_f)
        
        ref_images = torch.cat(ref_image_list, dim=0)
        ref_flows = torch.cat(ref_flow_list, dim=0)

        name = self.images[index].split('/')[-1]
        valid_name = self.images[index].split('/')[-3]
        
        image_for_post = self.rgb_loader(self.images[index])
        image_for_post = image_for_post.resize((self.testsize, self.testsize))
        
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        
        info = [gt.size, valid_name, name]
        
        gt = self.gt_transform(gt)
        
        return image, gt, flow, info, np.array(image_for_post), ref_images, ref_flows
    
    def get_image_class_dict(self, images):
        image_class = {}
        for image in images:
            _class = ((image.split('/')[-1]).split('.')[0]).split('_')[0]
            if _class not in image_class:
                image_class[_class] = []
            
            image_class[_class].append(image)
        
        return image_class

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            
            return img.convert('L')
    
    def __len__(self):
        return self.size

def get_testloader(val, shuffle=False, num_workers=12, pin_memory=False):
    dataset = SalObjDataset_test(val)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.TRAIN['batch_size'],
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    
    return data_loader