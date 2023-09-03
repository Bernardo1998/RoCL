from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
from torchvision import transforms
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from urllib import request
import zipfile
import json

from .vision import VisionDataset
from .utils import check_integrity, download_and_extract_archive


class TinyImagenet(VisionDataset):
    """`Tiny imagenet <http://vision.stanford.edu/teaching/cs231n/reports/2015/pdfs/yle_project.pdf>`
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'tiny-imagenet-200'
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    wnid = "wnids.txt"
    
    def __init__(self, root, train='train', transform=None, target_transform=None,
                 download=True, contrastive_learning=False):

        super(TinyImagenet, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # now a str: train/test/val
        self.learning_type = contrastive_learning

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.data = []
        self.targets = []
        
        wnid_path = os.path.join(self.root, self.base_folder, self.wnid)
        wnid_map = {}
        with open(wnid_path, "r") as wnidF:
        	for i,line in enumerate(wnidF):
        	    wnid = line.split('\t')[0].replace("\n","")
        	    wnid_map[wnid] = i
        #print(wnid_map)

        if self.train  == "train":
            image_folder = os.path.join(self.root, self.base_folder, self.train)
            subfolders = os.listdir(image_folder)
            for subfolder in subfolders:
                class_id = wnid_map[subfolder]
                images = os.listdir(os.path.join(image_folder, subfolder, "images"))
                for image in images:
                    image_path = os.path.join(image_folder, subfolder, "images" , image)
                    self.data.append(image_path)
                    self.targets.append(class_id)
        else:
            image_folder = os.path.join(self.root, self.base_folder, self.train, "images")
            annotations = os.path.join(self.root, self.base_folder, self.train, self.train+"_annotations.txt")
            with open(annotations, 'r') as antF:
                for line in antF:
                    elements = line.split('\t')
                    image_path = os.path.join(image_folder, elements[0])
                    class_id = wnid_map[elements[1]]
                    self.data.append(image_path)
                    self.targets.append(class_id)

        #self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        #self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        #self.data = np.random.randint(0,255,size=(len(self.data), 64,64,3)).astype(np.uint8)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # Read np array from an image path. Convert to RGB so all img has 3 channels, avoid 'not resizable' error.
        img = np.array(Image.open(img).convert('RGB')).astype(np.uint8)
        ori_img = img
        toTensor = transforms.ToTensor()
        ori_img = toTensor(ori_img)
        #print(type(img), img.dtype, img.shape, img.min(), img.max())
        
        if self.learning_type=='contrastive':
            img_2 = img.copy()

        elif self.learning_type=='linear_eval' or self.learning_type == "supervised":
            if self.train  == "train":
                img_2 = img.copy()

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img     = Image.fromarray(img)
        if self.learning_type=='contrastive':
            img_2   = Image.fromarray(img_2)
        elif self.learning_type=='linear_eval' or self.learning_type == "supervised":
            if self.train  == "train":
                img_2 = img.copy()

        if self.transform is not None:
            img     = self.transform(img)
            if self.learning_type=='contrastive':
                img_2   = self.transform(img_2)
            elif self.learning_type=='linear_eval' or self.learning_type == "supervised":
                if self.train  == "train":
                    img_2 = self.transform(img_2)

        if self.target_transform is not None:
            target = self.target_transform(target)
        #print(self.learning_type,self.train)
        if self.learning_type=='contrastive':
            return ori_img, img, img_2, target
        elif self.learning_type=='linear_eval' or self.learning_type == "supervised": # XF 11252022: change supervised to have the same output as linear_eval
            if self.train  == "train":
                return ori_img, img, img_2, target
            else:
                return img, target
        else:
            return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        # Ensure both wnid list and images in corresponding annotation files are found
        if not os.path.exists(os.path.join(root, self.base_folder)):
            print("Missing folder!")
            return False
        if self.wnid not in os.listdir(os.path.join(root, self.base_folder)):
            print("Missing wnid!")
            return False
        for fentry in ['val']:
            images = set(os.listdir(os.path.join(root, self.base_folder, fentry, "images")))
            filename = fentry + "_annotations.txt"
            fpath = os.path.join(root, self.base_folder, fentry, filename)
            with open(fpath, 'r') as anntFile:
                for line in anntFile:
                    image_name = line.split('\t')[0]
                    if not image_name in images:
                        print("Missing image!")
                        return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified.')
            return
        # Save and extract files
        #if not os.path.exists(os.path.join(self.root, self.base_folder)):
        #    os.mkdir(os.path.join(self.root, self.base_folder))
        local_path = os.path.join(self.root, self.filename)
        response = request.urlretrieve(self.url, local_path)
        with zipfile.ZipFile(local_path,"r") as zip_ref:
            zip_ref.extractall(os.path.join(self.root))

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
