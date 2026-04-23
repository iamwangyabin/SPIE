import os
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

def build_transform_coda_prompt(is_train, args):
    if is_train:        
        transform = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
        return transform

    t = []
    if args["dataset"].startswith("imagenet"):
        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
    else:
        t = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]

    return t

def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t


def build_imagenet_normalize():
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def use_vpt_nsp2pp_official_aug(args):
    return (
        args.get("model_name") == "vpt_nsp2pp"
        and str(args.get("augmentation_protocol", "benchmark")).lower() == "official"
    )


def build_vpt_nsp2pp_transform(is_train):
    bilinear = transforms.InterpolationMode.BILINEAR
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if is_train:
        return [
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET, interpolation=bilinear),
            transforms.RandomResizedCrop(224, interpolation=bilinear, antialias=True),
            transforms.ToTensor(),
            normalize,
        ]

    return [
        transforms.Resize((256, 256), interpolation=bilinear, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]


class iCIFAR224(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = False

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetR(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True

        if use_vpt_nsp2pp_official_aug(args):
            self.train_trsf = build_vpt_nsp2pp_transform(True)
            self.test_trsf = build_vpt_nsp2pp_transform(False)
            self.common_trsf = []
        elif args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
            self.common_trsf = [
                # transforms.ToTensor(),
            ]
        elif args["model_name"] == "arcl":
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
            self.common_trsf = [build_imagenet_normalize()]
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
            self.common_trsf = [
                # transforms.ToTensor(),
            ]

        self.class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/imagenet-r/train/"
        test_dir = "./data/imagenet-r/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iDomainNet(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True

        if use_vpt_nsp2pp_official_aug(args):
            self.train_trsf = build_vpt_nsp2pp_transform(True)
            self.test_trsf = build_vpt_nsp2pp_transform(False)
            self.common_trsf = []
        elif args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
            self.common_trsf = []
        elif args["model_name"] == "arcl":
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
            self.common_trsf = [build_imagenet_normalize()]
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
            self.common_trsf = []

        self.class_order = np.arange(200).tolist()

    def download_data(self):
        rootdir = self.args.get("domainnet_root", "./data/domainnet")
        train_txt = self.args.get(
            "domainnet_train_txt", "./utils/datautils/domainnet/train.txt"
        )
        test_txt = self.args.get(
            "domainnet_test_txt", "./utils/datautils/domainnet/test.txt"
        )

        train_images, train_labels = _load_path_label_list(rootdir, train_txt)
        test_images, test_labels = _load_path_label_list(rootdir, test_txt)

        self.train_data = train_images
        self.train_targets = train_labels
        self.test_data = test_images
        self.test_targets = test_labels


class iOfficeHome(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True

        if use_vpt_nsp2pp_official_aug(args):
            self.train_trsf = build_vpt_nsp2pp_transform(True)
            self.test_trsf = build_vpt_nsp2pp_transform(False)
            self.common_trsf = []
        elif args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
            self.common_trsf = []
        elif args["model_name"] == "arcl":
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
            self.common_trsf = [build_imagenet_normalize()]
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
            self.common_trsf = []

        self.class_order = np.arange(65).tolist()

    def download_data(self):
        rootdir = self.args.get("officehome_root", "./data/office-home")
        train_txt = self.args.get(
            "officehome_train_txt", "./utils/datautils/officehome/train.txt"
        )
        test_txt = self.args.get(
            "officehome_test_txt", "./utils/datautils/officehome/test.txt"
        )

        self.train_data, self.train_targets, self.test_data, self.test_targets = (
            _load_path_list_or_imagefolder_split(
                rootdir=rootdir,
                train_txt=train_txt,
                test_txt=test_txt,
                default_train_dir=os.path.join(rootdir, "train"),
                default_test_dir=os.path.join(rootdir, "test"),
            )
        )


class iNICOPP(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True

        if use_vpt_nsp2pp_official_aug(args):
            self.train_trsf = build_vpt_nsp2pp_transform(True)
            self.test_trsf = build_vpt_nsp2pp_transform(False)
            self.common_trsf = []
        elif args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
            self.common_trsf = []
        elif args["model_name"] == "arcl":
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
            self.common_trsf = [build_imagenet_normalize()]
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
            self.common_trsf = []

        self.class_order = np.arange(80).tolist()

    def download_data(self):
        rootdir = self.args.get("nicopp_root", "./data/nicopp")
        train_txt = self.args.get(
            "nicopp_train_txt", "./utils/datautils/nicopp/train.txt"
        )
        test_txt = self.args.get(
            "nicopp_test_txt", "./utils/datautils/nicopp/test.txt"
        )

        self.train_data, self.train_targets, self.test_data, self.test_targets = (
            _load_path_list_or_imagefolder_split(
                rootdir=rootdir,
                train_txt=train_txt,
                test_txt=test_txt,
                default_train_dir=os.path.join(rootdir, "train"),
                default_test_dir=os.path.join(rootdir, "test"),
            )
        )


class iImageNetA(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/imagenet-a/train/"
        test_dir = "./data/imagenet-a/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class CUB(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/cub/train/"
        test_dir = "./data/cub/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class objectnet(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/objectnet/train/"
        test_dir = "./data/objectnet/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class omnibenchmark(iData):
    def __init__(self, args=None):
        super().__init__()
        self.args = args or {}
        self.use_path = True
        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = []
        self.class_order = np.arange(300).tolist()

    def download_data(self):
        rootdir = self.args.get("omnibenchmark_root", "./data/omnibenchmark")
        train_dir = os.path.join(rootdir, "train")
        test_dir = os.path.join(rootdir, "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class vtab(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/vtab-cil/vtab/train/"
        test_dir = "./data/vtab-cil/vtab/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


def _load_path_label_list(rootdir, txt_path):
    images = []
    labels = []
    with open(txt_path, "r") as dict_file:
        for line in dict_file:
            line = line.strip()
            if not line:
                continue
            value, key = line.rsplit(" ", 1)
            images.append(os.path.join(rootdir, value))
            labels.append(int(key))

    return np.array(images), np.array(labels)


def _load_path_list_or_imagefolder_split(
    rootdir,
    train_txt,
    test_txt,
    default_train_dir,
    default_test_dir,
):
    if os.path.isfile(train_txt) and os.path.isfile(test_txt):
        train_images, train_labels = _load_path_label_list(rootdir, train_txt)
        test_images, test_labels = _load_path_label_list(rootdir, test_txt)
        return train_images, train_labels, test_images, test_labels

    train_dset = datasets.ImageFolder(default_train_dir)
    test_dset = datasets.ImageFolder(default_test_dir)
    train_images, train_labels = split_images_labels(train_dset.imgs)
    test_images, test_labels = split_images_labels(test_dset.imgs)
    return train_images, train_labels, test_images, test_labels
