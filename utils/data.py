import json
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

    dataset_name = str(args.get("dataset", "")).lower()
    imagenet_style_eval_datasets = {
        "domainnet",
        "sdomainet",
        "officehome",
        "nicopp",
        "cub",
        "food",
        "objectnet",
        "omnibenchmark",
        "vtab",
    }

    t = []
    if dataset_name.startswith("imagenet") or dataset_name in imagenet_style_eval_datasets:
        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
    else:
        t = [
            transforms.Resize((224, 224)),
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


DOMAINNET_OFFICIAL_TRAIN_TXT = "./utils/datautils/domainnet/train.txt"
DOMAINNET_OFFICIAL_TEST_TXT = "./utils/datautils/domainnet/test.txt"


def resolve_domainnet_txts(args):
    protocol = str(args.get("domainnet_protocol", "official")).lower()
    if protocol != "official" and (
        "domainnet_train_txt" not in args or "domainnet_test_txt" not in args
    ):
        raise ValueError(
            "DomainNet protocol '{}' requires domainnet_train_txt and "
            "domainnet_test_txt.".format(protocol)
        )

    return (
        args.get("domainnet_train_txt", DOMAINNET_OFFICIAL_TRAIN_TXT),
        args.get("domainnet_test_txt", DOMAINNET_OFFICIAL_TEST_TXT),
    )


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
        train_txt, test_txt = resolve_domainnet_txts(self.args)

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


class Food(iData):
    def __init__(self, args=None):
        super().__init__()
        self.args = args or {}
        self.use_path = True
        self.train_trsf = build_transform(True, self.args)
        self.test_trsf = build_transform(False, self.args)
        self.common_trsf = []
        self.class_order = np.arange(101).tolist()

    def download_data(self):
        rootdir = self.args.get("food_root", "./data/food")
        train_txt = self.args.get("food_train_txt", "./utils/datautils/food/train.txt")
        test_txt = self.args.get("food_test_txt", "./utils/datautils/food/test.txt")

        if os.path.isfile(train_txt) and os.path.isfile(test_txt):
            self.train_data, self.train_targets = _load_path_label_list(rootdir, train_txt)
            self.test_data, self.test_targets = _load_path_label_list(rootdir, test_txt)
            return

        official_root = _resolve_food101_root(rootdir)
        if official_root is not None:
            self.train_data, self.train_targets = _load_food101_split(official_root, "train")
            self.test_data, self.test_targets = _load_food101_split(official_root, "test")
            return

        train_dir = os.path.join(rootdir, "train")
        test_dir = os.path.join(rootdir, "test")
        if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
            raise FileNotFoundError(
                "Food dataset not found. Expected either custom food_train_txt/"
                "food_test_txt, Food-101 official layout under food_root or "
                "neighbor ./data/food-101 with images/ and meta/train.txt|json, "
                "or ImageFolder directories {}/train and {}/test. Current "
                "food_root={}".format(rootdir, rootdir, rootdir)
            )

        self.train_data, self.train_targets, self.test_data, self.test_targets = (
            _load_path_list_or_imagefolder_split(
                rootdir=rootdir,
                train_txt=train_txt,
                test_txt=test_txt,
                default_train_dir=train_dir,
                default_test_dir=test_dir,
            )
        )


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


def _resolve_food101_root(rootdir):
    candidates = [
        rootdir,
        os.path.join(rootdir, "food-101"),
        os.path.join(os.path.dirname(rootdir), "food-101"),
        "./data/food-101",
    ]
    seen = set()

    for candidate in candidates:
        candidate = os.path.normpath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)

        has_images = os.path.isdir(os.path.join(candidate, "images"))
        has_txt_split = (
            (
                os.path.isfile(os.path.join(candidate, "meta", "train.txt"))
                and os.path.isfile(os.path.join(candidate, "meta", "test.txt"))
            )
            or (
                os.path.isfile(os.path.join(candidate, "train.txt"))
                and os.path.isfile(os.path.join(candidate, "test.txt"))
            )
        )
        has_json_split = (
            (
                os.path.isfile(os.path.join(candidate, "meta", "train.json"))
                and os.path.isfile(os.path.join(candidate, "meta", "test.json"))
            )
            or (
                os.path.isfile(os.path.join(candidate, "train.json"))
                and os.path.isfile(os.path.join(candidate, "test.json"))
            )
        )

        if has_images and (has_txt_split or has_json_split):
            return candidate

    return None


def _load_food101_classes(rootdir):
    for classes_txt in (
        os.path.join(rootdir, "classes.txt"),
        os.path.join(rootdir, "meta", "classes.txt"),
        os.path.join(rootdir, "labels.txt"),
        os.path.join(rootdir, "meta", "labels.txt"),
    ):
        if os.path.isfile(classes_txt):
            with open(classes_txt, "r") as class_file:
                return [line.strip().split(maxsplit=1)[0] for line in class_file if line.strip()]

    images_dir = os.path.join(rootdir, "images")
    return sorted(
        entry.name for entry in os.scandir(images_dir) if entry.is_dir()
    )


def _load_food101_split(rootdir, split):
    classes = _load_food101_classes(rootdir)
    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
    split_txt_candidates = (
        os.path.join(rootdir, "{}.txt".format(split)),
        os.path.join(rootdir, "meta", "{}.txt".format(split)),
    )
    split_json_candidates = (
        os.path.join(rootdir, "{}.json".format(split)),
        os.path.join(rootdir, "meta", "{}.json".format(split)),
    )
    images = []
    labels = []

    split_txt = next((path for path in split_txt_candidates if os.path.isfile(path)), None)
    split_json = next((path for path in split_json_candidates if os.path.isfile(path)), None)

    if split_txt is not None:
        split_entries = _load_food101_txt_entries(split_txt)
    elif split_json is not None:
        split_entries = _load_food101_json_entries(split_json, split)
    else:
        raise FileNotFoundError(
            "Food-101 split file not found under {}: expected {}.txt or {}.json "
            "in the dataset root or meta/.".format(
                rootdir, split, split
            )
        )

    for rel_path in split_entries:
        image_path, class_name = _resolve_food101_image_entry(rootdir, rel_path)
        images.append(image_path)
        labels.append(class_to_idx[class_name])

    return np.array(images), np.array(labels)


def _load_food101_txt_entries(split_txt):
    entries = []
    with open(split_txt, "r") as split_file:
        for line in split_file:
            line = line.strip()
            if not line:
                continue
            entries.append(line.rsplit(maxsplit=1)[0])
    return entries


def _load_food101_json_entries(split_json, split):
    with open(split_json, "r") as split_file:
        metadata = json.load(split_file)

    if isinstance(metadata, list):
        return metadata

    if split in metadata and isinstance(metadata[split], list):
        return metadata[split]

    entries = []
    for class_name, class_entries in metadata.items():
        if not isinstance(class_entries, list):
            continue
        for entry in class_entries:
            if "/" in entry:
                entries.append(entry)
            else:
                entries.append(os.path.join(class_name, entry))
    return entries


def _resolve_food101_image_entry(rootdir, rel_path):
    rel_path = rel_path.strip()
    if rel_path.endswith((".jpg", ".jpeg", ".png")):
        rel_path_no_ext = os.path.splitext(rel_path)[0]
    else:
        rel_path_no_ext = rel_path

    class_name = rel_path_no_ext.split("/", 1)[0]
    image_path = os.path.join(rootdir, "images", rel_path_no_ext + ".jpg")
    return image_path, class_name


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
