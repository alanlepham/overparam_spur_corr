from data.resample_utils import resample, get_counts
from enum import Enum, auto
import os
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.confounder_dataset import ConfounderDataset
from robustness.tools.breeds_helpers import (
    ClassHierarchy,
    make_entity13,
    make_entity30,
    make_living17,
    make_nonliving26,
    BreedsDatasetGenerator,
)
from robustness import datasets
from robustness.tools.breeds_helpers import setup_breeds
from robustness.tools import folder
from robustness.tools.breeds_helpers import print_dataset_info
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
from p_tqdm import p_map
from PIL import Image, ImageDraw
from random import randint
import numpy as np


class BreedsDataset(ConfounderDataset):
    """
    CelebA dataset (already cropped and centered).
    Note: idx and filenames are off by one.
    """

    def __init__(
        self,
        root_dir,
        target_name,
        confounder_names,
        model_type,
        augment_data,
        extra_args=None,
    ):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.augment_data = augment_data
        self.model_type = model_type

        # breeds_dataset_type = extra_args.breeds_external_group

        breeds_dataset_type = extra_args.breeds_dataset_type
        breeds_proportions = []
        if extra_args.breeds_proportions:
            breeds_proportions = [
                int(item) for item in extra_args.breeds_proportions.split(",")
            ]

        if breeds_dataset_type is None:
            breeds_dataset_type = "entity_13"

        pair = "mammal-bird"
        if extra_args.breeds_pair:
            pair = extra_args.breeds_pair

        np_data_groups = f"./log/basic_breeds/{pair}_groups.npy"
        np_data_groups_dots = f"./log/basic_breeds/{pair}_groups_dots.npy"

        # TODO assumption based on rise machines
        if os.path.exists("/data/imagenetwhole"):
            data_dir = "/data/imagenetwhole"
        else:
            data_dir = "/data/imagenet"

        info_dir = "./data/BREEDS-Benchmarks/imagenet_class_hierarchy/modified"

        classes_available = pair.split("-")
        self.classes_available = classes_available

        ret = None
        if breeds_dataset_type == "entity13":
            ret = make_entity13(info_dir, split="rand")
        elif breeds_dataset_type == "entity30":
            ret = make_entity30(info_dir, split="rand")
        elif breeds_dataset_type == "living17":
            ret = make_living17(info_dir, split="rand")
        elif breeds_dataset_type == "nonliving26":
            ret = make_nonliving26(info_dir, split="rand")
        elif breeds_dataset_type == "custom_level2":
            DG = BreedsDatasetGenerator(info_dir)
            ret = DG.get_superclasses(
                level=2, ancestor=None, split="rand", balanced=True, verbose=False
            )
        elif breeds_dataset_type == "custom_level3":
            DG = BreedsDatasetGenerator(info_dir)
            ret = DG.get_superclasses(
                level=3, ancestor=None, split="rand", balanced=True, verbose=False
            )

        superclasses, subclass_split, label_map = ret
        train_subclasses, test_subclasses = subclass_split

        if not (os.path.exists(info_dir) and len(os.listdir(info_dir))):
            print("Downloading class hierarchy information into `info_dir`")
            setup_breeds(info_dir)

        # -----------------
        # Loading Dataset
        # ----------------
        self.full_dataset = load_breeds_groups(classes_available, data_dir, train_subclasses, label_map, test_subclasses)

        self.breeds_extra_test_group = None
        if extra_args.breeds_external_group:
            self.breeds_extra_test_group = load_breeds_groups([extra_args.breeds_external_group], data_dir, train_subclasses, label_map, test_subclasses)

        # --------------------------
        # Relabel and Compute Groups
        # --------------------------

        print("Realign dataset classes to 0,1")
        self.full_dataset = relabel_dataset_targets(self.full_dataset)
        if extra_args.breeds_color_dots:
            groups = self.load_groups(
                np_data_groups_dots,
                compute_groups_using_dots,
                extra_args.reload_breeds_groups,
                extra_args.breeds_cpu
            )
        else:
            groups = self.load_groups(
                np_data_groups,
                compute_groups_dominant_color,
                extra_args.reload_breeds_groups,
                extra_args.breeds_cpu
            )

        self.full_dataset.groups = np.array(groups)

        self.full_dataset = unisonShuffleDataset(
            self.full_dataset
        )  # Shuffle dataset since in order

        self.y_array = self.full_dataset.targets
        self.n_classes = 2

        self.n_groups = 4
        self.group_array = self.full_dataset.groups

        self.split_dict = {"train": 0, "val": 1, "test": 2}
        self.split_num_groups = {"train": 4, "val": 4, "test": 4}

        validation_percent = 0.1
        test_percent = 0.2

        validation_size = int(len(self.full_dataset) * validation_percent)
        test_size = int(len(self.full_dataset) * test_percent)
        train_size = len(self.full_dataset) - validation_size - test_size

        group_counts = (
            np.arange(self.n_groups).reshape(-1, 1) == self.group_array
        ).sum(1)
        print("group counts", group_counts)
        print("sizes", train_size, validation_size, test_size)

        self.split_array = np.array(
            [self.split_dict["train"]] * train_size
            + [self.split_dict["val"]] * validation_size
            + [self.split_dict["test"]] * test_size
        )

        self.split_sizes = {0: train_size, 1: validation_size, 2: test_size}
        if breeds_proportions:
            if max(breeds_proportions) <= 1:
                counts = get_counts(self.split_array, self.group_array)
                for i in range(12):
                    counts[i // 4][i % 4] = (
                        breeds_proportions[i] * self.split_sizes[i / 4]
                    )
            else:
                # truncate_dataset(full_dataset, expected_size=sum(breeds_proportions))

                counts = defaultdict(dict)
                for i in range(12):
                    counts[i // 4][i % 4] = breeds_proportions[i]
                print("New Counts", counts)
            self.split_array = [0, 1, 2]
            self.split_array = resample(self.split_array, self.group_array, counts)
            print(get_counts(self.split_array, self.group_array))

        print("Completed initializing breeds")

    def __len__(self):
        return len(self.full_dataset)

    def __getitem__(self, idx):
        x = self.full_dataset[idx][0]  # originally stored as (data, group_item)
        y = self.y_array[idx]
        g = self.group_array[idx]
        return x, y, g

    def group_str(self, group_idx):
        # First two groups are label 0 and second two groups are label 1
        # class: mammal, bird external class: anthropod
    
        # 0, 0 -> 0 -> color group 0, class 0
        # 0, 1 -> 1 -> color group 0, class 1

        # 1, 0 -> 2 -> color group 1, class 0
        # 1, 1 -> 3 -> color group 1, class 1

        if group_idx >= 2:
            class_num = group_idx - 2
            class_name = self.classes_available[class_num]
            return f"Color 1, Class: f{class_name}"

        class_num = group_idx
        class_name = self.classes_available[class_num]
        return f"Color 0, Class: f{class_name}"

    def load_groups(
        self, groups_file_name, load_func, reload_breeds_groups, breeds_cpu
    ):
        if not os.path.exists(groups_file_name) or reload_breeds_groups:
            self.full_dataset.groups = np.array([-1] * len(self.full_dataset))
            self.full_dataset = unisonShuffleDataset(
                self.full_dataset
            )  # Shuffle dataset before computing groups
            num_cpus = breeds_cpu if breeds_cpu else 16
            groups = load_func(self.full_dataset, num_cpus=num_cpus)
            with open(groups_file_name, "wb") as f:
                np.save(f, groups)
        else:
            with open(groups_file_name, "rb") as f:
                groups = np.load(f)

def load_breeds_groups(classes_available, data_dir, train_subclasses, label_map, test_subclasses):
    dataset_source = datasets.CustomImageNet(data_dir, train_subclasses)
    loaders_source = dataset_loader_helper(dataset_source)
    train_set, test_set = loaders_source
    train_set_source = filter_dataset_with_subclasses(
        train_set, label_map, classes_available
    )
    test_set_source = filter_dataset_with_subclasses(
        test_set, label_map, classes_available
    )

    dataset_target = datasets.CustomImageNet(data_dir, test_subclasses)
    loaders_target = dataset_loader_helper(dataset_target)

    train_target, test_target = loaders_target
    train_set_target = filter_dataset_with_subclasses(
        train_target, label_map, classes_available
    )
    test_set_target = filter_dataset_with_subclasses(
        test_target, label_map, classes_available
    )
    train_set = concat_datasets_inplace(train_set_source, train_set_target)
    test_set = concat_datasets_inplace(test_set_source, test_set_target)
    full_dataset = concat_datasets_inplace(train_set, test_set)
    return full_dataset

def colored_white(arr, size=40, color="white"):
    return colored_circle(arr, size=size, color=color)


def colored_black(arr, size=40, color="black"):
    return colored_circle(arr, size=size, color=color)


def colored_circle(arr, size=40, color="white"):
    image = Image.fromarray(arr)
    image = image.resize((224, 224))
    draw = ImageDraw.Draw(image)
    loc = randint(size, 224 - size)

    draw.ellipse((loc, loc, loc + size, loc + size), fill=color)
    return np.array(image, dtype=np.uint8)


def compute_groups_using_dots(full_dataset):
    indices = full_dataset.targets == 0
    class0_white = indices[: len(full_dataset) // 2]
    class0_black = indices[len(full_dataset) // 2 :]

    indices = full_dataset.targets == 1
    class1_white = indices[: len(indices) // 2]
    class1_black = indices[len(indices) // 2 :]

    full_dataset.samples[class0_white] = p_map(
        colored_white, full_dataset.samples[class0_white]
    )
    full_dataset.samples[class1_white] = p_map(
        colored_white, full_dataset.samples[class1_white]
    )

    full_dataset.samples[class0_black] = p_map(
        colored_black, full_dataset.samples[class0_black]
    )
    full_dataset.samples[class1_black] = p_map(
        colored_black, full_dataset.samples[class1_black]
    )

    groups = np.array([0] * len(full_dataset))

    groups[class0_white] = 0
    groups[class1_white] = 1

    groups[class0_black] = 2
    groups[class1_black] = 3
    return groups


def truncate_dataset(full_dataset, expected_size=1000):
    full_dataset.samples = full_dataset.samples[:expected_size]
    full_dataset.targets = full_dataset.targets[:expected_size]
    full_dataset.groups = full_dataset.groups[:expected_size]


def compute_groups_dominant_color(full_dataset, num_cpus=16):
    print("Computing dominant color metrics for dataset")
    samples = p_map(lambda item: item[0], full_dataset, num_cpus=num_cpus)
    dominant_metrics_computed = p_map(compute_dominant, samples, num_cpus=num_cpus)

    print("Computing labels for dataset")
    clt = KMeans(n_clusters=2)
    labels = clt.fit_predict(dominant_metrics_computed)
    print("Cluster centers", clt.cluster_centers_)

    print(labels)
    print("Relabeling groups for classification")
    groups = []
    for i in range(len(labels)):
        group_num = labels[i] * 2 + full_dataset.targets[i]
        # 0, 0 -> 0 -> color group 0, class 0
        # 0, 1 -> 1 -> color group 0, class 1

        # 1, 0 -> 2 -> color group 1, class 0
        # 1, 1 -> 3 -> color group 1, class 1

        groups.append(group_num)

    return groups


def unisonShuffleDataset(data_set):
    p = np.random.permutation(len(data_set))
    data_set.samples = np.array(data_set.samples)[p]
    data_set.targets = np.array(data_set.targets)[p]
    data_set.groups = np.array(data_set.groups)[p]
    return data_set


def relabel_dataset_targets(train_set):
    train_set.targets = np.unique(train_set.targets, return_inverse=1)[1]
    return train_set


def concat_datasets_inplace(train_set1, train_set2):
    train_set1.samples = train_set1.samples + train_set2.samples
    train_set1.targets = train_set1.targets + train_set2.targets
    train_set1.groups = train_set1.groups + train_set2.groups
    return train_set1


def filter_dataset_with_subclasses(
    train_set, label_map, allowed_classes=["reptile", "arthropod"]
):
    new_samples = []
    new_targets = []

    for i, target in enumerate(train_set.targets):
        class_name = label_map[target].split(",")[0]
        if class_name not in allowed_classes:
            continue

        new_samples += [train_set.samples[i]]
        new_targets += [target]
    train_set.samples = new_samples
    train_set.targets = new_targets
    train_set.groups = []
    return train_set


def dataset_loader_helper(dataset_source):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return load_dataset_values(
        transforms=(train_transform, test_transform),
        data_path=dataset_source.data_path,
        data_aug=True,
        dataset=dataset_source.ds_name,
        label_mapping=dataset_source.label_mapping,
        custom_class=dataset_source.custom_class,
        custom_class_args=dataset_source.custom_class_args,
    )


def load_dataset_values(
    transforms,
    data_path,
    data_aug=True,
    custom_class=None,
    dataset="",
    label_mapping=None,
    only_val=False,
    custom_class_args=None,
):
    """
    **INTERNAL FUNCTION**

    This is an internal function that makes a loader for any dataset. You
    probably want to call dataset.make_loaders for a specific dataset,
    which only requires workers and batch_size. For example:

    >>> cifar_dataset = CIFAR10('/path/to/cifar')
    >>> train_loader, val_loader = cifar_dataset.make_loaders(workers=10, batch_size=128)
    >>> # train_loader and val_loader are just PyTorch dataloaders
    """
    print(f"==> Preparing dataset {dataset}..")
    transform_train, transform_test = transforms
    if not data_aug:
        transform_train = transform_test

    if not custom_class:
        train_path = os.path.join(data_path, "train")
        test_path = os.path.join(data_path, "val")
        if not os.path.exists(test_path):
            test_path = os.path.join(data_path, "test")

        if not os.path.exists(test_path):
            raise ValueError(
                "Test data must be stored in dataset/test or {0}".format(test_path)
            )

        if not only_val:
            train_set = folder.ImageFolder(
                root=train_path, transform=transform_train, label_mapping=label_mapping
            )
        test_set = folder.ImageFolder(
            root=test_path, transform=transform_test, label_mapping=label_mapping
        )
    else:
        if custom_class_args is None:
            custom_class_args = {}
        if not only_val:
            train_set = custom_class(
                root=data_path,
                train=True,
                download=True,
                transform=transform_train,
                **custom_class_args,
            )
        test_set = custom_class(
            root=data_path,
            train=False,
            download=True,
            transform=transform_test,
            **custom_class_args,
        )

    return train_set, test_set


def get_dominant_color(image, k=4):
    """
    takes an image as input
    returns the dominant color of the image as a list

    dominant color is found by running k means on the
    pixels & returning the centroid of the largest cluster

    processing time is sped up by working with a smaller image;
    this resizing can be done with the image_processing_size param
    which takes a tuple of image dims as input

    >>> get_dominant_color(my_image, k=4, image_processing_size = (25, 25))
    [56.2423442, 34.0834233, 70.1234123]
    """

    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster and assign labels to the pixels
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(image)

    # count labels to find most popular
    label_counts = Counter(labels)

    # subset out most popular centroid
    top_most_common, second_most_common = label_counts.most_common(2)
    dominant_color = clt.cluster_centers_[top_most_common[0]]
    non_dominant_color = clt.cluster_centers_[second_most_common[0]]
    return dominant_color, non_dominant_color


def compute_average(image):
    image = image.numpy().transpose((1, 2, 0)) * 255

    average_color = np.round(image.mean(axis=0).mean(axis=0))
    return average_color


def compute_dominant(image):
    orig_image = image
    try:
        image_255 = image.numpy().transpose((1, 2, 0)) * 255

        dominant, _ = get_dominant_color(image_255, k=3)
        return dominant
    except ValueError:
        # In the case dominant color fails use average. TODO find invalid cases
        return compute_average(orig_image)
