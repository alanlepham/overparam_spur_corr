from enum import Enum
import os
from opt_einsum.paths import auto
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.confounder_dataset import ConfounderDataset
from robustness.tools.breeds_helpers import ClassHierarchy, make_entity13
from robustness import datasets
from robustness.tools.breeds_helpers import setup_breeds
from robustness.tools import folder
from robustness.tools.breeds_helpers import print_dataset_info


class BreedsSubgroupMode(Enum):
    SOURCE = auto()
    TARGET = auto()


def transform_groups(subgroup):
    breeds_groups = []
    for item in subgroup:
        class_name, breeds_type = item.split("-")
        breeds_type_enum = {
            item.name.lower(): item
            for item in [BreedsSubgroupMode.SOURCE, BreedsSubgroupMode.TARGET]
        }[breeds_type.lower()]
        breeds_groups += [[class_name, breeds_type_enum]]
    return breeds_groups


class BreedsDataset(ConfounderDataset):
    """
    CelebA dataset (already cropped and centered).
    Note: idx and filenames are off by one.
    """

    def __init__(
        self, root_dir, target_name, confounder_names, model_type, augment_data
    ):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.augment_data = augment_data
        self.model_type = model_type

        data_dir = "/data/imagenet"
        mode = "entity_13"
        info_dir = "./BREEDS-Benchmarks/imagenet_class_hierarchy/modified"

        subgroups_1 = ["reptile-Target", "arthropod-Source"]
        subgroups_2 = ["reptile-Source", "arthropod-Target"]

        groups_available = transform_groups(subgroups_1) + transform_groups(subgroups_2)
        self.groups_available = groups_available

        classes_available = set([item[0] for item in groups_available])

        ret = None
        if mode == "entity_13":
            ret = make_entity13(info_dir, split="rand")

        superclasses, subclass_split, label_map = ret
        train_subclasses, test_subclasses = subclass_split

        if not (os.path.exists(info_dir) and len(os.listdir(info_dir))):
            print("Downloading class hierarchy information into `info_dir`")
            setup_breeds(info_dir)

        dataset_source = datasets.CustomImageNet(data_dir, train_subclasses)
        loaders_source = dataset_loader_helper(dataset_source)

        train_set, test_set = loaders_source
        train_set_source = filter_dataset_with_subclasses(
            train_set,
            label_map,
            classes_available,
            BreedsSubgroupMode.SOURCE,
            groups_available,
        )
        test_set_source = filter_dataset_with_subclasses(
            test_set,
            label_map,
            classes_available,
            BreedsSubgroupMode.SOURCE,
            groups_available,
        )

        dataset_target = datasets.CustomImageNet(data_dir, test_subclasses)
        loaders_target = dataset_loader_helper(dataset_target)

        train_target, test_target = loaders_target
        train_set_target = filter_dataset_with_subclasses(
            train_target,
            label_map,
            classes_available,
            BreedsSubgroupMode.TARGET,
            groups_available,
        )
        test_set_target = filter_dataset_with_subclasses(
            test_target,
            label_map,
            classes_available,
            BreedsSubgroupMode.TARGET,
            groups_available,
        )

        self.train_set = concat_datasets_inplace(train_set_source, train_set_target)
        self.test_set = concat_datasets_inplace(test_set_source, test_set_target)

        self.full_dataset = concat_datasets_inplace(self.train_set, self.test_set)
        self.y_array = self.full_dataset.targets
        self.n_classes = 2

        # Map to groups
        self.n_groups = 4
        self.group_array = self.full_dataset.groups

        validation_size = 0.1
        self.split_array = (
            [self.split_dict["train"]] * len(self.train_set) * (1 - validation_size)
            + [self.split_dict["val"]] * len(self.train_set) * validation_size
            + [self.split_dict["val"]] * len(self.test_set)
        )
        self.split_dict = {"train": 0, "val": 1, "test": 2}

    def __getitem__(self, idx):
        x = self.full_dataset[idx]
        y = self.y_array[idx]
        g = self.group_array[idx]
        return x, y, g

    def group_str(self, group_idx):
        return self.groups_available[group_idx][0]


def concat_datasets_inplace(train_set1, train_set2):
    train_set1.samples = train_set1 + train_set2.samples
    train_set1.targets = train_set1 + train_set2.targets
    train_set1.groups = train_set1 + train_set2.groups


def filter_dataset_with_subclasses(
    train_set,
    label_map,
    allowed_classes=["reptile", "arthropod"],
    breeds_mode=BreedsSubgroupMode.SOURCE,
    subgroups_available=[],
):
    new_samples = []
    new_targets = []

    group_array = []
    for i, target in enumerate(train_set.targets[:10]):
        class_name = label_map[target].split(",")[0]
        if class_name not in allowed_classes:
            continue

        new_samples += [train_set.samples[i]]
        new_targets += [target]

        group_index = subgroups_available.index(
            [class_name, breeds_mode]
        )  # Use the index to determine the group

        group_array.append(group_index)

    train_set.samples = new_samples
    train_set.targets = new_targets
    train_set.groups = group_array


def dataset_loader_helper(dataset_source):
    return load_dataset_values(
        transforms=(dataset_source.train_transform, dataset_source.transform_test),
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
