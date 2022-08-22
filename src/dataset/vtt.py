import logging
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image, ImageFile
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

import src.utils.datatool as dtool
from src.dataset.text import SimpleTokenizer
from src.dataset.vision import ConsistentTransform, VideoFrameReader

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# fmt: off
CATEGORIES = ['Dish', 'Drink and Snack', 'Electrical Appliance', 'Furniture and Decoration', 'Gadgets', 'Housework', 'Leisure and Performance', 'Nursing and Care', 'Pets and Fruit', 'Science and Craft', 'Sport', 'Vehicle']
TOPICS = ['Add Oil to Your Car', 'Arc Weld', 'Assemble Bed', 'Assemble Cabinet', 'Assemble Desktop PC', 'Assemble Office Chair', 'Assemble Sofa', 'Attend N B A Skills Challenge', 'Bandage Dog Paw', 'Bandage Head', 'Blow Sugar', 'Boil Noodles', 'Build Simple Floating Shelves', 'Carve Pumpkin', 'Change Battery Of Watch', 'Change Bike Chain', 'Change Bike Tires', 'Change Car Tire', 'Change Guitar Strings', 'Change Mobile Phone Battery', 'Change Toner Cartridge', 'Change a Tire', 'Clean Bathtub', 'Clean Cement Floor', 'Clean Fish', 'Clean Hamster Cage', 'Clean Laptop Keyboard', 'Clean Leather Seat', 'Clean Rusty Pot', 'Clean Shrimp', 'Clean Toilet', 'Clean Wooden Floor', 'Cook Omelet', 'Cut And Restore Rope Trick', 'Cut Cantaloupe', 'Cut Grape Fruit', 'Cut Mango', 'Do Lino Printing', 'Draw Blood', 'Drill Hole', 'Fix Laptop Screen Scratches', 'Fuel Car', 'Give An Intramuscular Injection', 'Glue Ping Pong Rubber', 'Graft', 'Grill Steak', 'Hang Wallpaper', 'Install Air Conditioner', 'Install Bicycle Rack', 'Install Ceiling Fan', 'Install Ceramic Tile', 'Install Closestool', 'Install Curtain', 'Install License Plate Frame', 'Install Shower Head', 'Install Wood Flooring', 'Iron Clothes', 'Jack Up a Car', 'Load Grease Gun', 'Lubricate A Lock', 'Make Banana Ice Cream', 'Make Bed', 'Make Bread and Butter Pickles', 'Make Burger', 'Make Candle', 'Make Chinese Lantern', 'Make Chocolate', 'Make Cocktail', 'Make Coffee', 'Make Cookie', 'Make Flower Crown', 'Make Flower Press', 'Make French Fries', 'Make French Strawberry Cake', 'Make French Toast', 'Make Homemade Ice Cream', 'Make Hummus', 'Make Irish Coffee', 'Make Jello Shots', 'Make Kerala Fish Curry', 'Make Kimchi Fried Rice', 'Make Lamb Kebab', 'Make Lemonade', 'Make Matcha Tea', 'Make Meringue', 'Make Orange Juice', 'Make Pancakes', 'Make Paper Dice', 'Make Paper Easter Baskets', 'Make Paper Wind Mill', 'Make Pickles', 'Make Pizza', 'Make RJ45 Cable', 'Make Salad', 'Make Salmon', 'Make Sandwich', 'Make Slime With Glue', 'Make Soap', 'Make Strawberry Smoothie', 'Make Sugar Coated Haws', 'Make Taco Salad', 'Make Tea', 'Make Wireless Earbuds', 'Make Youtiao', 'Make a Latte', 'Open A Lock With Paperclips', 'Open Champagne Bottle', 'Operate Fire Extinguisher', 'Pack Sleeping Bag', 'Park Parallel', 'Paste Car Sticker', 'Paste Screen Protector On Pad', 'Patch Bike Inner Tube', 'Perform CPR', 'Perform Paper To Money Trick', 'Perform Vanishing Glass Trick', 'Pitch A Tent', 'Plant Tree', 'Play Curling', 'Play Frisbee With A Dog', 'Polish Car', 'Practise Karate', 'Practise Pole Vault', 'Practise Skiing Aerials', 'Practise Triple Jump', 'Practise Weight Lift', 'Prepare Canvas', 'Prepare Standard Solution', 'Prepare Sumi Ink', 'Pump Up Bicycle Tire', 'Put On Hair Extensions', 'Put On Quilt Cover', 'Raise Flag', 'Refill A Lighter', 'Refill A Stapler', 'Refill Cartridge', 'Refill Fountain Pen', 'Refill Mechanical Pencils', 'Remove Blackheads With Glue', 'Remove Crayon From Walls', 'Remove Scratches From Windshield', 'Replace A Bulb', 'Replace A Wiper Head', 'Replace Battery On Key To Car', 'Replace Battery On TV Control', 'Replace Blade Of A Saw', 'Replace CD Drive With SSD', 'Replace Car Fuse', 'Replace Car Window', 'Replace Door Knob', 'Replace Drumhead', 'Replace Electrical Outlet', 'Replace Faucet', 'Replace Filter For Air Purifier', 'Replace Graphics Card', 'Replace Hard Disk', 'Replace Laptop Screen', 'Replace Light Socket', 'Replace Memory Chip', 'Replace Mobile Screen Protector', 'Replace Rearview Mirror Glass', 'Replace Refrigerator Water Filter', 'Replace SIM Card', 'Replace Sewing Machine Needle', 'Replace Toilet Seat', 'Replace Tyre Valve Stem', 'Resize Watch Band', 'Rewrap Battery', 'Roast Chestnut', 'Set Up A Hamster Cage', 'Shave Beard', 'Smash Garlic', 'Sow', 'Throw Hammer', 'Tie Boat To Dock', 'Transplant', 'Unclog Sink With Baking Soda', 'Use Analytical Balance', 'Use Earplugs', 'Use Epinephrine Auto-injector', 'Use Jack', 'Use Neti Pot', 'Use Rice Cooker To Cook Rice', 'Use Sewing Machine', 'Use Soy Milk Maker', 'Use Tapping Gun', 'Use Toaster', 'Use Triple Beam Balance', 'Use Vending Machine', 'Use Volumetric Flask', 'Use Volumetric Pipette', 'Wash Dish', 'Wash Dog', 'Wash Hair', 'Wear Contact Lenses', 'Wear Shin Guards', 'Wrap Gift Box', 'Wrap Zongzi']
# fmt: on


class VTTDataset(Dataset):

    CHANNEL = 3

    def __init__(
        self,
        split: str = "train",
        data_root: str = "/data/vtt",
        meta_path: str = "meta/vtt.jsonl",
        state_root: str = "states",
        frame_root: str = "frames",
        max_transformations: int = 12,
        max_words: int = 24,
        prefix_start: bool = True,
        suffix_end: bool = True,
        load_trans_frames: bool = False,
        n_segment: int = 3,
        frames_per_segment: int = 1,
        transform_cfg: Dict[str, Any] = {},
        return_raw_text: bool = False,
    ):
        self.data_root = Path(data_root).expanduser()
        self.data = dtool.JSONLList(
            self.data_root / meta_path, lambda x: x["split"] == split
        )
        self.state_root = self.data_root / state_root
        self.frame_root = self.data_root / frame_root

        self.max_transformations = max_transformations
        self.max_states = max_transformations + 1
        self.max_words = max_words
        self.prefix_start = prefix_start
        self.suffix_end = suffix_end

        self.load_frames = load_trans_frames
        self.tokenizer = SimpleTokenizer()
        self.transform = ConsistentTransform(**transform_cfg)
        if self.load_frames:
            self.n_segment = n_segment
            self.frames_per_segment = frames_per_segment
            self.video_reader = VideoFrameReader(
                n_segment=n_segment,
                frames_per_segment=frames_per_segment,
                list2tensor=True,
                transform=self.transform,
            )
        self.return_raw_text = return_raw_text

    def __len__(self):
        return len(self.data)

    def _read_states(self, sample):
        """n_states * C * H * W"""
        n_states = len(sample["annotation"]) + 1
        states_path_list = [
            self.state_root / f"{sample['id']}_{n_states}_{i}.jpg"
            for i in range(n_states)
        ]
        states = torch.zeros(
            self.max_states,
            self.CHANNEL,
            self.transform.n_px,
            self.transform.n_px,
            dtype=torch.float,
        )
        mask = torch.zeros(self.max_states, dtype=torch.bool)

        # the state of ConsistTransform is changed here only once for each sample
        # so that all states, frames are augmented (resize, crop, flip) with the
        # same arguments
        for (i, state_path) in enumerate(states_path_list):
            states[i] = self.transform(
                Image.open(str(state_path)), change_state=(i == 0)
            )
        mask[: len(states_path_list)] = True
        return states, mask

    def _read_labels(self, sample):
        """n_trans * n_words"""
        ids = torch.zeros(
            self.max_transformations, self.max_words, dtype=torch.int64
        )
        mask = torch.zeros(
            self.max_transformations, self.max_words, dtype=torch.bool
        )

        for i, step in enumerate(sample["annotation"]):
            words = torch.tensor(
                ([self.tokenizer.start_idx] if self.prefix_start else [])
                + self.tokenizer.encode(step["label"])
                + ([self.tokenizer.end_idx] if self.suffix_end else [])
            )
            ids[i, : len(words)] = words
            mask[i, : len(words)] = True

        return ids, mask

    def _read_categories(self, sample):
        category = CATEGORIES.index(sample["category"])
        topic = TOPICS.index(sample["topic"])
        return category, topic

    def _read_trans_frames(self, sample):
        """n_trans * T * C * H * W, T: frames sampled per video clip"""
        n_steps = len(sample["annotation"])
        clips_root_list = [
            self.frame_root / f"{sample['id']}_{n_steps}_{i}"
            for i in range(n_steps)
        ]
        clips = torch.zeros(
            self.max_transformations,
            self.n_segment * self.frames_per_segment,
            self.CHANNEL,
            self.transform.n_px,
            self.transform.n_px,
            dtype=torch.float,
        )
        mask = torch.zeros(self.max_transformations, dtype=torch.bool)

        for i, clip_root in enumerate(clips_root_list):
            clips[i] = self.video_reader.sample(clip_root)
        mask[: len(clips_root_list)] = True
        return clips, mask

    def __getitem__(self, index):
        meta = self.data[index]
        states, states_mask = self._read_states(meta)
        label_ids, label_mask = self._read_labels(meta)
        category, topic = self._read_categories(meta)
        res = {
            "label_ids": label_ids,
            "label_mask": label_mask,
            "states": states,
            "states_mask": states_mask,
            "category": category,
            "topic": topic,
        }
        if self.load_frames:
            trans, trans_mask = self._read_trans_frames(meta)
            res["trans"] = trans
            res["trans_mask"] = trans_mask
        if self.return_raw_text:
            res["text"] = [step["label"] for step in meta["annotation"]]
        return res


class VTTDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 6,
        pin_memory: bool = False,
        dataset_cfg: Dict[str, Any] = None,
        transform_cfg: Dict[str, Any] = None,
    ):

        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_cfg = dataset_cfg
        self.transform_cfg = transform_cfg

    def _dataloader(self, split):
        if split in ["train"]:
            dataset_cfg = self.dataset_cfg["train"]
            transform_cfg = self.transform_cfg["train"]
        else:
            dataset_cfg = self.dataset_cfg["eval"]
            transform_cfg = self.transform_cfg["eval"]

        dataset = VTTDataset(
            **dataset_cfg, split=split, transform_cfg=transform_cfg
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=(split == "train"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")

    def test_dataloader(self):
        return self._dataloader("test")
