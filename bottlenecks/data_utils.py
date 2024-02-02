import sys
from configs import *
import transformers

def collate_fn(batch):
    return {
        'image': [x['image'] for x in batch],
        'labels': [x['labels'] for x in batch]
    }

def remove_prefixes(strings):
    prefixes = ['a', 'an', 'the']
    result = []

    for string in strings:
        words = string.split()
        if words[0].lower() in prefixes:
            result.append(' '.join(words[1:]))
        else:
            result.append(string)

    return result

def preprocess_loader(loader, concepts: list):
    preprocessed_batches = []
    processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    for batch in tqdm(loader):
        preprocessed_batch = preprocess_batch(batch, processor, concepts)
        preprocessed_batches.append(preprocessed_batch)
    return preprocessed_batches

def preprocess_batch(batch, processor, concepts: list):
    return processor(text=concepts, images=batch['image'], return_tensors="pt", padding=True), batch['labels']

def prepared_dataloaders(hf_link: str, concepts: list, test_size: int=0.2, prep_loaders='all', batch_size: int=32):
    from datasets import load_dataset
    from datasets import DatasetDict
    dataset = load_dataset(hf_link)
    dataset = dataset["train"].train_test_split(test_size=0.2)
    val_test = dataset["test"].train_test_split(test_size=0.5)
    dataset = DatasetDict({
        "train": dataset["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    })

    train_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)

    if prep_loaders == 'all':
        train_loader_preprocessed = preprocess_loader(train_loader, concepts)
        val_loader_preprocessed = preprocess_loader(val_loader, concepts)
        test_loader_preprocessed = preprocess_loader(test_loader, concepts)
    elif prep_loaders == 'train':
        train_loader_preprocessed = preprocess_loader(train_loader, concepts)
        return train_loader_preprocessed
    elif prep_loaders == 'val':
        val_loader_preprocessed = preprocess_loader(val_loader, concepts)
        return val_loader_preprocessed
    elif prep_loaders == 'test':
        test_loader_preprocessed = preprocess_loader(test_loader, concepts)
        return test_loader_preprocessed

    #return train_loader_preprocessed, val_loader_preprocessed, test_loader_preprocessed
