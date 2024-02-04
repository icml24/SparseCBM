import torch
import transformers
import torch.nn as nn
import seaborn as sns
from configs import *
import matplotlib.pyplot as plt
import torch.nn.functional as F

class SparseCBLmodel_for_logits(nn.Module):
    """
    Basic architecture: CLIP model + two layers: CBL and FC (head)
    """
    def __init__(self, num_concepts: int, num_classes: int, model_name: str="openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = transformers.CLIPModel.from_pretrained(model_name)
        self.processor = transformers.CLIPProcessor.from_pretrained(model_name)
        for param in self.clip.parameters():
            param.requires_grad=False
        self.cbl = nn.Linear(num_concepts, num_concepts, bias=False)
        self.head = nn.Linear(num_concepts, num_classes, bias=False)
        
    def forward(self, **batch):
        cbl_out = self.cbl(self.clip(**batch).logits_per_image)
        return cbl_out, self.head(cbl_out)


def contrastive_loss(logits, dim: int):
    """
    Contrastive loss, adapted from https://sachinruk.github.io/blog/2021-03-07-clip.html
    """
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()

def criterion_cbl(similarity: torch.Tensor) -> torch.Tensor:
    """
    Args:
        similarity: is equal to logits_per_image
    """
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0

def gumbel_contrastive_loss(logits, hard: bool=False, dim: int=0):
    """
    Simple contrastive loss based on Gumbel-Softmax distribution
    """
    gumbel_softmax_samples = F.gumbel_softmax(logits, tau=1.0, dim=dim, hard=hard)
    neg_ce = -torch.log(gumbel_softmax_samples)
    return neg_ce.mean()

def criterion_gumbel(similarity: torch.Tensor, hard: bool=False) -> torch.Tensor:
    """
    Self-supervised contrastive loss for CBL training
    """
    first_loss = gumbel_contrastive_loss(similarity, hard=hard, dim=0)
    second_loss = gumbel_contrastive_loss(similarity, hard=hard, dim=1)
    return (first_loss + second_loss) / 2.0

def criterion_l1(model, l1_lambda=1e-3):
    """
    Implements L1 regularization.
    During training should be scaled by len(concepts)
    """
    l1_loss = 0.0
    for param in model.cbl.parameters():
        l1_loss += torch.norm(param, p=1)
        
    return l1_loss * l1_lambda

def draw_bottleneck(image, cbl_logits, k: int, concepts: list, draw_probs: bool=False):
    """
    Having a CBL outputs draws a bottleneck scores for a single PIL image
    """
    top_values, top_indices = torch.topk(cbl_logits, k)
    
    if draw_probs == True:
        top_values = torch.nn.functional.softmax(top_values, dim=-1)
    
    import pandas as pd
    data = pd.DataFrame({
        'Concepts': [concepts[i] for i in top_indices.squeeze().tolist()],
        'Probability': top_values.squeeze().tolist()
    })

    plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x='Probability', y='Concepts', data=data)
    plt.xlabel('Weight', fontsize=16)
    plt.ylabel('Concepts', fontsize=16)
    plt.title('Top {} Logits'.format(k), fontsize=16)

    for i, value in enumerate(top_values.squeeze().tolist()):
        plt.text(value + 0.01, i, f'{value:.2f}', va='center')
 
    plt.show()
