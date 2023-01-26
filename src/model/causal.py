"""Nodes Definition.

Observable Nodes
---
- images: observable, corresponding to states
- transformations: observable,
- topic: observable,

- states: latent, corresponding to
- pattern: latent,
"""
import torch
from torch import nn

# Part 1: Image Encoder Decoder


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, images):
        pass


class ImageDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, states):
        pass


# Part 2: Transformation Description Encoder Decoder


class TransformationTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, descriptions):
        pass


class TransformationTextDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, transformations):
        pass


# Part 3: State Transformation Encoder Decoder
# (assume topic node is z)


class TopicEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn_topic = None

    def forward(self, initial_state, transformations):
        topic = self.fn_topic(initial_state, transformations)
        return topic


class StateDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn_initial_state = None
        self.fn_state = None

    def forward(self, transformations, topic):
        initial_state = self.fn_initial_state(topic)
        states = [initial_state]
        for transformation in transformations:
            state = self.fn_state(states[-1], transformation)
            states.append(state)
        return states


class TransformationDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn_transformation = None

    def forward(self, states, topic):
        transformations = []
        for state in states:
            transformation = self.fn_transformation(state, topic)
            transformations.append(transformation)
        return transformations
