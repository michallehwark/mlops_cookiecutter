import pytest
from tests import _PATH_DATA
import torch
from torch.utils.data import DataLoader, TensorDataset
import mlops_cc.models.predict_model
import os
import MyAwesomeModel

@pytest.mark.skipif(not os.path.exists("data/processed/images.pt"), reason="data images not found")
@pytest.mark.skipif(not os.path.exists("data/processed/labels.pt"), reason="data labels not found")
def test_train_data():
    
    images = torch.load("data/processed/images.pt")
    labels = torch.load("data/processed/labels.pt")
    # Test
    assert len(images) == len(labels) == 25000, "Testing sizes for images and/or labels do not match."

@pytest.mark.skipif(not os.path.exists(_PATH_DATA + "\\raw\\test.npz"), reason="test data not found")
def test_predict_data():

    test_images, test_labels = mlops_cc.models.predict_model.get_data(_PATH_DATA + "\\raw\\test.npz")
    # Teset
    assert len(test_images) == len(test_labels) == 5000, "Testing sizes for images and/or labels do not match."

def test_model_input_size():

    with pytest.raises(ValueError, match='Expected input 784, 10'):
        MyAwesomeModel(torch.randn(1,2))
    #with pytest.raises(ValueError, match='Expected input to a 4D tensor'):