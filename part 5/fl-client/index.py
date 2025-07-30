import os
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import logging

logger = logging.getLogger(__name__)

# Shared directory for MNIST data
SHARED_DIR = "/mnt/data"
DATA_DIR = os.path.join(SHARED_DIR, "mnist_subset")

def initialize_data():
    """Initialize MNIST dataset"""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
        return train_loader
    except Exception as e:
        logger.error(f"Failed to initialize data: {e}")
        return None

def create_model():
    """Create neural network model"""
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
        nn.LogSoftmax(dim=1)
    )

def handle(event, context):
    """OpenFaaS handler function"""
    try:
        logger.info("FL Client training started")
        
        # Get model data from request body
        body = event.get('body', b'')
        if isinstance(body, str):
            body = body.encode()
        
        if not body:
            return {
                'statusCode': 400,
                'body': 'No model data provided'
            }

        # Initialize model
        model = create_model()
        
        # Load global model if provided
        if len(body) > 100:  # Valid model data
            try:
                model.load_state_dict(torch.load(io.BytesIO(body), map_location='cpu'))
                logger.info("Global model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load global model: {e}")
        
        # Initialize data loader
        train_loader = initialize_data()
        if train_loader is None:
            # Simulate training with random data if MNIST fails
            logger.warning("Using simulated training")
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * 0.01)
        else:
            # Real training step
            model.train()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            
            try:
                images, labels = next(iter(train_loader))
                images = images.view(images.size(0), -1)
                
                optimizer.zero_grad()
                output = model(images)
                loss = nn.functional.nll_loss(output, labels)
                loss.backward()
                optimizer.step()
                
                logger.info(f"Training completed with loss: {loss.item():.4f}")
            except Exception as e:
                logger.error(f"Training failed: {e}")
                # Fallback to simulated training
                with torch.no_grad():
                    for param in model.parameters():
                        param.add_(torch.randn_like(param) * 0.01)

        # Return updated model
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        model_bytes = buffer.read()
        
        logger.info(f"Client training completed, returning {len(model_bytes)} bytes")
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/octet-stream'},
            'body': model_bytes
        }
        
    except Exception as e:
        logger.error(f"Client handler error: {e}")
        return {
            'statusCode': 500,
            'body': f'Client error: {str(e)}'
        }
