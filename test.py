import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import glob
from train import Net  # Changed to import Net instead of SimpleCNN

def check_model_architecture(model):
    """Check various architectural requirements of the model"""
    # Check for batch normalization
    has_batchnorm = any(isinstance(module, nn.BatchNorm2d) for module in model.modules())
    assert has_batchnorm, "Model must use batch normalization"
    
    # Check for dropout
    has_dropout = any(isinstance(module, nn.Dropout2d) for module in model.modules())
    assert has_dropout, "Model must use dropout"
    
    # Check for Global Average Pooling
    has_gap = any(isinstance(module, nn.AvgPool2d) for module in model.modules())
    assert has_gap, "Model must use Global Average Pooling"
    
    # Get parameter count
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {param_count:,}")
    assert param_count < 50000, f"Model has too many parameters: {param_count}"
    
    return {
        'has_batchnorm': has_batchnorm,
        'has_dropout': has_dropout,
        'has_gap': has_gap,
        'param_count': param_count
    }

def check_input_output(model):
    """Check input and output dimensions"""
    dummy_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(dummy_input)
        assert output.shape[1] == 10, f"Output size is not 10: {output.shape[1]}"
        print("Input/Output dimensions validated")
    except Exception as e:
        raise AssertionError(f"Model failed on 28x28 input: {e}")

def check_accuracy(model, testloader, device):
    """Check model accuracy on test set"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Model accuracy: {accuracy:.2f}%")
    assert accuracy > 70, f"Accuracy is too low: {accuracy}%"
    return accuracy

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Find and load the latest model
    model_files = glob.glob("model_*.pth")
    if not model_files:
        raise FileNotFoundError("No model files found. Please train a model first.")
    latest_model_file = max(model_files, key=os.path.getmtime)
    
    # Initialize and load model
    model = Net().to(device)
    model.load_state_dict(torch.load(latest_model_file))
    model.eval()
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    
    # Run all checks
    print("\nRunning model architecture checks...")
    arch_results = check_model_architecture(model)
    
    print("\nChecking input/output dimensions...")
    check_input_output(model)
    
    print("\nChecking model accuracy...")
    accuracy = check_accuracy(model, testloader, device)
    
    # Print summary
    print("\nTest Summary:")
    print(f"✓ Batch Normalization: {'Present' if arch_results['has_batchnorm'] else 'Missing'}")
    print(f"✓ Dropout: {'Present' if arch_results['has_dropout'] else 'Missing'}")
    print(f"✓ Global Average Pooling: {'Present' if arch_results['has_gap'] else 'Missing'}")
    print(f"✓ Parameter Count: {arch_results['param_count']:,}")
    # print(f"✓ Accuracy: {accuracy:.2f}%")
    
    print("\nAll tests passed successfully! 🎉")

if __name__ == "__main__":
    main()
