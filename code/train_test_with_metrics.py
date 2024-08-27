import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

"""
*******************************************
Prerequisite Code:
Load datasets and DataLoaders,
Define model and move instance to device
"""
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, inputs):
        inputs_flat = self.flatten(inputs)
        logits = self.linear_relu_stack(inputs_flat)
        return logits


device = (
    # cuda is available on systems with NVIDIA GPUs
    "cuda" if torch.cuda.is_available()
    # mps is available on macOS systems that support Metal (Apple silicon e.g. M1/M2 chips)
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

nn_model = NeuralNetwork().to(device)
"""
*******************************************
"""

# define hyperparameters
learning_rate = 1e-3
batch_size = 64  # would be better before defining DataLoaders above
epochs = 5


def train_loop(train_loader, model, loss_fn, optimizer):
    ds_size = len(train_loader.dataset)  # total number of samples in training dataset

    # Set the model to training mode - important for batch_idx normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    # iterates training data in batches
    # for each batch loop, makes a prediction, calculates loss, optimises the parameters
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Compute prediction and loss
        pred = model(inputs)  # forward pass
        loss = loss_fn(pred, labels)

        # Backpropagation
        loss.backward()  # compute gradients of loss wrt model parameters
        optimizer.step()  # update model parameters based on computed gradients
        optimizer.zero_grad()  # reset gradients to prevent accumulating across batches

        # every 100 batches, log current loss and progress
        if batch_idx % 100 == 0:
            loss, samples_processed = loss.item(), (batch_idx + 1) * batch_size
            print(f"loss: {loss:>7f}  [{samples_processed:>5d}/{ds_size:>5d}]")


def test_loop(test_loader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalisation and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()

    ds_size = len(test_loader.dataset)  # total number of samples in test dataset
    num_batches = len(test_loader)

    # initialise cumulative total test loss
    total_test_loss = 0

    # initialise tensors to store predictions and labels
    all_preds = torch.tensor([], dtype=torch.long, device=device)  # ensure it's on the correct device
    all_labels = torch.tensor([], dtype=torch.long, device=device)  # ensure it's on the correct device

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        # iterates test data in batches
        for inputs, labels in test_loader:
            pred = model(inputs)  # forward pass, pred.shape = [64, 10] = [samples_in_batch, class_scores-logits_for_each_sample]
            total_test_loss += loss_fn(pred, labels).item()  # calculates loss for batch and accumulates it in total_test_loss

            # concatenate predictions (index of highest raw logit) and labels (index of correct class)
            all_preds = torch.cat((all_preds, pred.argmax(1)))
            all_labels = torch.cat((all_labels, labels))

    # average loss across all batches
    total_test_loss /= num_batches

    """
    METRICS
    Accuracy:
        Ratio of the number of correct predictions to the total number of samples.
        
        How many predictions were accurate - overall performance.
        Simple to understand and compute.
        Useful when classes are balanced and every prediction is equally important and want a general sense of models performance.
        Not informative if the dataset is imbalanced, as can give misleading sense of performance by overestimating the importance of the majority class.
        
    Precision:
        Ratio of true positive predictions to the sum of true and false positive predictions (or negative to sum of negatives).
        
        For those predicted class x, how many are actually class x.
        Useful for understanding how many of the positive (or negative) predictions were correct.
        Important when the cost of false positives is high, such as in spam detection or medical diagnosis (false positives could mean misdiagnosis).
        Can be misleading if not considered alongside recall.
        
    Recall (aka Recall Sensitivity):
        Ratio of true positive predictions to the sum of true positive and false negative predictions.
        
        For all in class x, how many were predicted class x.
        Performance with respect to positive class.
        Useful for understanding how many of the actual positives were correctly predicted.
        Crucial when the cost of false negatives is high, such as in detecting fraudulent transactions (missing a fraud is costly).
        Can be misleading if not considered alongside precision.
        
    False Negative Rate:
        Ratio of actual positives that are incorrectly classified as negatives.
        
        For all in class x, how many were incorrectly identified as NOT in class x.  
        Flip side of recall (sensitivity).  
        Provides insight into how often the model fails to identify the positive class and misses positive cases.  
        Crucial when the cost of missing a positive case (false negative) is high (e.g. in medical diagnosis could mean a significant number of patients with a disease are not being identified).  

    Specificity (aka True Negative Rate (TNR)):  
        Ratio of true negative predictions to the sum of true negative and false positive predictions - coverage of actual negative samples.  

        For all NOT in class x, how many were predicted NOT in class x.  
        Performance with respect to negative class.  
        Useful in scenarios where it is important to correctly identify the negative class. Particularly when negative class is much more common than positive class (e.g. the absence of a disease, non-fraudulent transactions).  
        High specificity indicates model is effectively reducing the number of false alarms (false positives).  

    False Positive Rate (FPR):  
        Ratio of false positive predictions to the sum of true negative and false positive predictions.  

        For all NOT in class x, how many were incorrectly identified as being in class x.  
        Flip side of specificity.  
        Provides insight into how often the model makes mistakes by labeling negative cases as positive. 
        
    F1-Score:
        The harmonic mean of precision and recall, providing a balance between them.
    
        Useful when you need to balance precision and recall, especially in cases of imbalanced datasets.
        It might not give a clear picture if the precision and recall vary significantly.
        
    Weighted metrics:
        Takes into account the number of samples (support) for each class when averaging. Classes with more samples have a larger influence on the final metric.
        Provides a more accurate representation of the model's performance on imbalanced datasets because it considers the distribution of classes,
        however it can mask poor performance on minority classes.
    """

    # calculate accuracy
    # if just calculating accuracy, 'correct' can be moved into for loop initialised as an int = 0;
    # then use: correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
    correct = (all_preds == all_labels).type(torch.float).sum().item()  # compares predicted class to true class (indices), converts to float tensor, sums correct predictions and extracts to float
    accuracy = correct / ds_size

    # total number of classes
    num_classes = torch.numel(torch.unique(all_labels))

    # initialize tensors for metrics
    precision_per_class = torch.zeros(num_classes)
    recall_per_class = torch.zeros(num_classes)
    f1_per_class = torch.zeros(num_classes)
    fnr_per_class = torch.zeros(num_classes)
    specificity_per_class = torch.zeros(num_classes)
    fpr_per_class = torch.zeros(num_classes)

    # calculate metrics for each class
    for cla in range(num_classes):
        true_positive = ((all_preds == cla) & (all_labels == cla)).sum().item()
        false_positive = ((all_preds == cla) & (all_labels != cla)).sum().item()
        false_negative = ((all_preds != cla) & (all_labels == cla)).sum().item()
        true_negative = ((all_preds != cla) & (all_labels != cla)).sum().item()

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        fnr = false_negative / (false_negative + true_positive) if (false_negative + true_positive) > 0 else 0
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
        fpr = false_positive / (false_positive + true_negative) if (false_positive + true_negative) > 0 else 0

        # unweighted metrics per-class
        precision_per_class[cla] = precision
        recall_per_class[cla] = recall
        fnr_per_class[cla] = fnr
        specificity_per_class[cla] = specificity
        fpr_per_class[cla] = fpr
        f1_per_class[cla] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # number of samples per class (support) for use in weighted calculations
    support_per_class = torch.tensor([(all_labels == i).sum().item() for i in range(num_classes)])

    # calculate weighted average metrics
    weighted_precision = (precision_per_class * support_per_class).sum().item() / ds_size
    weighted_recall = (recall_per_class * support_per_class).sum().item() / ds_size
    weighted_f1 = (f1_per_class * support_per_class).sum().item() / ds_size
    weighted_fnr = (fnr_per_class * support_per_class).sum().item() / ds_size
    weighted_specificity = (specificity_per_class * support_per_class).sum().item() / ds_size
    weighted_fpr = (fpr_per_class * support_per_class).sum().item() / ds_size

    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {total_test_loss:>8f}")  # accuracy displayed as percentage
    print(f" Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1-Score: {weighted_f1:.4f}")
    print(f" FNR: {weighted_fnr:.4f}, Specificity: {weighted_specificity:.4f}, FPR: {weighted_fpr:.4f}\n")


# define loss function and optimizer
ce_loss: nn.Module = nn.CrossEntropyLoss()
sgd_optimizer: torch.optim.Optimizer = torch.optim.SGD(nn_model.parameters(), lr=learning_rate)


# iterate train/test loops in epochs
# after each epoch, DataLoader will shuffle the samples if shuffle=True was defined which helps reduce overfitting
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, nn_model, ce_loss, sgd_optimizer)
    test_loop(test_dataloader, nn_model, ce_loss)
print("Done!")
