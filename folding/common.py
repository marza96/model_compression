import torch
import tqdm
import numpy as np  


def save_model(model, i):
    sd = model.state_dict()
    torch.save(sd, i)


def load_model(model, i):
    sd = torch.load(i, map_location=torch.device('cpu'))
    model.load_state_dict(sd)


def evaluate_acc(model, loader, device="cuda", stop=10e6):
    model.eval()

    unique_labels = list()
    correct       = 0
    total         = 0

    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(loader, desc="eval"):
            outputs = model(inputs.to(device))
            pred = outputs.argmax(dim=1)
            correct += (labels.to(device) == pred).sum().item()
            total += len(labels)

            unique = torch.unique(labels).cpu().numpy().tolist()
            unique_labels += unique

            unique_labels = np.unique(unique_labels).tolist()

            if total > stop:
                break
    
    print("ACC: ", correct / total)
    return correct / total


def evaluate_loss(model, loader, device="cuda", stop=10e6):
    model.eval()

    loss_acum     = 0.0
    total         = 0

    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(loader, desc="eval"):
            outputs = model(inputs.to(device))

            loss = torch.nn.functional.cross_entropy(outputs, labels.to(device))

            loss_acum += loss.mean()

            total += 1
            
            if total > stop:
                break
            
    loss = loss_acum / total
    print("LOSS: ", loss)
