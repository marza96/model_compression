import os
import wandb
import torch
import torchvision

import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.functional import rotate


from tqdm import tqdm
from .common import evaluate_acc


def save_model(model, i):
    sd = model.state_dict()
    torch.save(sd, i)


def load_model(model, i):
    sd = torch.load(i)
    model.load_state_dict(sd)


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        # torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def train_loop(*, model, optimizer, loss_fn, epochs, train_loader, test_loader, device="cuda", scheduler=None):
    model.apply(init_weights)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5)

    for _ in tqdm(range(epochs)):
        model.train()
        loss_acum = 0.0
        total = 0
        for _, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs.to(device))
            loss    = loss_fn(outputs, labels.to(device))

            loss.backward()

            loss_acum += loss.mean()
            total += 1
            optimizer.step()
            
        train_loss = loss_acum / total

        unique_labels = list()
        correct = 0
        total_lab = 0
        
        test_loss = None
        with torch.no_grad():
            model.eval()
            loss_acum = 0.0
            total = 0
            for _, (inputs, labels) in enumerate(test_loader):
                outputs = model(inputs.to(device))
                loss    = loss_fn(outputs, labels.to(device))

                loss_acum += loss.mean()
                total += 1

                pred = outputs.argmax(dim=1)
                correct += (labels.to(device) == pred).sum().item()
                total_lab += len(labels)

                unique = torch.unique(labels).cpu().numpy().tolist()
                unique_labels += unique

                unique_labels = np.unique(unique_labels).tolist()

            test_loss = loss_acum / total
            test_acc  = correct / total_lab

            wandb.log({"train_loss": train_loss, "test_loss": test_loss, "test_acc": test_acc})

        # scheduler.step(test_loss)
        if scheduler is not None:
            print("SCHEDULE", scheduler)
            scheduler.step()
        
    print("TRAIN ACC:", evaluate_acc(model, train_loader, device=device))

    return model


def train_from_cfg(train_cfg):
    for i in range(train_cfg.num_experiments):
        model_cls      = train_cfg.models[i]["model"]
        model_args     = train_cfg.models[i]["args"]
        optimizer_args = train_cfg.configs[i]["optimizer"]["args"]
        optimizer_cls  = train_cfg.configs[i]["optimizer"]["class"]
        epochs         = train_cfg.configs[i]["epochs"]
        device         = train_cfg.configs[i]["device"]
        loss_fn        = train_cfg.configs[i]["loss_fn"]
        scheduler      = train_cfg.configs[i]["scheduler"]
        model_mod      = train_cfg.configs[i]["model_mod"]
        train_loader   = train_cfg.loaders[i]["train"]
        test_loader    = train_cfg.loaders[i]["test"]
        exp_name       = train_cfg.names[i]
        root_path      = train_cfg.root_path
        model          = model_cls(**model_args).to(device)
        optimizer      = optimizer_cls(model.parameters(), **optimizer_args)

        desc = {key: optimizer_args[key] for key in optimizer_args.keys()}
        desc.update(
            {"experiment": exp_name}
        )

        proj_name = train_cfg.proj_name

        if scheduler is not None:
            print("HAVE SCHEDULER")
            scheduler = scheduler[0](optimizer, *scheduler[1])

        wandb.init(
            project=proj_name,
            config=desc,
            name=exp_name
        )

        if model_mod is not None:
            model = model_mod(model)

        trained_model = train_loop(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            scheduler=scheduler
        )

        wandb.finish()

        os.makedirs("%s/pt_models/" %(root_path), exist_ok=True)
        save_model(trained_model, "%s/pt_models/%s.pt" %(root_path, exp_name))
