import wandb
from .common import evaluate_acc, evaluate_loss, load_model


class NetworkFold:
    def __init__(self, fold_cfg, log_wandb=True):
        self.fold_cfg  = fold_cfg
        self.log_wandb = log_wandb

    def __call__(self):
        accuracies = list()

        for i in range(self.fold_cfg.num_experiments):
            model_cls      = self.fold_cfg.models[i]["model"]
            model_args     = self.fold_cfg.models[i]["args"]
            loader_train   = self.fold_cfg.loaders[i]["loader_train"]
            loader_test    = self.fold_cfg.loaders[i]["loader_test"]
            exp_name       = self.fold_cfg.names[i]["experiment_name"]
            model_name     = self.fold_cfg.names[i]["model_name"]
            device         = self.fold_cfg.configs[i]["device"]
            fold_method    = self.fold_cfg.configs[i]["fold_method"]
            model_mod      = self.fold_cfg.configs[i]["model_mod"]
            root_path      = self.fold_cfg.root_path
            proj_name      = self.fold_cfg.proj_name

            model          = model_cls(**model_args)

            if self.log_wandb is True:
                desc = {"experiment": exp_name}

                wandb.init(
                    project=proj_name,
                    config=desc,
                    name=exp_name
                )

            if model_mod is not None:
                model = model_mod(model)

            load_model(model, "%s/pt_models/%s.pt" %(root_path, model_name))

            folded = fold_method(model, log_wandb=self.log_wandb)

            train_acc = evaluate_acc(folded, loader_train, device=device)
            test_acc = evaluate_acc(folded, loader_test, device=device)

            train_loss = evaluate_loss(folded, loader_train, device=device)
            test_loss = evaluate_loss(folded, loader_test, device=device)

            if self.log_wandb is True:
                wandb.log({"folded train acc": train_acc})
                wandb.log({"folded test acc": test_acc})
                wandb.log({"r": 1.0 - fold_method.r})

            print(test_acc)
            accuracies.append(test_acc)

        return accuracies

