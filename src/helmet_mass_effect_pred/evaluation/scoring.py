from torch import tensor as tnsr
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAccuracy,
    BinaryAUROC,
)


def OTV_scorer(
    wandb_logger,
    predictions,
    test_targets,
    target_col_names,
    probabilities=None,
    num_classes: int = 3,
):
    if num_classes > 2:
        ACC = MulticlassAccuracy(num_classes=num_classes, average=None)
        acc = ACC(tnsr(predictions), tnsr(test_targets))
        for i in range(num_classes):
            print(f"Accuracy for target {target_col_names[i]}: {acc[i]}")
            wandb_logger.log({f"Accuracy for target {target_col_names[i]}": acc[i]})
    accuracy = (predictions == test_targets).mean()
    print(f"Overall Accuracy: {accuracy}")
    wandb_logger.log({"Overall Accuracy": accuracy})

    if probabilities is None:
        return

    if num_classes > 2:
        AUROC = MulticlassAUROC(num_classes=num_classes, average=None)
        auroc = AUROC(tnsr(probabilities).float(), tnsr(test_targets))
        for i in range(num_classes):
            print(f"AUROC for target {target_col_names[i]}: {auroc[i]}")
            wandb_logger.log({f"AUROC for target {target_col_names[i]}": auroc[i]})
        wandb_logger.log({"mean AUROC": auroc.mean()})
        print(f"Mean AUROC: {auroc.mean()}")
    else:
        AUROC = BinaryAUROC()
        auroc = AUROC(tnsr(probabilities[:, 1]).float(), tnsr(test_targets))
        print(f"AUROC: {auroc}")
        wandb_logger.log({"AUROC": auroc})
    return
