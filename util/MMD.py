import torch.nn as nn
import torch


## 参考自 https://github.com/GabrieleGhisleni/Unsupervised-domain-adaptation/blob/master/main.ipynb

def mmd_adapt(src_encoder, src_data, tgt_data, device, optimizer, epochs, scheduler, classifier, kernel="multiscale",
              mmd_lambda=0.25):
    for epoch in epochs:
        print(f'Starting epoch {epoch}')
        classifier.train()
        src_encoder.train()
        total_classifier_loss, mmd_total_loss, n_samples = 0, 0, 0
        data_zip = enumerate(zip(src_data, tgt_data))
        for step, ((source_images, labels), (target_images,)) in data_zip:
            optimizer.zero_grad()
            source_images.to(device, dtype=torch.float)
            target_images.to(device, dtype=torch.float)
            labels.to(device, dtype=torch.long)

            # mmd step
            shrinked_feature_map_source = src_encoder(source_images)
            shrinked_feature_map_target = src_encoder(target_images)

            mmd_loss = maximum_mean_discrepancies(
                shrinked_feature_map_source,
                shrinked_feature_map_target,
                kernel
            )

            mmd_loss_adjusted = (mmd_lambda * mmd_loss)

            logits = classifier(shrinked_feature_map_source)
            classification_loss = nn.CrossEntropyLoss(logits, labels)

            loss = classification_loss + mmd_loss_adjusted
            loss.backward()
            optimizer.step()

            total_classifier_loss += classification_loss.item()
            mmd_total_loss += mmd_loss.item()
            n_samples += source_images.size()[0]
            classifier_loss = total_classifier_loss / n_samples

            print(f'classifier_loss:{classifier_loss}')
            print(f'mmd_loss:{mmd_total_loss / n_samples}')
            print(f'total_loss:{(total_classifier_loss + mmd_total_loss) / n_samples}')
        scheduler.step()


def maximum_mean_discrepancies(x, y, device, kernel="multiscale"):
    """
        # https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy/notebook
        Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.
        """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)
