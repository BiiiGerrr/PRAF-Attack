import os
import json
import hashlib
import random
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision
from PIL import Image
import hydra
from omegaconf import DictConfig, OmegaConf
from config_schema import MainConfig
from functools import partial
from typing import List, Dict, Optional
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from surrogates import (
    ClipB16FeatureExtractor,
    ClipB32FeatureExtractor,
    ClipLaionFeatureExtractor,
    EnsembleFeatureLoss,
    EnsembleFeatureExtractor,
    DINOv2FeatureExtractor,
    InternVL3_1B_FeatureExtractor,
)

from utils import hash_training_config, ensure_dir


BACKBONE_MAP: Dict[str, type] = {
    "B16": ClipB16FeatureExtractor,
    "B32": ClipB32FeatureExtractor,
    "Laion": ClipLaionFeatureExtractor,
    "DINO": DINOv2FeatureExtractor,
    "InternVL": InternVL3_1B_FeatureExtractor,
}



def calculate_layer_loss(adv_feat, tgt_feat, lambda_cls, lambda_spatial, keep_ratio):


    cls_sim = F.cosine_similarity(adv_feat[:, 0, :], tgt_feat[:, 0, :], dim=-1)
    l_cls = 1.0 - cls_sim.mean()

    adv_patch = adv_feat[:, 1:, :]
    tgt_patch = tgt_feat[:, 1:, :]

    patch_sims = F.cosine_similarity(adv_patch, tgt_patch, dim=-1)

    seq_len = patch_sims.shape[1]
    k = int(seq_len * keep_ratio)

    topk_sims, topk_indices = torch.topk(patch_sims, k, dim=1, largest=True)

    l_spatial = 1.0 - topk_sims.mean()

    total_loss = lambda_cls * l_cls + lambda_spatial * l_spatial

    return total_loss, {
        "l_cls": l_cls.item(),
        "l_spatial": l_spatial.item()
    }, topk_indices


def load_captions(caption_path):
    if not os.path.exists(caption_path):
        return []
    with open(caption_path, 'r') as f:
        captions = [line.strip() for line in f.readlines()]
    return captions


def get_models(cfg: MainConfig):
    if not cfg.model.ensemble and len(cfg.model.backbone) > 1:
        raise ValueError("When ensemble=False, only one backbone can be specified")

    models = []
    for backbone_name in cfg.model.backbone:
        if backbone_name not in BACKBONE_MAP:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        model_class = BACKBONE_MAP[backbone_name]
        model = model_class().eval().to(cfg.model.device).requires_grad_(False)
        models.append(model)
        print(f"  --> Loaded model: {backbone_name} ({model.__class__.__name__})")

    if cfg.model.ensemble:
        ensemble_extractor = EnsembleFeatureExtractor(models)
        print(f"Successfully created Ensemble with {len(models)} models.")
    else:
        ensemble_extractor = models[0]

    return ensemble_extractor, models


def get_ensemble_loss(cfg: MainConfig, models: List[nn.Module]):
    return EnsembleFeatureLoss(models)


def set_environment(seed=2023):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)


@hydra.main(version_base=None, config_path="config", config_name="ensemble_5models")
def main(cfg: MainConfig):
    set_environment()

    ensemble_extractor, models = get_models(cfg)
    ensemble_loss = get_ensemble_loss(cfg, models)

    transform_fn = transforms.Compose([
        transforms.Resize(cfg.model.input_res, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(cfg.model.input_res),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Lambda(lambda img: to_tensor(img)),
    ])


    clean_data = ImageFolderWithPaths(cfg.data.cle_data_path, transform=transform_fn)
    data_loader_cle = DataLoader(clean_data, batch_size=1, shuffle=False)

    print(f"Loading target data from: {cfg.data.tgt_data_path}")
    target_data = ImageFolderWithPaths(cfg.data.tgt_data_path, transform=transform_fn)
    data_loader_tgt = DataLoader(target_data, batch_size=1, shuffle=False)

    source_crop = (
        transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale)
        if cfg.model.use_source_crop else torch.nn.Identity()
    )
    target_crop = (
        transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale)
        if cfg.model.use_target_crop else torch.nn.Identity()
    )

    print(f"Clean samples: {len(clean_data)}, Target samples: {len(target_data)}")

    for i, ((image_cle, _, path_cle), (image_tgt, _, path_tgt)) in enumerate(zip(data_loader_cle, data_loader_tgt)):
        if i >= cfg.data.num_samples:
            break

        file_name = path_cle[0].split("/")[-1]
        tgt_name = path_tgt[0].split("/")[-1]
        print(f"\n[{i}] Source: {file_name} -> Target: {tgt_name}")


        attack_imgpair(
            cfg=cfg,
            ensemble_extractor=ensemble_extractor,
            ensemble_loss=ensemble_loss,
            source_crop=source_crop,
            target_crop=target_crop,
            img_index=i,
            image_org=image_cle,
            image_tgt=image_tgt,
            path_org=path_cle,
        )


def attack_imgpair(
        cfg: MainConfig,
        ensemble_extractor: nn.Module,
        ensemble_loss: nn.Module,
        source_crop: Optional[transforms.RandomResizedCrop],
        target_crop: Optional[transforms.RandomResizedCrop],
        img_index: int,
        image_org: torch.Tensor,
        path_org: List[str],
        image_tgt: torch.Tensor,
):
    image_org = image_org.to(cfg.model.device)
    image_tgt = image_tgt.to(cfg.model.device)

    attack_type = cfg.attack
    attack_fn = {"fgsm": fgsm_attack}[attack_type]

    adv_image = attack_fn(
        cfg=cfg,
        ensemble_extractor=ensemble_extractor,
        ensemble_loss=ensemble_loss,
        source_crop=source_crop,
        target_crop=target_crop,
        img_index=img_index,
        image_org=image_org,
        image_tgt=image_tgt,
    )

    # Save images
    for path_idx in range(len(path_org)):
        folder = path_org[path_idx].split("/")[-2]
        name = path_org[path_idx].split("/")[-1]

        folder_to_save = os.path.join(cfg.data.output, "img", folder)
        ensure_dir(folder_to_save)

        save_name = name.replace(".JPEG", ".png") if "JPEG" in name else name
        torchvision.utils.save_image(adv_image[path_idx], os.path.join(folder_to_save, save_name))


def log_metrics(pbar, metrics, img_index, epoch=None):
    pbar_metrics = {k: f"{v:.5f}" if "sim" in k else f"{v:.3f}" for k, v in metrics.items()}
    pbar.set_postfix(pbar_metrics)
    prefix = f"[img={img_index}]"
    if epoch is not None: prefix += f"[epoch={epoch}]"
    kv = " ".join([f"{k}={v:.6f}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in metrics.items()])
    print(f"{prefix} {kv}")


def fgsm_attack(
        cfg: MainConfig,
        ensemble_extractor: nn.Module,
        ensemble_loss: nn.Module,
        source_crop: Optional[transforms.RandomResizedCrop],
        target_crop: Optional[transforms.RandomResizedCrop],
        img_index: int,
        image_org: torch.Tensor,
        image_tgt: torch.Tensor,
):
    image_org = image_org.to(cfg.model.device)
    image_tgt = image_tgt.to(cfg.model.device)

    selection_mode = cfg.optim.get("selection_mode", "top135")
    loss_keep_ratio = cfg.optim.get("loss_keep_ratio", 0.6)


    if img_index == 0:
        print(f"\n" + "=" * 40)
        print(f"  [Hyper-Parameter Check]")
        model_names = [m.__class__.__name__ for m in ensemble_extractor.extractors] \
            if isinstance(ensemble_extractor, EnsembleFeatureExtractor) else [ensemble_extractor.__class__.__name__]
        print(f"  - Active Backbones: {model_names}")
        print(f"  - Selection Mode: {selection_mode}")
        # print(f"  - Reselect Interval: {reselect_interval} steps")
        print(f"  - Loss Calculation Ratio: {loss_keep_ratio}")
        print(f"  - Lambda CLS: {cfg.optim.lambda_cls}")
        print(f"  - Lambda Spa: {cfg.optim.lambda_spatial}")
        print(f"  - Lambda Feature: {cfg.optim.lambda_feature}")
        print(f"  - use_target_crop: {cfg.model.use_target_crop}")
        print("=" * 40 + "\n")


    ensemble_extractor.adaptive_register_hooks(
        image_org, image_tgt, mode=selection_mode
    )

    delta = torch.zeros_like(image_org, requires_grad=True)
    pbar = tqdm(range(cfg.optim.steps), desc=f"Attack progress", leave=False)


    with torch.no_grad():
        tgt_input = image_tgt
        ensemble_loss.set_ground_truth(tgt_input)
        _ = ensemble_extractor(tgt_input)

        tgt_mid_features = {}
        raw_feats = ensemble_extractor.get_intermediate_features()
        for mid, layers in raw_feats.items():
            tgt_mid_features[mid] = {lid: f.clone().detach() for lid, f in layers.items()}

    stage_res = [56] * 100 + [112] * 100 + [224] * 100

    for epoch in pbar:

        current_res = stage_res[epoch] if epoch < len(stage_res) else stage_res[-1]


        if epoch == 0 or (epoch < len(stage_res) and stage_res[epoch] != stage_res[epoch - 1]):
            with torch.no_grad():
                if current_res < 224:

                    low_res_tgt = F.interpolate(image_tgt, size=(current_res, current_res),mode='nearest')

                    active_tgt = F.interpolate(low_res_tgt, size=(224, 224), mode='nearest')
                else:
                    active_tgt = image_tgt


                ensemble_loss.set_ground_truth(active_tgt)
                _ = ensemble_extractor(active_tgt)

                cached_tgt_mid_features = {}
                raw_feats = ensemble_extractor.get_intermediate_features()
                for mid, layers in raw_feats.items():
                    cached_tgt_mid_features[mid] = {lid: f.clone().detach() for lid, f in layers.items()}

                print(f"  [Condition Update] Epoch {epoch}: Target features re-encoded at res {current_res}")


        tgt_mid_features = cached_tgt_mid_features

        adv_image = image_org + delta
        metrics = {"mean_delta": torch.mean(torch.abs(delta)).item()}


        adv_features_global = ensemble_extractor(adv_image)
        global_sim = ensemble_loss(adv_features_global)
        metrics["g_sim"] = global_sim.item()

        total_loss = 0
        if cfg.model.use_source_crop:
            adv_input_local = source_crop(adv_image)
            adv_features_local = ensemble_extractor(adv_input_local)
            local_sim = ensemble_loss(adv_features_local)
            metrics["l_sim"] = local_sim.item()
            total_loss += -local_sim

            if cfg.optim.lambda_feature > 0:
                adv_mid_local = ensemble_extractor.get_intermediate_features()
                l_feat = 0
                sum_cls = 0
                sum_spatial = 0
                count = 0

                for mid in adv_mid_local:
                    for lid in adv_mid_local[mid]:

                        loss_val, loss_components, selected_indices = calculate_layer_loss(
                            adv_mid_local[mid][lid],
                            tgt_mid_features[mid][lid],
                            cfg.optim.lambda_cls,
                            cfg.optim.lambda_spatial,
                            keep_ratio=loss_keep_ratio
                        )
                        l_feat += loss_val
                        sum_cls += loss_components["l_cls"]
                        sum_spatial += loss_components["l_spatial"]
                        count += 1

                if count > 0:
                    l_feat /= count
                    metrics["l_fea_loss"] = l_feat.item()
                    metrics["f_cls"] = sum_cls / count
                    metrics["f_spatial"] = sum_spatial / count
                    total_loss += cfg.optim.lambda_feature * l_feat
        else:
            total_loss += -global_sim
            if cfg.optim.lambda_feature > 0:
                adv_mid_global = ensemble_extractor.get_intermediate_features()
                l_feat = 0
                count = 0
                for mid in adv_mid_global:
                    for lid in adv_mid_global[mid]:
                        loss_val, loss_components, selected_indices = calculate_layer_loss(
                            adv_mid_global[mid][lid],
                            tgt_mid_features[mid][lid],
                            cfg.optim.lambda_cls,
                            cfg.optim.lambda_spatial,
                            keep_ratio=loss_keep_ratio
                        )
                        l_feat += loss_val
                        count += 1
                if count > 0:
                    l_feat /= count
                    metrics["g_fea_loss"] = l_feat.item()
                    total_loss += cfg.optim.lambda_feature * l_feat

        metrics["loss"] = total_loss.item()
        log_metrics(pbar, metrics, img_index, epoch)

        grad = torch.autograd.grad(total_loss, delta)[0]
        delta.data = torch.clamp(delta.data - cfg.optim.alpha * torch.sign(grad),
                                 -cfg.optim.epsilon, cfg.optim.epsilon)

    ensemble_extractor.remove_hooks()
    return torch.clamp((image_org + delta) / 255.0, 0.0, 1.0)


if __name__ == "__main__":
    main()