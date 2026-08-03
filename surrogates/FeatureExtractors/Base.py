import torch
from torch import nn, Tensor
from abc import abstractmethod
from typing import List, Any, Callable, Dict
import torch.nn.functional as F

class BaseFeatureExtractor(nn.Module):
    def __init__(self):
        self.hooks = []
        self.intermediate_features = {}
        self.num_layers = 0
        super(BaseFeatureExtractor, self).__init__()
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def get_text_features(self, text: List[str]) -> Tensor:
        pass

    def register_hooks(self):
        pass


    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.intermediate_features = {}

    def generate_hook_fn(self, layer_idx):
        def hook_fn(module, input, output):

            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output

            self.intermediate_features[layer_idx] = out

        return hook_fn

    def register_hooks(self):
        self.remove_hooks()
        if self.num_layers == 0:
            return

        ratios = [7 / 12, 9 / 12, 11 / 12]
        target_layers = [int(self.num_layers * r) for r in ratios]
        target_layers = sorted(list(set([l for l in target_layers if l < self.num_layers])))

        print(f"[{self.__class__.__name__}] Hooking layers: {target_layers} / {self.num_layers}")

        encoder_layers = self.model.vision_model.encoder.layers

        for layer_idx in target_layers:
            hook = encoder_layers[layer_idx].register_forward_hook(
                self.generate_hook_fn(layer_idx)
            )
            self.hooks.append(hook)

    def adaptive_register_hooks(self, image_org, image_tgt, mode="top3",):
        """

        """
        self.remove_hooks()
        if self.num_layers == 0:
            return

        if hasattr(self.model, "vision_model"):  # CLIP
            encoder_layers = self.model.vision_model.encoder.layers
        elif hasattr(self.model, "vision_tower"):  # InternVL
            encoder_layers = self.model.vision_tower.encoder.layer
        elif hasattr(self.model, "encoder"):  # DINOv2
            encoder_layers = self.model.encoder.layer
        else:
            raise AttributeError(f"Could not find encoder layers for {self.__class__.__name__}")

        # encoder_layers = self.model.vision_model.encoder.layers
        # encoder_layers = self.model.vision_tower.encoder.layer


        temp_features = {}
        temp_hooks = []

        def get_temp_hook(idx):
            return lambda m, inp, out: temp_features.update({idx: out[0] if isinstance(out, tuple) else out})

        for i in range(self.num_layers):
            temp_hooks.append(encoder_layers[i].register_forward_hook(get_temp_hook(i)))


        img_adv = image_org.clone().detach().requires_grad_(True)

        _ = self(img_adv)
        adv_feats = {k: v for k, v in temp_features.items()}
        temp_features.clear()

        with torch.no_grad():
            _ = self(image_tgt)
            tgt_feats = {k: v.clone() for k, v in temp_features.items()}

        final_layer_idx = self.num_layers - 1
        loss_global = 1.0 - F.cosine_similarity(
            adv_feats[final_layer_idx][:, 0, :],
            tgt_feats[final_layer_idx][:, 0, :],
            dim=-1
        ).mean()

        grad_global = torch.autograd.grad(loss_global, img_adv, retain_graph=True)[0].flatten()


        layer_alignments = {}
        if mode == "bottom3":
            start_layer = int(self.num_layers * 0.0)
        else:
            start_layer = int(self.num_layers * 0.25)

        for i in range(start_layer, final_layer_idx):
            adv_sp = torch.mean(adv_feats[i][:, 1:, :], dim=1)
            tgt_sp = torch.mean(tgt_feats[i][:, 1:, :], dim=1)
            loss_i = 1.0 - F.cosine_similarity(adv_sp, tgt_sp, dim=-1).mean()

            grad_i = torch.autograd.grad(loss_i, img_adv, retain_graph=True)[0].flatten()
            sim = F.cosine_similarity(grad_global, grad_i, dim=0).item()
            layer_alignments[i] = sim

        for h in temp_hooks:
            h.remove()
        temp_features.clear()

        sorted_layers = sorted(layer_alignments.items(), key=lambda x: x[1], reverse=True)
        ranked_layer_ids = [x[0] for x in sorted_layers]

        if mode == "first3":

            target_layers = list(range(min(3, self.num_layers)))

        elif mode == "top3":
            target_layers = ranked_layer_ids[:3]

        elif mode == "top135":
            indices = [0, 2, 4]
            target_layers = [ranked_layer_ids[i] for i in indices if i < len(ranked_layer_ids)]

        else:
            raise ValueError(f"Unknown selection mode: {mode}")


        target_layers = sorted(target_layers)

        print(f"\n[{self.__class__.__name__}] Selection Mode: {mode}")
        print(f"[{self.__class__.__name__}] Selected Layers: {target_layers}")

        alignments_str = ", ".join([
            f"{layer_alignments[l]:.3f}" if l in layer_alignments else "N/A"
            for l in target_layers
        ])
        print(f"[{self.__class__.__name__}] Alignments: [{alignments_str}]\n")

        for layer_idx in target_layers:
            hook = encoder_layers[layer_idx].register_forward_hook(self.generate_hook_fn(layer_idx))
            self.hooks.append(hook)

class EnsembleFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureExtractor, self).__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor) -> Dict[int, Tensor]:
        features = {}
        for i, model in enumerate(self.extractors):
            features[i] = model(x).squeeze()
        return features

    def get_text_features(self, text: List[str]) -> Dict[int, Tensor]:
        features = {}
        for i, model in enumerate(self.extractors):
            features[i] = model.get_text_features(text).squeeze()
        return features

    def register_hooks(self):
        for model in self.extractors:
            model.register_hooks()

    def remove_hooks(self):
        for model in self.extractors:
            model.remove_hooks()

    def get_intermediate_features(self) -> Dict[int, Dict[int, Tensor]]:

        features = {}
        for i, model in enumerate(self.extractors):
            features[i] = model.intermediate_features
        return features


    def adaptive_register_hooks(self, image_org, image_tgt, mode="top3"):
        for model in self.extractors:
            model.adaptive_register_hooks(image_org, image_tgt, mode=mode)

class EnsembleFeatureLoss(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        """
        """
        self.ground_truth.clear()
        for model in self.extractors:

            self.ground_truth.append(model(x).to(x.device))

    def set_manual_ground_truth(self, features_list: List[torch.Tensor]):
        """

        """
        self.ground_truth = [f.clone().detach() for f in features_list]

    def __call__(self, feature_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss = 0
        for index, model in enumerate(self.extractors):
            gt_features = self.ground_truth[index]


            adv_feature = feature_dict[index]

            if adv_feature.dim() == 1:
                adv_feature = adv_feature.unsqueeze(0)

            similarities = torch.sum(adv_feature * gt_features, dim=1)

            avg_similarity = torch.mean(similarities)

            loss += avg_similarity

        loss = loss / len(self.extractors)

        return loss


