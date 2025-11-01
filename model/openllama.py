import torch.nn.functional as F
from .ImageBind import *
from .ImageBind import data
from transformers import LlamaForCausalLM, AutoTokenizer
from .AnomalyGPT_models import *
from transformers import StoppingCriteria, StoppingCriteriaList,LlamaTokenizer
import kornia as K
from peft import LoraConfig,TaskType,get_peft_model
import torch
from torch.nn.utils import rnn
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel
from model.freq_branch import DCTBranch, GatedFuse
from model.ImageBind.models import imagebind_model
from model.ImageBind.models.imagebind_model import ModalityType
import torch.nn.functional as F
# --- robust ModalityType alias ---
try:
    # 常见路径
    from .ImageBind.data import ModalityType   # noqa: F401
except Exception:
    try:
        # 有些分支把它放到 imagebind_model 里
        from .imagebind_model import ModalityType  # noqa: F401
    except Exception:
        # 兜底占位（大多数实现里用到的 key 名就是 "vision"）
        class ModalityType:
            VISION = "vision"


CLASS_NAMES=['face','object']
normal_sentences_all = []
abnormal_sentences_all = []
normal_sentences_all.append(data.load_and_transform_text('real', torch.cuda.current_device()))
abnormal_sentences_all.append(data.load_and_transform_text('fake', torch.cuda.current_device()))

def encode_text_with_prompt_ensemble(model, obj, device, ctx):
    global normal_sentences_all,abnormal_sentences_all
    normal_sentences = torch.cat(normal_sentences_all).to(device)
    abnormal_sentences = torch.cat(abnormal_sentences_all).to(device)
    with torch.no_grad():
        class_embeddings_normal = model({ModalityType.TEXT: normal_sentences})[ModalityType.TEXT][0]
        class_embeddings_abnormal = model({ModalityType.TEXT: abnormal_sentences})[ModalityType.TEXT][0]

    class_embeddings_normal=torch.cat((ctx.cuda(),class_embeddings_normal),dim=0)
    class_embeddings_abnormal=torch.cat((ctx.cuda(),class_embeddings_abnormal),dim=0)

    class_embeddings_normal = class_embeddings_normal.unsqueeze(0)
    class_embeddings_normal = class_embeddings_normal.mean(dim=1, keepdim=True)
    class_embeddings_normal = class_embeddings_normal / class_embeddings_normal.norm(dim=-1, keepdim=True)

    class_embeddings_abnormal = class_embeddings_abnormal.unsqueeze(0)
    class_embeddings_abnormal = class_embeddings_abnormal.mean(dim=1, keepdim=True)
    class_embeddings_abnormal = class_embeddings_abnormal / class_embeddings_abnormal.norm(dim=-1, keepdim=True)
    
    text_features = torch.cat([class_embeddings_normal, class_embeddings_abnormal], dim=1)

    return text_features

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
        if stop_count >= self.ENCOUNTERS:
            return True
        return False

def build_one_instance(tokenizer, conversation):
    text_list = []
    turn_num = len(conversation)
    input_ids, target_ids = [], []
    for i in range(turn_num):
        turn = conversation[i]
        role = turn['from']
        if i == 0: # the first human turn
            assert role == 'human'
            text = turn['value'] + '\n### Assistant:'
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100]*len(one_input_id) # do not perform loss regression on human prompt
        else:
            if role == 'human':
                text = 'Human: ' + turn['value'] + '\n### Assistant:'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100]*len(one_input_id)
            elif role == 'gpt':
                text = turn['value'] + '\n###'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            else:
                raise Exception('Wrong Role!!!')
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids

def process_batch_instance(tokenizer, batch_of_conversations, max_tgt_len):
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        _, one_input_ids, one_target_ids = build_one_instance(tokenizer, conversation)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))

    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:,:max_tgt_len]
    target_ids = target_ids[:,:max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()



PROMPT_START = '### Human: <Img>'
class OpenLLAMAPEFTModel(nn.Module):

    '''LoRA for LLaMa model'''

    def __init__(self, imagesize=224, **args):
        super(OpenLLAMAPEFTModel, self).__init__()
        # -------- 基础超参（带默认值，防 KeyError）--------
        self.args = dict(args)
        self.imagesize    = int(imagesize)
        self.max_tgt_len  = int(self.args.get('max_tgt_len', 128))
        self.device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # -------- 视觉主干（ImageBind huge）--------
        # 注意：如果 imagebind_huge 需要额外键，可在 self.args 补齐默认值
        self.visual_encoder, self.visual_hidden_size = imagebind_model.imagebind_huge(self.args)
        self.visual_encoder = self.visual_encoder.bfloat16()  # 主干用 bf16
        self.dct_branch = DCTBranch(k_keep=16, hidden=64, out_dim=int(self.visual_hidden_size))
        self.fuse_gate  = GatedFuse(dim=int(self.visual_hidden_size))
        # -------- 频域分支 + 门控（与视觉向量同维）--------
        # 原来：


        D = int(self.visual_hidden_size)  # 例如 1280（和 ImageBind huge 一致）
        self.dct_branch = DCTBranch(k_keep=16, hidden=64, out_dim=D).bfloat16()
        self.fuse_gate  = GatedFuse(dim=D).bfloat16()

        # -------- 其余分支（保持你原有模块）--------
        self.iter        = 0
        self.locator     = ConvLayer(2, 128, 3).bfloat16()
        self.classifier  = Classifier(2, 3).bfloat16()
        self.image_decoder = LinearLayer(1280, 1024, 3).bfloat16()  # 你的下游头假定视觉向量 1280 维
        self.prompt_learner = ForgeryPromptLearner(6, 4096).bfloat16()

        ctx_vectors = torch.empty(16, 1024)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors).bfloat16()

        # -------- LLM（Vicuna，本地加载）--------
        enable_llm = bool(self.args.get('enable_llm', True))
        llama_path = self.args.get('llama_path', "/root/autodl-tmp/models/vicuna-7b-v1.5-16k")

        if enable_llm:
            self.llama_tokenizer = LlamaTokenizer.from_pretrained(
                llama_path, use_fast=False, local_files_only=True
            )
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            self.llama_tokenizer.padding_side = "right"

            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_path, local_files_only=True
            )

            # ---- LoRA（可选，r=0 表示关闭）----
            lora_r        = int(self.args.get('lora_r', 0))
            lora_alpha    = int(self.args.get('lora_alpha', 16))
            lora_dropout  = float(self.args.get('lora_dropout', 0.05))
            lora_targets  = self.args.get('lora_target_modules',
                                        ['q_proj','k_proj','v_proj','o_proj'])
            lora_bias     = self.args.get('lora_bias', 'none')

            if lora_r > 0:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=True,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=lora_targets,
                    bias=lora_bias,
                )
                self.llama_model = get_peft_model(self.llama_model, peft_config)

            self.llama_model = self.llama_model.bfloat16()
            print('Language decoder initialized.')

            # 视觉 → LLM 投影，维度与视觉向量一致
            self.llama_proj = nn.Linear(self.visual_hidden_size, 4096).bfloat16()
        else:
            # 不启用 LLM 的占位（方便纯视觉/探针训练）
            self.llama_tokenizer = None
            self.llama_model     = None
            self.llama_proj      = nn.Identity()

        # 其余需要的属性
        # self.some_flag = self.args.get('some_flag', default_value)


    def encode_image(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].bfloat16() for key in inputs}
        # with torch.no_grad():
        embeddings = self.visual_encoder(inputs)
        image_embeds = embeddings['vision'][0] # bsz x 1024
        patch_features = embeddings['vision'][1] # bsz x h*w x 1280
        patch_tokens = self.image_decoder(patch_features) # bsz x h*w x 1024

        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama, patch_tokens
    
    def encode_image_for_web_demo(self, image_paths,img_tensor=None):
        if img_tensor is not None:
            inputs = {ModalityType.VISION: img_tensor.to(self.device)}
        else:
            inputs = {ModalityType.VISION: data.load_and_transform_vision_data_for_web_demo(image_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].bfloat16() for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0] # bsz x 1024
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
        patch_tokens = self.image_decoder(patch_features) # bsz x h*w x 1024

        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama, patch_tokens

    

    # ---- 替换 encode_image_from_tensor ----
    # ====== openllama.py: 替换你的 encode_image_from_tensor ======
    def encode_image_from_tensor(self, image_tensors):
        """
        输入:
            image_tensors: Tensor，形状 (B,3,H,W) 或 (3,H,W)，数值范围 [0,1]
        返回:
            inputs_llama:  若 enable_llm=True 则为 (B,1,4096)；否则 None
            atts_llama:    若 enable_llm=True 则为 (B,1) 的 attention mask；否则 None
            patch_tokens:  (B,L,1280) 的 float32 张量（已统一为 (B,L,D)）
        """
        import torch

        # ---- 1) 组装 batch & 设备/类型对齐 ----
        x = image_tensors
        if x.dim() == 3:
            x = x.unsqueeze(0)                               # (1,3,H,W)
        x = x.contiguous().to(self.device)

        # 跟随视觉编码器参数 dtype（ImageBind 多为 bfloat16）
        v_dtype = next(self.visual_encoder.parameters()).dtype
        x = x.to(v_dtype)                                    # 避免 conv 输入/权重 dtype 不一致

        # 尝试使用 ModalityType 作为 key；失败则退回 "vision"
        try:
            from .ImageBind.data import ModalityType
            inputs = {ModalityType.VISION: x}
            vision_key = ModalityType.VISION
        except Exception:
            inputs = {"vision": x}
            vision_key = "vision"

        # ---- 2) 前向，拿到全局向量与 patch 特征 ----
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)

        if vision_key not in embeddings:
            raise KeyError(f"'vision' output not found in embeddings keys: {list(embeddings.keys())}")

        vision_out = embeddings[vision_key]                  # 预期为 (global_embed, patch_feats)
        # 兼容 tuple/list 结构
        if isinstance(vision_out, (list, tuple)) and len(vision_out) >= 2:
            global_embed, patch_feats = vision_out[0], vision_out[1]
        else:
            raise RuntimeError("Unexpected vision output structure from visual_encoder.")

        # 某些实现会给各层 tokens 列表，这里取最后一层
        if isinstance(patch_feats, (list, tuple)):
            patch_feats = patch_feats[-1]

        if patch_feats.dim() != 3:
            raise RuntimeError(f"expect 3D patch feats, got {tuple(patch_feats.shape)}")

        # ---- 3) 统一 patch feats 为 (B,L,D) ，并转 float32 ----
        # 常见 L=196/197/256/257；若当前是 (L,B,D) 且第 1 维是小 batch，则转置到 (B,L,D)
        s0, s1, s2 = patch_feats.shape
        if s0 in (196, 197, 256, 257) and s1 <= max(64, x.shape[0] + 2):
            patch_tokens = patch_feats.permute(1, 0, 2).contiguous().float()   # (B,L,D)
        else:
            patch_tokens = patch_feats.contiguous().float()                    # (B,L,D)

        # 全局向量也转为 float32 以便后续使用/投影
        image_embeds = global_embed.contiguous().float()                       # (B,1280)

        # ---- 4) 构造 LLaMA 输入（若启用） ----
        inputs_llama, atts_llama = None, None
        if getattr(self, "enable_llm", False):
            # self.llama_proj: (1280 -> 4096)
            # 跟随投影权重 dtype 再做转换
            proj_dtype = self.llama_proj.weight.dtype if hasattr(self.llama_proj, "weight") else torch.float32
            v = image_embeds.to(proj_dtype)
            inputs_llama = self.llama_proj(v).unsqueeze(1)                     # (B,1,4096)
            # attention mask 用 long
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long, device=self.device)

        return inputs_llama, atts_llama, patch_tokens  # patch_tokens: (B,L,1280) float32



# ---- 替换 encode_image_feats_from_tensor ----
# ====== openllama.py: 替换你的 encode_image_feats_from_tensor ======
    @torch.no_grad()
    def encode_image_feats_from_tensor(self, image_tensors, pool: str = "mean_nocls"):
        """
        将 patch tokens 聚合为图像级向量:
            pool = "mean"        -> 对所有 token 取均值
            pool = "cls"         -> 取第 0 个 token 作为 CLS
            pool = "mean_nocls"  -> 跳过第 0 个 token，再对其余取均值（推荐）
        返回:
            f_rgb: (B,1280) float32
        """
        _, _, patch_tokens = self.encode_image_from_tensor(image_tensors)      # (B,L,1280) float32
        if patch_tokens.dim() != 3:
            raise RuntimeError(f"expect (B,L,D) tokens, got {tuple(patch_tokens.shape)}")

        if pool == "mean":
            f_rgb = patch_tokens.mean(dim=1)                                   # (B,1280)
        elif pool == "cls":
            f_rgb = patch_tokens[:, 0, :]                                      # (B,1280)
        elif pool == "mean_nocls":
            if patch_tokens.size(1) <= 1:
                f_rgb = patch_tokens.mean(dim=1)
            else:
                f_rgb = patch_tokens[:, 1:, :].mean(dim=1)                     # (B,1280)
        else:
            raise ValueError(f"Unknown pool mode: {pool}")

        return f_rgb.contiguous().float()


    def prompt_wrap(self, img_embeds, input_ids, target_ids, attention_mask, anomaly_embedding = None, prediction=None):
        '''
            input_ids, target_ids, attention_mask: bsz x s2
        '''
        input_ids = input_ids.to(self.device) # bsz x s2
        target_ids = target_ids.to(self.device) # bsz x s2
        attention_mask = attention_mask.to(self.device) # bsz x s2

        batch_size = img_embeds.shape[0]
        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(p_before, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        if prediction.shape[0]==2:
            p_middle1 = '</Img> According to KFD prediction, the forgery score is '+str(int(prediction[0,1].item()*100))[:5]+'%. '
            p_middle2 = '</Img> According to KFD prediction, the forgery score is '+str(int(prediction[1,1].item()*100))[:5]+'%. '
            p_middle_tokens1 = self.llama_tokenizer(p_middle1, 
                return_tensors="pt", add_special_tokens=False).to(self.device)
            # peft model need deeper call
            p_middle_embeds1 = self.llama_model.model.model.embed_tokens(p_middle_tokens1.input_ids) # bsz x s1 x embed_dim
            p_middle_tokens2 = self.llama_tokenizer(p_middle2, 
                return_tensors="pt", add_special_tokens=False).to(self.device)
            # peft model need deeper call
            p_middle_embeds2 = self.llama_model.model.model.embed_tokens(p_middle_tokens2.input_ids) # bsz x s1 x embed_dim
            min_len=min(p_middle_embeds1.shape[1],p_middle_embeds2.shape[1])
            p_middle_embeds=torch.cat((p_middle_embeds1[:,:min_len,:],p_middle_embeds2[:,:min_len,:]),dim=0)
        if prediction.shape[0]==1:
            p_middle1 = '</Img> According to deepfake prediction, the forgery score is '+str(int(prediction[0,1].item()*100))[:5]+'%. '
            p_middle_tokens1 = self.llama_tokenizer(p_middle1, 
                return_tensors="pt", add_special_tokens=False).to(self.device)
            # peft model need deeper call
            p_middle_embeds = self.llama_model.model.model.embed_tokens(p_middle_tokens1.input_ids) # bsz x s1 x embed_dim

        p_after_embeds = self.llama_model.model.model.embed_tokens(input_ids).expand(batch_size, -1, -1) # bsz x s2 x embed_dim
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim

        inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_middle_embeds, anomaly_embedding, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
        if prediction != None:
            prediction=prediction.unsqueeze(2).repeat(1,1,4096)
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_middle_embeds, anomaly_embedding, prediction, p_after_embeds], dim=1)
        # create targets
        empty_targets = (
            torch.ones([batch_size, 3+p_before_embeds.size()[1]+1+p_middle_embeds.size()[1] + anomaly_embedding.size()[1]], dtype=torch.long).to(self.device).fill_(-100)  
        ) # bsz x (1 + s1 + 1)
        targets = torch.cat([empty_targets, target_ids], dim=1) # bsz x (1 + s1 + 1 + s2)
        assert inputs_embeds.size()[1] == targets.size()[1]

        atts_prefix = torch.ones([batch_size, 3+p_before_embeds.size()[1]+1+p_middle_embeds.size()[1] + anomaly_embedding.size()[1]], dtype=torch.long).to(self.device) # bsz x (1 + s1 +1)
        attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)
        assert attention_mask.size() == targets.size() # bsz x (1 + s1 + 1 + s2)
        return inputs_embeds, targets, attention_mask 



    def forward(self, inputs):
        image_paths = inputs['images']

        # ========== 取得主干特征 ==========
        img_embeds, _, patch_tokens = self.encode_image_from_tensor(image_paths)

        # ========== 频域分支 + 门控融合（新增）==========
        # 假设 inputs['images'] 已是 [0,1] 的张量；若是路径或已做过mean/std标准化，请看方案 B
        x_rgb_01 = inputs['images']                    # (B,3,H,W) in [0,1]
        last_tokens = patch_tokens[-1] if isinstance(patch_tokens, (list, tuple)) else patch_tokens  # (B,L,1024)
        f_rgb_vec   = last_tokens.mean(dim=1) if last_tokens.dim() == 3 else last_tokens            # (B,1024)
        with torch.cuda.amp.autocast(enabled=False):   # 频域与门控用 fp32 更稳
            f_freq = self.dct_branch(x_rgb_01.float()) # (B, D)
            f_rgb  = f_rgb_vec.float()  
            fused_vis = self.fuse_gate(f_rgb, f_freq).to(f_rgb_vec.dtype)           
            img_embeds = self.fuse_gate(f_rgb, f_freq).to(img_embeds.dtype)  # (B, D)
        # =============================================

        class_name = inputs['class_names']
        feats_text_tensor = encode_text_with_prompt_ensemble(
            self.visual_encoder, class_name, self.device, self.ctx
        )

        anomaly_maps = []
        for layer in range(len(patch_tokens)):
            patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(dim=-1, keepdim=True)
            anomaly_map = (100.0 * patch_tokens[layer].bfloat16() @ feats_text_tensor.transpose(-2, -1))
            B, L, C = anomaly_map.shape
            H = int(np.sqrt(L))
            anomaly_map = F.interpolate(
                anomaly_map.permute(0, 2, 1).view(B, 2, H, H).float(),
                size=self.imagesize, mode='bilinear', align_corners=True
            )
            anomaly_maps.append(anomaly_map)

        anomaly_map_all = self.locator(anomaly_maps)
        prediction = self.classifier(anomaly_maps)
        label = torch.stack(inputs['labels'], dim=0).to(self.device)
        gt = torch.stack(inputs['masks'], dim=0).to(self.device)

        f_loss = self.loss_focal(anomaly_map_all, gt)
        d_loss = self.loss_dice(anomaly_map_all[:, 1:, :, :], gt)
        loss_pixel = d_loss + f_loss
        anomaly_map_all = anomaly_map_all[:, 1:, :, :]

        clsloss = nn.BCELoss()(prediction, label.bfloat16())
        out = torch.argmax(prediction.data, 1)
        label1 = torch.argmax(label, dim=1).cuda()
        cls_acc = torch.sum(out == label1).item() / len(out)

        anomaly_maps_cat = torch.cat(anomaly_maps, dim=1)
        anomaly_map_prompts = self.prompt_learner(anomaly_maps_cat, anomaly_map_all, prediction)

        output_texts = inputs['texts']
        input_ids, target_ids, attention_mask = process_batch_instance(
            self.llama_tokenizer, output_texts, self.max_tgt_len
        )
        inputs_embeds, targets, attention_mask = self.prompt_wrap(
            img_embeds, input_ids, target_ids, attention_mask, anomaly_map_prompts, prediction
        )
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds.bfloat16(),
            attention_mask=attention_mask.bfloat16(),
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()

        return anomaly_map_all

        

    def extract_multimodal_feature(self, inputs, img_tensor=None):
        features = []    
        prompt = inputs['prompt']
        c_name = 'face'
        for name in CLASS_NAMES:
            if name in prompt:
                c_name = name
                break
        image_embeds, _, patch_tokens = self.encode_image_for_web_demo(inputs['image_paths'],img_tensor)
        feats_text_tensor = encode_text_with_prompt_ensemble(self.visual_encoder, [c_name], self.device,self.ctx)
        
        anomaly_maps = []
        if img_tensor is None:
            image_paths=data.load_and_transform_vision_data(inputs['image_paths'], self.device)
            img_tensor=torch.cat([i.unsqueeze(0) for i in image_paths],dim=0)
        for layer in range(len(patch_tokens)):
            patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(dim=-1, keepdim=True)
            anomaly_map = (100.0 * patch_tokens[layer] @ feats_text_tensor.transpose(-2,-1))
            B, L, C = anomaly_map.shape
            H = int(np.sqrt(L))
            anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H).float(),
                                        size=self.imagesize, mode='bilinear', align_corners=True)
            anomaly_maps.append(anomaly_map.bfloat16())
        anomaly_map_all = self.locator(anomaly_maps)
        features.append(image_embeds)
    

        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return feature_embeds, anomaly_map_all,anomaly_maps

    def prepare_generation_embedding(self, inputs, img_tensor=None):
        prompt = inputs['prompt']
        feature_embeds, anomaly_map,maps = self.extract_multimodal_feature(inputs, img_tensor=img_tensor)

        prediction=self.classifier(maps)
        pred=prediction
        inputs['modality_embeds'].append(feature_embeds)

        batch_size = feature_embeds.shape[0]
        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(p_before, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        
        if prediction != None:
            if prediction.shape[0]==2:
                p_middle1 = '</Img> According to KFD prediction, the forgery score is '+str(int(prediction[0,1].item()*100))[:5]+'%. '
                p_middle2 = '</Img> According to KFD prediction, the forgery score is '+str(int(prediction[1,1].item()*100))[:5]+'%. '
                p_middle_tokens1 = self.llama_tokenizer(p_middle1, 
                    return_tensors="pt", add_special_tokens=False).to(self.device)
                p_middle_embeds1 = self.llama_model.model.model.embed_tokens(p_middle_tokens1.input_ids) # bsz x s1 x embed_dim
                p_middle_tokens2 = self.llama_tokenizer(p_middle2, 
                    return_tensors="pt", add_special_tokens=False).to(self.device)
                p_middle_embeds2 = self.llama_model.model.model.embed_tokens(p_middle_tokens2.input_ids) # bsz x s1 x embed_dim
                min_len=min(p_middle_embeds1.shape[1],p_middle_embeds2.shape[1])
                p_middle_embeds=torch.cat((p_middle_embeds1[:,:min_len,:],p_middle_embeds2[:,:min_len,:]),dim=0)
            if prediction.shape[0]==1:
                p_middle1 = '</Img> According to deepfake prediction, the forgery score is '+str(int(prediction[0,1].item()*100))[:5]+'%. '
                p_middle_tokens1 = self.llama_tokenizer(p_middle1, 
                    return_tensors="pt", add_special_tokens=False).to(self.device)
                p_middle_embeds = self.llama_model.model.model.embed_tokens(p_middle_tokens1.input_ids) # bsz x s1 x embed_dim
        else:
            p_middle = '</Img> '
            p_middle_tokens = self.llama_tokenizer(p_middle, 
                return_tensors="pt", add_special_tokens=False).to(self.device)
            p_middle_embeds = self.llama_model.model.model.embed_tokens(p_middle_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        anomaly_map_cat=torch.cat(maps,dim=1)
        anomaly_map_prompts = self.prompt_learner(anomaly_map_cat,anomaly_map[:,1:,:,:],prediction)


        text = prompt + '\n### Assistant:'
        p_after_tokens = self.llama_tokenizer(text, add_special_tokens=False, return_tensors='pt').to(self.device)
        p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s2 x embed_dim
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim
        prediction=prediction.unsqueeze(1).repeat(1,1,2048)
        if img_tensor is not None:
            inputs_embeds=bos_embeds
        else:
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, feature_embeds, p_middle_embeds, anomaly_map_prompts, prediction, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
        anomaly_map=F.interpolate(anomaly_map.float(),(self.imagesize,self.imagesize))

        return inputs_embeds, anomaly_map[:,1:,:,:], pred

    def generate(self, inputs, img_tensor=None):

        '''
            inputs = {
                'image_paths': optional,
                'audio_paths': optional
                'video_paths': optional
                'thermal_paths': optional
                'mode': generation mode,
                'prompt': human input prompt,
                'max_tgt_len': generation length,
                'top_p': top_p,
                'temperature': temperature
                'modality_embeds': None or torch.tensor
                'modality_cache': save the image cache
            }
        '''

        input_embeds, pixel_output, prob = self.prepare_generation_embedding(inputs, img_tensor=img_tensor)
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2277], encounters=1)])
        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds.bfloat16(),
            max_new_tokens=inputs['max_tgt_len'],
            top_p=inputs['top_p'],
            temperature=inputs['temperature'],
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            attention_mask=torch.ones((input_embeds.shape[0],input_embeds.shape[1])).cuda(),
            pad_token_id=self.llama_tokenizer.eos_token_id
        )
        output_text = self.llama_tokenizer.decode(outputs[0][:-2], skip_special_tokens=True)
        return output_text, pixel_output, prob

# ====== openllama.py: 放在类外的工具函数（或放到类里也行，关键是可被调用）======
def _to_BLD(x, prefer_last: bool = True):
    """
    统一把各种结构（tensor / list / tuple / dict）转成 (B, L, D) 的 float32：
      - Tensor( B, L, D ) 直接返回
      - Tensor( B, D, H, W ) => (B, H*W, D)
      - list/tuple: 取最后一个（或第一个），再递归
      - dict: 优先常见 key，否则取最后一个 value 再递归
    """
    import torch
    if torch.is_tensor(x):
        if x.dim() == 3:  # (B, L, D)
            return x.float().contiguous()
        if x.dim() == 4:  # (B, D, H, W) -> (B, H*W, D)
            B, D, H, W = x.shape
            return x.permute(0, 2, 3, 1).reshape(B, H * W, D).float().contiguous()
        raise RuntimeError(f"Unexpected tensor rank for patch_features: {tuple(x.shape)}")

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise RuntimeError("Empty list/tuple for patch_features.")
        return _to_BLD(x[-1] if prefer_last else x[0], prefer_last=prefer_last)

    if isinstance(x, dict):
        for k in ["patch_tokens", "tokens", "patch", "features", "feat", "patches"]:
            if k in x:
                return _to_BLD(x[k], prefer_last=prefer_last)
        if len(x) == 0:
            raise RuntimeError("Empty dict for patch_features.")
        return _to_BLD(list(x.values())[-1] if prefer_last else list(x.values())[0],
                       prefer_last=prefer_last)

    raise RuntimeError("Cannot infer (B,L,D) from structure; unsupported type.")
