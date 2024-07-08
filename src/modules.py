from collections import defaultdict
from functools import partial

import torch

from src.approach import ModelModule
from src.modeling_big_bird_decompx import BigBirdForSequenceClassification as BFSC_decompx
from src.modeling_big_bird_decompx import CONFIGS


class ModelDecompxModule(ModelModule):
    def __init__(self, args):
        super().__init__(args)

        # Load embedding model
        self.model = BFSC_decompx.from_pretrained(
            args.model_name,
            num_labels=2,
            output_attentions=True,
            output_hidden_states=True,
        )

    def forward(self, inputs_embeds, attention_mask, token_type_ids):

        decompx_config = CONFIGS["DecompX"]

        # Encode input
        embedding_model_output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
            decompx_config=decompx_config,
        )

        if decompx_config.output_all_layers:
            logits, hidden_states, attentions, decompx_last_layer_outputs, decompx_all_layers_outputs = embedding_model_output
        else:
            logits, hidden_states, attentions, decompx_last_layer_outputs = embedding_model_output

        return {
            "pred_logits": logits,
            "hidden_states": hidden_states,
            "attentions": attentions,
            "decompx_last_layer": decompx_last_layer_outputs,
            "decompx_all_layers": decompx_all_layers_outputs if decompx_config.output_all_layers else None,
        }

    def compute_attributions(self):
        raise NotImplementedError


#COPIED/ADAPTED FROM: https://github.com/mt-upc/transformer-contributions/blob/main/src/contributions.py
class ModelAltiModule(ModelModule):
    def __init__(self, args):
        super().__init__(args)

        self.config = {"models":
            {"bert":
                {
                    "layer": 'bert.encoder.layer',
                    "ln1": 'attention.output.LayerNorm',
                    "ln2": 'output.LayerNorm',
                    "values": 'attention.self.value',
                    "dense": 'attention.output.dense',
                    "pre_layer_norm": 'False',
                },
            }
        }

        self.modules_config = self.config['models']['bert']

        self.num_attention_heads = self.model.config.num_attention_heads
        self.attention_head_size = int(self.model.config.hidden_size / self.model.config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

    def save_activation(self, name, mod, inp, out):
        if self.modules_config['ln1'] in name or self.modules_config['ln2'] in name:
            self.func_inputs[name].append(inp)
            self.func_outputs[name].append(out)
        if self.modules_config['values'] in name:
            self.func_outputs[name].append(out)

    def clean_hooks(self):
        for k, v in self.handles.items():
            self.handles[k].remove()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def get_model_info(self):
        ln1_name = self.modules_config['ln1']
        ln2_name = self.modules_config['ln2']
        values_name = self.modules_config['values']
        model_layer_name = self.modules_config['layer']
        dense_name = self.modules_config['dense']
        pre_layer_norm = self.modules_config['pre_layer_norm']
        lnf_name = self.modules_config['final_layer_norm']


        return {'ln1_name': ln1_name,
                'ln2_name': ln2_name,
                'values_name': values_name,
                'model_layer_name': model_layer_name,
                'dense_name': dense_name,
                'pre_layer_norm': pre_layer_norm,
                'lnf_name': lnf_name}

    def get_modules_model(self, layer):
        model_info_dict = self.get_model_info()  
        model_layer_name = model_info_dict['model_layer_name']

        dense = self.get_module(self.model, model_info_dict['dense_name'], layer, model_layer_name)
        fc1 = self.get_module(self.model, model_info_dict['fc1_name'], layer, model_layer_name)
        fc2 = self.get_module(self.model, model_info_dict['fc2_name'], layer, model_layer_name)
        ln1 = self.get_module(self.model, model_info_dict['ln1_name'], layer, model_layer_name)
        ln2 = self.get_module(self.model, model_info_dict['ln2_name'], layer, model_layer_name)

        return {'dense': dense,
                'fc1': fc1,
                'fc2': fc2,
                'ln1': ln1,
                'ln2': ln2}

    @torch.no_grad()
    def get_contributions(self, hidden_states_model, attentions, func_inputs, func_outputs):
        #   hidden_states_model: Representations from previous layer and inputs to self-attention. (batch, seq_length, all_head_size)
        #   attentions: Attention weights calculated in self-attention. (batch, num_heads, seq_length, seq_length)

        ln1_name = self.modules_config['ln1']
        ln2_name = self.modules_config['ln2']
        values_name = self.modules_config['values']
        model_layer_name = self.modules_config['layer']
        dense_name = self.modules_config['dense']
        pre_layer_norm = self.modules_config['pre_layer_norm']
        if pre_layer_norm == 'True':
            pre_layer_norm = True
        elif pre_layer_norm == 'False':
            pre_layer_norm = False

        model_importance_list = []
        transformed_vectors_norm_list = []
        transformed_vectors_list = []
        resultants_list = []

        model_importance_list2 = []
        transformed_vectors_norm_list2 = []
        transformed_vectors_list2 = []
        resultants_list2 = []
        contributions_data = {}

        model_importance_list_l2 = []

        try:
            num_layers = self.model.config.n_layers
        except:
            num_layers = self.model.config.num_hidden_layers

        for layer in range(num_layers):
            hidden_states = hidden_states_model[layer].detach()
            attention_probs = attentions[layer].detach()

            #   value_layer: Value vectors calculated in self-attention. (batch, num_heads, seq_length, head_size)
            #   dense: Dense layer in self-attention. nn.Linear(all_head_size, all_head_size)
            #   LayerNorm: nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            #   pre_ln_states: Vectors just before LayerNorm (batch, seq_length, all_head_size)

            value_layer = self.transpose_for_scores(func_outputs[model_layer_name + '.' + str(layer) + '.' + values_name][0])

            dense = self.get_module(self.model, dense_name, layer, model_layer_name)
            ln1 = self.get_module(self.model, ln1_name, layer, model_layer_name)
            ln2 = self.get_module(self.model, ln2_name, layer, model_layer_name)
            pre_ln_states = func_inputs[model_layer_name + '.' + str(layer) + '.' + ln1_name][0][0]
            post_ln_states = func_outputs[model_layer_name + '.' + str(layer) + '.' + ln1_name][0]
            pre_ln2_states = func_inputs[model_layer_name + '.' + str(layer) + '.' + ln2_name][0][0]
            if pre_ln2_states.dim() == 2:
                pre_ln2_states = pre_ln2_states.unsqueeze(0)
            post_LayerNorm_FFN = func_outputs[model_layer_name + '.' + str(layer) + '.' + ln2_name][0]
            if post_LayerNorm_FFN.dim() == 2:
                post_LayerNorm_FFN = post_LayerNorm_FFN.unsqueeze(0)

            # VW_O
            dense_bias = dense.bias
            dense = dense.weight.view(self.all_head_size, self.num_attention_heads, self.attention_head_size)
            transformed_layer = torch.einsum('bhsv,dhv->bhsd', value_layer, dense) #(batch, num_heads, seq_length, all_head_size)
            del dense

            # AVW_O
            # (batch, num_heads, seq_length, seq_length, all_head_size)
            weighted_layer = torch.einsum('bhks,bhsd->bhksd', attention_probs, transformed_layer)
            del transformed_layer
            
            # Sum each weighted vectors αf(x) over all heads:
            # (batch, seq_length, seq_length, all_head_size)
            summed_weighted_layer = weighted_layer.sum(dim=1) # sum over heads
            del weighted_layer

            # Make residual matrix (batch, seq_length, seq_length, all_head_size)
            hidden_shape = hidden_states.size()
            device = hidden_states.device
            residual = torch.einsum('sk,bsd->bskd', torch.eye(hidden_shape[1]).to(device), hidden_states)

            # AVW_O + residual vectors -> (batch,seq_len,seq_len,embed_dim)
            residual_weighted_layer = summed_weighted_layer + residual
            
            #@torch.no_grad()
            def l_transform(x, w_ln):
                '''Computes mean and performs hadamard product with ln weight (w_ln) as a linear transformation.'''
                ln_param_transf = torch.diag(w_ln)
                ln_mean_transf = torch.eye(w_ln.size(0)).to(w_ln.device) - \
                    1 / w_ln.size(0) * torch.ones_like(ln_param_transf).to(w_ln.device)

                out = torch.einsum(
                    '... e , e f , f g -> ... g',
                    x,
                    ln_mean_transf,
                    ln_param_transf
                )
                return out

            if pre_layer_norm == False:
                # LN 1
                ln1_weight = ln1.weight.data
                ln1_eps = ln1.eps
                ln1_bias = ln1.bias

                # Transformed vectors T_i(x_j)
                transformed_vectors = l_transform(residual_weighted_layer, ln1_weight)
            else:
                transformed_vectors = residual_weighted_layer

            # Output vectors 1 per source token
            attn_output = transformed_vectors.sum(dim=2)

            if pre_layer_norm == False:
                # Lb_O
                dense_bias_term = l_transform(dense_bias, ln1_weight)
                # y_i
                ln_std_coef = 1/(pre_ln_states + ln1_eps).std(-1, unbiased=False).view(1, -1, 1)
                resultant = (attn_output + dense_bias_term)*ln_std_coef + ln1_bias
                transformed_vectors_std = l_transform(residual_weighted_layer, ln1_weight)*ln_std_coef.unsqueeze(-1)
                real_resultant = post_ln_states
            else:
                dense_bias_term = dense_bias
                resultant = attn_output + dense_bias_term
                transformed_vectors_std = transformed_vectors
                real_resultant = pre_ln2_states

            # Assert interpretable expression of attention is equal to the output of the attention block
            assert torch.dist(resultant, real_resultant).item() < 1e-3 * real_resultant.numel()
            del real_resultant

            #transformed_vectors_norm_std = torch.norm(transformed_vectors_std, dim=-1) # (batch, seq_len, seq_len)

            importance_matrix = -torch.nn.functional.pairwise_distance(transformed_vectors_std, resultant.unsqueeze(2),p=1)
            #importance_matrix_l2 = -torch.nn.functional.pairwise_distance(transformed_vectors_std, resultant.unsqueeze(2),p=2)

            
            model_importance_list.append(torch.squeeze(importance_matrix).cpu().detach())
            #model_importance_list_l2.append(torch.squeeze(importance_matrix_l2).cpu().detach())

            #transformed_vectors_norm_list.append(torch.squeeze(transformed_vectors_norm_std).cpu().detach())
            #transformed_vectors_list.append(torch.squeeze(transformed_vectors_std).cpu().detach())
            resultants_list.append(torch.squeeze(resultant).cpu().detach())

            """
            ############################
            # LN 2
            ln2_weight = ln2.weight.data
            ln2_eps = ln2.eps
            ln2_bias = ln2.bias

            ln2_std_coef = 1/(pre_ln2_states + ln2_eps).std(-1, unbiased=False).view(1, -1, 1) # (batch,seq_len,1)
            transformed_vectors_std2 = l_transform(transformed_vectors_std, ln2_weight)*ln2_std_coef.unsqueeze(-1)
            resultant2 = post_LayerNorm_FFN

            transformed_vectors_norm_std2 = torch.norm(transformed_vectors_std2, dim=-1) # (batch, seq_len, seq_len)

            importance_matrix2 = -torch.nn.functional.pairwise_distance(transformed_vectors_std2, resultant2.unsqueeze(2),p=1)

            model_importance_list2.append(torch.squeeze(importance_matrix2).cpu().detach())
            transformed_vectors_norm_list2.append(torch.squeeze(transformed_vectors_norm_std2).cpu().detach())
            transformed_vectors_list2.append(torch.squeeze(transformed_vectors_std2).cpu().detach())
            resultants_list2.append(torch.squeeze(resultant2).cpu().detach())
            """

        contributions_model = torch.stack(model_importance_list)
        ###
        #contributions_model_l2 = torch.stack(model_importance_list_l2)
        ###
        #transformed_vectors_norm_model = torch.stack(transformed_vectors_norm_list)
        #transformed_vectors_model = torch.stack(transformed_vectors_list)
        resultants_model = torch.stack(resultants_list)

        contributions_data['contributions'] = contributions_model
        ###
        #contributions_data['contributions_l2'] = contributions_model_l2
        ###
        #contributions_data['transformed_vectors'] = transformed_vectors_model
        #contributions_data['transformed_vectors_norm'] = transformed_vectors_norm_model
        contributions_data['resultants'] = resultants_model

        #contributions_model2 = torch.stack(model_importance_list2)
        #transformed_vectors_norm_model2 = torch.stack(transformed_vectors_norm_list2)
        #transformed_vectors_model2 = torch.stack(transformed_vectors_list2)
        #resultants_model2 = torch.stack(resultants_list2)

        #contributions_data['contributions2'] = contributions_model2
        #contributions_data['transformed_vectors2'] = transformed_vectors_model2
        #contributions_data['transformed_vectors_norm2'] = transformed_vectors_norm_model2
        #contributions_data['resultants2'] = resultants_model2

        return contributions_data

    def get_prediction(self, input_model):
        with torch.no_grad():
            output = self.model(input_model, output_hidden_states=True, output_attentions=True)
            prediction_scores = output['logits']

            return prediction_scores

    def get_module(self, model, module_name, layer, model_layer_name=None):
        parsed_module_name = module_name.split('.')
        tmp_module = model
        if model_layer_name:
            parsed_layer_name = model_layer_name.split('.')
            # Loop to find layers module
            for sub_module in parsed_layer_name:
                tmp_module = getattr(tmp_module, sub_module)
            # Select specific layer
            tmp_module = tmp_module[layer]
        # Loop over layer module to find module_name
        for sub_module in parsed_module_name:
            tmp_module = getattr(tmp_module, sub_module)
        return tmp_module

    def compute_joint_attention_alti(self, att_mat):
        # https://github1s.com/mt-upc/transformer-contributions/blob/main/src/utils_contributions.py#L455-L456
        """ Compute attention rollout given contributions or attn weights + residual."""
        joint_attentions = torch.zeros(att_mat.size()).to(att_mat.device)

        layers = joint_attentions.shape[0]

        joint_attentions = att_mat[0].unsqueeze(0)

        for i in range(1,layers):

            C_roll_new = torch.matmul(att_mat[i],joint_attentions[i-1])

            joint_attentions = torch.cat([joint_attentions, C_roll_new.unsqueeze(0)], dim=0)
            
        return joint_attentions

    def normalize_contributions(self, model_contributions,scaling='minmax',resultant_norm=None):
        """Normalization of the matrix of contributions/weights extracted from the model."""

        normalized_model_contributions = torch.zeros(model_contributions.size()).to(model_contributions.device)
        for l in range(0,model_contributions.size(0)):

            if scaling == 'min_max':
                ## Min-max normalization
                min_importance_matrix = model_contributions[l].min(-1, keepdim=True)[0]
                max_importance_matrix = model_contributions[l].max(-1, keepdim=True)[0]
                normalized_model_contributions[l] = (model_contributions[l] - min_importance_matrix) / (max_importance_matrix - min_importance_matrix)
                normalized_model_contributions[l] = normalized_model_contributions[l] / normalized_model_contributions[l].sum(dim=-1,keepdim=True)

            elif scaling == 'sum_one':
                normalized_model_contributions[l] = model_contributions[l] / model_contributions[l].sum(dim=-1,keepdim=True)
                #normalized_model_contributions[l] = normalized_model_contributions[l].clamp(min=0)

            # For l1 distance between resultant and transformer vectors we apply min_sum
            elif scaling == 'min_sum':
                if resultant_norm == None:
                    min_importance_matrix = model_contributions[l].min(-1, keepdim=True)[0]
                    normalized_model_contributions[l] = model_contributions[l] + torch.abs(min_importance_matrix)
                    normalized_model_contributions[l] = normalized_model_contributions[l] / normalized_model_contributions[l].sum(dim=-1,keepdim=True)
                else:
                    # print('resultant_norm[l]', resultant_norm[l].size())
                    # print('model_contributions[l]', model_contributions[l])
                    # print('normalized_model_contributions[l].sum(dim=-1,keepdim=True)', model_contributions[l].sum(dim=-1,keepdim=True))
                    normalized_model_contributions[l] = model_contributions[l] + torch.abs(resultant_norm[l].unsqueeze(1))
                    normalized_model_contributions[l] = torch.clip(normalized_model_contributions[l],min=0)
                    normalized_model_contributions[l] = normalized_model_contributions[l] / normalized_model_contributions[l].sum(dim=-1,keepdim=True)
            elif scaling == 'softmax':
                normalized_model_contributions[l] = torch.nn.functional.softmax(model_contributions[l], dim=-1)
            elif scaling == 't':
                model_contributions[l] = 1/(1 + model_contributions[l])
                normalized_model_contributions[l] =  model_contributions[l]/ model_contributions[l].sum(dim=-1,keepdim=True)
            else:
                print('No normalization selected!')
        return normalized_model_contributions

    def forward(self, inputs_embeds, attention_mask, token_type_ids):
        # NOTE: I'm not sure this is ready to be batched. In particular the way some of the normalization values are
        # computed seem to indicate otherwise. The same for resultants. So we get attributions example by example.
        # Also, in the original repository they also use it example by example.
        with torch.no_grad():
            self.handles = {}
            for name, module in self.model.named_modules():
                self.handles[name] = module.register_forward_hook(partial(self.save_activation, name))

            self.func_outputs = defaultdict(list)
            self.func_inputs = defaultdict(list)

            output = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )

            prediction_scores = output['logits'].detach()
            hidden_states = output['hidden_states']
            attentions = output['attentions']

            bs, *_ = inputs_embeds.shape
            msk_values = attention_mask.sum(dim=-1)

            all_normalized_contributions = []
            all_contributions_mix = []

            for example_ix in range(bs):
                # hidden_states: (num_layers + 1) x [batch_size, seq_length, embed_dim]
                hs = [x[example_ix, :msk_values[example_ix], :].unsqueeze(0) for x in hidden_states]
                # attentions: num_layers x [batch_size, num_heads, seq_length, seq_length]
                at = [x[example_ix, :, :msk_values[example_ix], :msk_values[example_ix]].unsqueeze(0) for x in attentions]
                # We keep the same structure as the original. func_inputs is list then tuple. func_outputs is list.
                fnc_inp = {k:[(v[0][0][example_ix, :msk_values[example_ix]].unsqueeze(0),)] for k, v in self.func_inputs.items()}
                fnc_out = {k:[v[0][example_ix, :msk_values[example_ix]].unsqueeze(0)] for k, v in self.func_outputs.items()}
                contributions_data = self.get_contributions(hs, at, fnc_inp, fnc_out)

                # Clean forward_hooks dictionaries
                self.clean_hooks()

                # NOTE: We want to use the proposed method
                # Original: https://github1s.com/mt-upc/transformer-contributions/blob/main/attributions.py#L147-L148
                # https://github.com/mt-upc/transformer-contributions/blob/main/Text_classification.ipynb
                resultant_norm = torch.norm(torch.squeeze(contributions_data['resultants']), p=1, dim=-1)
                normalized_contributions = self.normalize_contributions(contributions_data['contributions'],scaling='min_sum',resultant_norm=resultant_norm)
                contributions_mix = self.compute_joint_attention_alti(normalized_contributions)
                #contributions_mix_relevances = contributions_mix[:, 0]

                all_normalized_contributions.append(normalized_contributions)
                all_contributions_mix.append(contributions_mix)

                # NOTE: ALTI + L2 - We don't use this one
                # resultant_norm = torch.norm(torch.squeeze(contributions_data['resultants']),p=2,dim=-1)
                # normalized_contributions = self.normalize_contributions(contributions_data['contributions_l2'],scaling='min_sum',resultant_norm=resultant_norm)
                # contributions_mix = self.compute_joint_attention_alti(normalized_contributions)

                # NOTE: It is also possible to do this with resultants2, to compare with globenc - We don't use it

            return {
                "pred_logits": prediction_scores,
                "hidden_states": hidden_states,
                "attentions": attentions,
                "contributions": all_normalized_contributions,
                "contributions_aggregated": all_contributions_mix,
            }
