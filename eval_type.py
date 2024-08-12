import torch
import numpy as np
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor, LogitsProcessorList


def get_relative_top_filter(scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
    scores_normalized = scores.log_softmax(dim=-1) 
    sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)
    return scores_normalized < probs_thresh

def generate(args, Model, question, vector):
    if args.dataset == 'Gsm8k':
        stop_word_list = ["Q:", "\end{code}"]
    elif args.dataset == 'StraQA':
        stop_word_list = ["Q:", "\n\n##"]
    else:
        stop_word_list = []

    stopping_words = get_stop_wordids(Model.tokenizer, stop_word_list)

    inputs = Model.tokenizer(question, return_tensors="pt")
    input_ids = inputs.input_ids.to('cuda')
    input_ids_all = input_ids
    attention_mask = inputs.attention_mask.to('cuda')
    past_key_values = None
    new_tokens_list = []

    with torch.no_grad():
        processors = LogitsProcessorList()
        if not args.penalty:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=1.2))
        else:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=args.penalty))

        max_new_tokens = args.max_len
        for _ in range(max_new_tokens):
            outputs = Model.model(input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
                output_hidden_states=True,
                return_dict=True
            )
            logits = outputs.hidden_states
            past_key_values = outputs.past_key_values

            # combine start
            mature_layer = len(logits)-1
            head_layer = Model.model.get_output_embeddings()

            layer_dict = {int(i):0 for i in args.concerned_labels.split(',')}

            for premature_layer in layer_dict.keys():
                layer_dict[premature_layer] = head_layer(logits[premature_layer])[0, -1, :].log_softmax(dim=-1)

            if not args.wo_final:
                layer_dict[mature_layer] = head_layer(logits[mature_layer])[0, -1, :].log_softmax(dim=-1)
                final_logits = layer_dict[mature_layer]

            layer_list = [layer_dict[layer] for layer in layer_dict.keys()]
            vector_param = vector.param.unsqueeze(1)  # (num_labels+1, 1, vocab_size)
            # vector_param = vector.unsqueeze(1)
            layer_list = torch.stack(layer_list).unsqueeze(dim=1) # (num_labels+1, 1, vocab_size)

            layer_list = layer_list.to(vector_param.device)
            temp_logits = vector_param * layer_list  # (num_labels+1, 1, vocab_size)
            next_token_logits = temp_logits.sum(dim=0)  # (1, vocab_size)
            # combine end
            if not args.wo_final:
                relative_top_mask = get_relative_top_filter(final_logits, 0.1).to(next_token_logits.device)
                next_token_logits = torch.where(relative_top_mask, -1000, next_token_logits)

            next_token_logits = processors(input_ids_all, next_token_logits)
            next_token = torch.argmax(next_token_logits, dim=-1)
            new_tokens_list.append(next_token.item())
            
            if next_token.item() == Model.tokenizer.eos_token_id:
                break

            new_tokens_list, stop = if_stop(new_tokens_list, stopping_words)
            if stop:
                break

            input_ids = next_token.unsqueeze(0)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            input_ids_all = torch.cat([input_ids_all, next_token[:, None]], dim=-1)
    
        output_sequence = Model.tokenizer.decode(new_tokens_list, skip_special_tokens=True)
        return output_sequence

def base_gen(args, Model, question, useless):
    if args.dataset == 'Gsm8k':
        stop_word_list = ["Q:", "\end{code}"]
    elif args.dataset == 'StraQA':
        stop_word_list = ["Q:", "\n\n##"]
    else:
        stop_word_list = []

    stopping_words = get_stop_wordids(Model.tokenizer, stop_word_list)

    inputs = Model.tokenizer(question, return_tensors="pt")
    input_ids = inputs.input_ids.to('cuda')

    input_ids_all = input_ids
    attention_mask = inputs.attention_mask.to('cuda')
    past_key_values = None
    new_tokens_list = []

    with torch.no_grad():
        processors = LogitsProcessorList()
        if not args.penalty:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=1.0))
        else:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=args.penalty))

        max_new_tokens = args.max_len
        for _ in range(max_new_tokens):
            outputs = Model.model(input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
                output_hidden_states=True,
                return_dict=True
            )
            logits = outputs.hidden_states
            past_key_values = outputs.past_key_values
            
            logits = outputs.logits[:,-1,:]
            next_token_logits = logits.log_softmax(dim=-1)

            next_token_logits = processors(input_ids_all, next_token_logits)
            next_token = torch.argmax(next_token_logits, dim=-1)

            new_tokens_list.append(next_token.item())
            if next_token.item() == Model.tokenizer.eos_token_id:
                break

            new_tokens_list, stop = if_stop(new_tokens_list, stopping_words)
            if stop:
                break

            input_ids = next_token.unsqueeze(0)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            input_ids_all = torch.cat([input_ids_all, next_token[:, None]], dim=-1)
    
        output_sequence = Model.tokenizer.decode(new_tokens_list, skip_special_tokens=True)
        return output_sequence

# special
def get_stop_wordids(tokenizer, stop_words):
    if not stop_words:
        return None

    list_stop_word_ids = []
    for stop_word in stop_words:
        stop_word_ids = tokenizer.encode('\n' + stop_word)[3:]
        list_stop_word_ids.append(stop_word_ids)
        # print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
    return list_stop_word_ids

def if_stop(new_tokens_list, stopping_words):
    if not stopping_words:
        return new_tokens_list, False

    end_words = [i[-1] for i in stopping_words]
    new_token = new_tokens_list[-1]
    if new_token in end_words:
        idx = end_words.index(new_token)
        seq_len = len(new_tokens_list)
        word_len = len(stopping_words[idx])
        if new_tokens_list[seq_len-word_len:] == stopping_words[idx]:
            return new_tokens_list[:seq_len-word_len], True
        else:
            return new_tokens_list, False
    else:
        return new_tokens_list, False
