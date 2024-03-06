import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#model_id = "mistralai/Mixtral-8x7B-v0.1"
# teacher_model_id = "mistralai/Mixtral-8x7B-v0.1"
# teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)

# student_model_id = "RWKV/v5-Eagle-7B-HF"
# student_tokenizer = AutoTokenizer.from_pretrained(student_model_id, trust_remote_code=True)

#from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
#student_tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")

#print(student_tokenizer(" L")['input_ids'][0])

def calc_teacher2student_tok_idx(student_tokenizer, teacher_tokenizer):
    teacher_vocab = teacher_tokenizer.get_vocab()

    # find the longest matching student token for each Teacher token
    teacher2student_tok_idx = []
    for og_tok_str, teacher_tok_idx in sorted(teacher_vocab.items(), key=lambda x: x[1]):   
        tok_str = og_tok_str
        if og_tok_str[0] == '▁':
            # stupid problem because calling .decode would not return the leading space(s) for llama under HF tokenizer
            tok_str = og_tok_str.replace('▁', ' ')
            student_tok_idx = student_tokenizer(tok_str)['input_ids'][0]
        elif og_tok_str in ['<unk>','<s>','</s>']:
            student_tok_idx = 0
        elif og_tok_str.startswith("<0x"):
            student_tok_idx = int(og_tok_str[1:-1], 16) + 1
        else:
            tok_str = teacher_tokenizer.decode(teacher_tok_idx)
            student_tok_idx = student_tokenizer(tok_str)['input_ids'][0]
            
        teacher2student_tok_idx.append(student_tok_idx)

        # # print teacher tokens that don't have exact matches in student
        # if student_tok_idx != 0:
        #     s = student_tokenizer.decode([student_tok_idx])
        #     if student_tok_idx < 257:
        #         continue
        #     try:
        #         s = s.decode('utf-8')
        #     except:
        #         pass        
        #     if tok_str != s:
        #         print(teacher_tok_idx, student_tok_idx, f"'{og_tok_str}'", s if isinstance(s, bytes) else f"'{s}'")

        # if len(teacher2student_tok_idx) > 512:
        #     break
    return teacher2student_tok_idx

#teacher2student_tok_idx = torch.LongTensor(calc_teacher2student_tok_idx(student_tokenizer, teacher_tokenizer))

#exit()

def calc_student2teacher_tok_idx(student_tokenizer, teacher_tokenizer):
    teacher2student_tok_idx = calc_teacher2student_tok_idx(student_tokenizer, teacher_tokenizer)

    student2teacher_tok_idx = [0] * 65536 #len(student_tokenizer) # FIXME - no way to know the max token count really, so hardcoded it for now
    for teacher_tok_idx, student_tok_idx in enumerate(teacher2student_tok_idx):
        student2teacher_tok_idx[student_tok_idx] = teacher_tok_idx

    # FIXME - consider if we want to somehow calc fractional ones, in which case we can't use this kind of return value
    return student2teacher_tok_idx


# student and teacher will tokenize differently, but should match up at a majority of locations (e.g. at spaces and the most common non-space tokens)
# we need the KL loss to count only at points where these match up
# so have each one generate a bool array of byte positions, with True in positions that they have a token start

# we need an array of student to teacher token position offsets where the text start position matches, so we know which to generate the KL loss against (some will not be supposed to do KL loss when there is no start position match)



# # # FIXME - decode from dataset input_ids
# student_input_ids = student_tokenizer("Hello, my name is Inigo Montoya. Prepare to die. Your mother was a hamster, and your father smelt of elderberries.")['input_ids']
# # #student_tokens = student_tokenizer.convert_ids_to_tokens(student_input_ids) # this stupid crap didn't work, had to do it manually
# student_tokens = [student_tokenizer.decode([id]) for id in student_input_ids]
# print(student_tokens)
# input_text = student_tokenizer.decode(student_input_ids)
# teacher_tokens = [teacher_tokenizer.decode([id]) for id in teacher_tokenizer(input_text)['input_ids']]
# print(teacher_tokens)

# import itertools

# input_text = student_tokenizer.decode(student_input_ids)
# #student_text_offsets = student_tokenizer(input_text, return_offsets_mapping=True)['offset_mapping'][..., 0]
# student_text_offsets = list(itertools.accumulate(map(len, student_tokens)))
# print(student_text_offsets)

# teacher_text_offsets = [x[0] for x in teacher_tokenizer(input_text, return_offsets_mapping=True)['offset_mapping'][1:]]
# print(teacher_text_offsets)



# NOTE - text positions are now compared at the END of a token, not the beginning, since this is where the next prediction would begin and is what has to match
def calc_student2teacher_seqidx_line(student_tokenizer, teacher_tokenizer, input_text, student_input_ids):
    print("input_text", input_text)
    teacher_inputs = teacher_tokenizer(input_text, return_offsets_mapping=True)
    #print(teacher_inputs['input_ids'])
    teacher_text_offsets = [x[0] for x in teacher_inputs['offset_mapping'][1:]]
    teacher_text_end_offset_2_seqidx = {}
    teacher_tok_num = 0
    for teacher_tok_num, teacher_offset in enumerate(teacher_text_offsets):
        if teacher_tok_num > 0:
            teacher_text_end_offset_2_seqidx[teacher_offset] = teacher_tok_num # NOTE - would be teacher_tok_num-1 but llama tokenizer adds a BOS token always
    teacher_text_end_offset_2_seqidx[len(input_text)] = teacher_tok_num+1 # put in the last one at the end of the text since the iteration only sees starts not ends of token texts
    #print("teacher_text_end_offset_2_seqidx", teacher_text_end_offset_2_seqidx)
    student2teacher_seqidx = []
    student_text_end_offset = 0
    for student_input_id in student_input_ids:
        token_text_len = len(student_tokenizer.encoder[student_input_id.item()]) # RWKV tokenizer specific
        student_text_end_offset += token_text_len
        if token_text_len > 0 and student_text_end_offset in teacher_text_end_offset_2_seqidx:
            student2teacher_seqidx.append(teacher_text_end_offset_2_seqidx[student_text_end_offset])
        else:
            student2teacher_seqidx.append(-1)
        #student_text_end_offset += len(student_tokenizer.decode([student_input_id]))
    print("student2teacher_seqidx", student2teacher_seqidx)
    return student2teacher_seqidx

def calc_student2teacher_seqidx(student_tokenizer, teacher_tokenizer, student_input_ids_batch):
    if isinstance(student_input_ids_batch, torch.Tensor):
        if len(student_input_ids_batch.shape) == 1:
            student_input_ids_batch = student_input_ids_batch.unsqueeze(0)
    elif not isinstance(student_input_ids_batch[0], list):
        student_input_ids_batch = [student_input_ids_batch]

    input_text_batch = student_tokenizer.batch_decode(student_input_ids_batch)
    student2teacher_seqidx_batch = [calc_student2teacher_seqidx_line(student_tokenizer, teacher_tokenizer, input_text_line, student_input_ids_line) for input_text_line, student_input_ids_line in zip(input_text_batch, student_input_ids_batch)]
    return student2teacher_seqidx_batch, teacher_tokenizer(input_text_batch, return_tensors="pt", padding=True)['input_ids']



# now we have a list that maps from student token offset to teacher token offset, or -1 if there is no correspondence

# use this to calculate KL loss where a teacher token offset exists by comparing logits for everything in the teacher2student_tok_idx mapping

# FIXME - what should we do in the case where the student has a better (longer) token than the teacher? e.g. ' Prepare' versus ' Pre''pare'
# we could just ignore those offsets, but probably better not to?

#print(calc_student2teacher_seqidx(student_tokenizer, teacher_tokenizer, student_tokenizer("Hello, my name is Inigo Montoya. Prepare to die. Your mother was a hamster, and your father smelt of elderberries.")['input_ids']))

if __name__ == "__main__":
    teacher_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" #"mistralai/Mixtral-8x7B-v0.1"
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_id, trust_remote_code=True)

    student_model_id = "RWKV/v5-Eagle-7B-HF"
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_id, trust_remote_code=True)
    student_tokenizer.encoder[0] = bytes() # FIXME - hack to allow faster lookup

    teacher2student_tok_idx = calc_teacher2student_tok_idx(student_tokenizer, teacher_tokenizer)
    #print(teacher2student_tok_idx)

    def pad_sequence_right(seq, max_len, pad_tok):
        return seq + [pad_tok] * (max_len - len(seq))
    
    inputs = [
        "Hello, my name is Inigo Montoya. Prepare to die. Your mother was a hamster, and your father smelt of elderberries.",
        "Only fools fall in but a lover is a lifetime achievement of crazy epic proportionality."
    ]
    # bugfix for HF RWKV tokenizer left padding batches, so we have to do it ourselves manually to right pad
    # fortunately this is just needed in this test code
    pad_token_id = 0
    student_input_ids = [student_tokenizer(input_line)['input_ids'] for input_line in inputs]
    max_length = max(len(t) for t in student_input_ids)
    student_input_ids = [pad_sequence_right(t, max_length, pad_token_id) for t in student_input_ids]

    for i in range(len(inputs)):
        student_tokens = [student_tokenizer.decode([id]) for id in student_input_ids[i]]
        print(student_tokens)
        teacher_tokens = [teacher_tokenizer.decode([id]) for id in teacher_tokenizer(inputs[i])['input_ids']]
        print(teacher_tokens)

    student_input_ids = torch.tensor(student_input_ids, dtype=torch.long)
    print('\nstudent_input_ids')
    print(student_input_ids)

    seqidx, teacher_tokenization = calc_student2teacher_seqidx(student_tokenizer, teacher_tokenizer, student_input_ids)
    print("\nstudent2teacher_seqidx")
    print(seqidx)
    print("\nteacher_input_ids")
    print(teacher_tokenization)
    




# teacher_seqidxs = teacher_inputs['offset_mapping'][..., 0]
# teacher_tok_start_map = torch.zeros(input_chars_bytelen, dtype=torch.bool, device=)
# teacher_tok_start_map[teacher_seqidxs] = True

# student_seqidxs = torch.cumsum(length_by_token_id[student_input_ids])
# student_tok_start_map = torch.zeros(input_chars_bytelen, dtype=torch.bool, device=)
# student_tok_start_map[student_seqidxs] = True

# text = "Hello my name is"
# teacher_inputs = teacher_tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
# print(teacher_inputs)
# #teacher_inputs = teacher_inputs['input_ids']
# #print(input_ids['input_ids'])

# #model = AutoModelForCausalLM.from_pretrained(teacher_model_id, load_in_4bit=True)
# teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_id).to(0)

# # #             input_ids: (`optional`) `torch.LongTensor` of shape `(batch_size, sequence_length)`
# # #                The sequence used as a prompt for the generation. If `None` the method initializes
# # #                it as an empty `torch.LongTensor` of shape `(1,)`.
# # model_inputs = model.prepare_inputs_for_generation(
# #     input_ids, past=None, attention_mask=None, use_cache=False
# # )

# outputs = teacher_model(input_ids=teacher_inputs['input_ids'].to(0))
# print(outputs)
# # next_token_logits = outputs[0][:, -1, :]
# # print(next_token_logits)

# #outputs = model.generate(**inputs, max_new_tokens=20)
# #print(outputs[1])
# #print(tokenizer.decode(outputs[0], skip_special_tokens=True))
