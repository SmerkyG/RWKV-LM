import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#model_id = "mistralai/Mixtral-8x7B-v0.1"
# teacher_model_id = "mistralai/Mixtral-8x7B-v0.1"
# teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)

# student_model_id = "RWKV/v5-Eagle-7B-HF"
# student_tokenizer = AutoTokenizer.from_pretrained(student_model_id, trust_remote_code=True)

from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
student_tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")
student_tokenizer.idx2token[0] = b'' # FIXME - hack to allow faster lookup and removal of rwkv zero tokens that separate regions of text

#print(student_tokenizer(" L")['input_ids'][0])

def pad_sequence_right(seq, max_len, pad_tok):
    return seq + [pad_tok] * (max_len - len(seq))


def calc_teacher2student_tok_idx(student_tokenizer, teacher_tokenizer):
    teacher_vocab = teacher_tokenizer.get_vocab()

    # find the longest matching student token for each Teacher token
    teacher2student_tok_idx = []
    for og_tok_str, teacher_tok_idx in sorted(teacher_vocab.items(), key=lambda x: x[1]):   
        tok_str = og_tok_str
        if og_tok_str[0] == '▁':
            # stupid problem because calling .decode would not return the leading space(s) for llama under HF tokenizer
            tok_str = og_tok_str.replace('▁', ' ')
            student_tok_idx = student_tokenizer.encode(tok_str)[0]
        elif og_tok_str in ['<unk>','<s>','</s>']:
            student_tok_idx = 0
        elif og_tok_str.startswith("<0x"):
            student_tok_idx = int(og_tok_str[1:-1], 16) + 1
        else:
            tok_str = teacher_tokenizer.decode(teacher_tok_idx)
            student_tok_idx = student_tokenizer.encode(tok_str)[0]
            
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


# new version that tokenizes teacher like student to ensure every student token has exactly one teacher token
def calc_student2teacher_seqidx(student_tokenizer, teacher_tokenizer, student_input_ids_batch):
    if isinstance(student_input_ids_batch, torch.Tensor):
        if len(student_input_ids_batch.shape) == 1:
            student_input_ids_batch = student_input_ids_batch.unsqueeze(0)
        student_input_ids_batch = student_input_ids_batch.tolist()
    elif not isinstance(student_input_ids_batch[0], list):
        student_input_ids_batch = [student_input_ids_batch]

    teacher_seq_idx_batch = []
    teacher_input_ids_batch = []
    for i, student_input_ids_line in enumerate(student_input_ids_batch):
        teacher_seq_idx_line = []
        teacher_input_ids_line = []
        last_i = 0
        for i, student_input_id in enumerate(student_input_ids_line):
            if student_input_id == 0:
                last_i = i + 1
                teacher_input_ids_line += [2, 1]
                teacher_seq_idx_line.append(-1)
                continue

            student_text = student_tokenizer.decode(student_input_ids_line[last_i:i+1])
            if student_text == '\ufffd':
                teacher_seq_idx_line.append(-1)
                continue;
            
            last_i = i + 1
            teacher_input_ids = teacher_tokenizer.encode('\0'+student_text, add_special_tokens=False)[2:]
            #print(student_text, teacher_input_ids)
            teacher_input_ids_line += teacher_input_ids
            teacher_seq_idx_line.append(len(teacher_input_ids_line)-1) # match the LAST teacher token in the range generated for this student token, not the first!

        teacher_input_ids_batch.append(teacher_input_ids_line)
        teacher_seq_idx_batch.append(teacher_seq_idx_line)
        #print(teacher_tokenizer.decode(teacher_input_ids_line))

    #print(teacher_input_ids_batch)
    #print(teacher_seq_idx_batch)
        
    max_length = max(len(t) for t in teacher_input_ids_batch)
    teacher_input_ids_batch = [pad_sequence_right(t, max_length, 0) for t in teacher_input_ids_batch]


    return teacher_seq_idx_batch, teacher_input_ids_batch, None

    




# NOTE - text positions are now compared at the END of a token, not the beginning, since this is where the next prediction would begin and is what has to match
def calc_student2teacher_seqidx_line(student_tokenizer, teacher_tokenizer, input_text, student_input_ids, teacher_seqidx_offset = 0):
    #print("input_text", input_text)
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
        token_text_len = len(student_tokenizer.idx2token[student_input_id]) # RWKV tokenizer specific
        student_text_end_offset += token_text_len
        if token_text_len > 0 and student_text_end_offset in teacher_text_end_offset_2_seqidx:
            student2teacher_seqidx.append(teacher_seqidx_offset + teacher_text_end_offset_2_seqidx[student_text_end_offset])
        else:
            student2teacher_seqidx.append(-1)
        #student_text_end_offset += len(student_tokenizer.decode([student_input_id]))
    #print("student2teacher_seqidx", student2teacher_seqidx)
    return student2teacher_seqidx

def calc_student2teacher_seqidx_old(student_tokenizer, teacher_tokenizer, student_input_ids_batch):
    if isinstance(student_input_ids_batch, torch.Tensor):
        if len(student_input_ids_batch.shape) == 1:
            student_input_ids_batch = student_input_ids_batch.unsqueeze(0)
        student_input_ids_batch = student_input_ids_batch.tolist()
    elif not isinstance(student_input_ids_batch[0], list):
        student_input_ids_batch = [student_input_ids_batch]

    input_text_batch = []
    for student_input_ids_line in student_input_ids_batch:
        input_text_batch.append(student_tokenizer.decode(student_input_ids_line))
    student2teacher_seqidx_batch = [calc_student2teacher_seqidx_line(student_tokenizer, teacher_tokenizer, input_text_line, student_input_ids_line) for input_text_line, student_input_ids_line in zip(input_text_batch, student_input_ids_batch)]
    return student2teacher_seqidx_batch, teacher_tokenizer(input_text_batch, return_tensors="pt", padding=True)['input_ids'], input_text_batch

def calc_student2teacher_seqidx_unbatched(student_tokenizer, teacher_tokenizer, student_input_ids_batch):
    if isinstance(student_input_ids_batch, torch.Tensor):
        if len(student_input_ids_batch.shape) == 1:
            student_input_ids_batch = student_input_ids_batch.unsqueeze(0)
        student_input_ids_batch = student_input_ids_batch.tolist()
    elif not isinstance(student_input_ids_batch[0], list):
        student_input_ids_batch = [student_input_ids_batch]

    input_text_batch = []
    for student_input_ids_line in student_input_ids_batch:
        input_text_batch.append(student_tokenizer.decode(student_input_ids_line))
    teacher_input_ids_batch = teacher_tokenizer(input_text_batch, padding=False)

    teacher_seqidx_offset = 0
    teacher_seqidx_unbatched = []
    teacher_input_ids_unbatched = []
    for i, student_input_ids_line in enumerate(student_input_ids_batch):
        input_text_line = student_tokenizer.decode(student_input_ids_line)
        teacher_seqidx_line = calc_student2teacher_seqidx_line(student_tokenizer, teacher_tokenizer, input_text_line, student_input_ids_line, teacher_seqidx_offset)
        teacher_seqidx_unbatched += teacher_seqidx_line

        teacher_input_ids_line = teacher_input_ids_batch[i]
        teacher_input_ids_unbatched += teacher_input_ids_line
    
        teacher_seqidx_offset += len(teacher_input_ids_line)

    return teacher_seqidx_unbatched, teacher_input_ids_unbatched, input_text_batch



# now we have a list that maps from student token offset to teacher token offset, or -1 if there is no correspondence

# use this to calculate KL loss where a teacher token offset exists by comparing logits for everything in the teacher2student_tok_idx mapping

# FIXME - what should we do in the case where the student has a better (longer) token than the teacher? e.g. ' Prepare' versus ' Pre''pare'
# we could just ignore those offsets, but probably better not to?

#print(calc_student2teacher_seqidx(student_tokenizer, teacher_tokenizer, student_tokenizer("Hello, my name is Inigo Montoya. Prepare to die. Your mother was a hamster, and your father smelt of elderberries.")['input_ids']))

if __name__ == "__main__":
    teacher_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" #"mistralai/Mixtral-8x7B-v0.1"
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_id, trust_remote_code=True)

    student_model_id = "RWKV/v5-Eagle-7B-HF"
    #student_tokenizer = AutoTokenizer.from_pretrained(student_model_id, trust_remote_code=True)
    #student_tokenizer.encoder[0] = bytes() # FIXME - hack to allow faster lookup


    teacher2student_tok_idx = calc_teacher2student_tok_idx(student_tokenizer, teacher_tokenizer)
    #print(teacher2student_tok_idx)

    inputs = [
        "Hello, my name is Inigo Montoya. Prepare to die. Your mother was a hamster, and your father smelt of elderberries.",
        "Only fools fall in but a lover is a lifetime achievement of crazy epic proportionality."
    ]
    # bugfix for HF RWKV tokenizer left padding batches, so we have to do it ourselves manually to right pad
    # fortunately this is just needed in this test code
    pad_token_id = 0
    student_input_ids = [student_tokenizer.encode(input_line) for input_line in inputs]
    max_length = max(len(t) for t in student_input_ids)
    student_input_ids = [pad_sequence_right(t, max_length, pad_token_id) for t in student_input_ids]

    #student_input_ids = [[66, 45439, 47813, 3460, 8110, 34991, 62, 25857, 3336, 74, 21751, 36208, 92, 640, 634, 46, 627, 46, 628, 3496, 59, 650, 59, 625, 45, 622, 57, 59, 36603, 48, 23900, 48659, 94, 55839, 59, 44086, 51291, 4811, 4418, 2143, 5385, 49545, 5383, 65, 630, 47, 718, 47, 705, 47, 712, 59, 676, 692, 48, 49545, 59, 28352, 92, 5892, 2073, 3527, 94, 61328, 39498, 4450, 31645, 47, 28352, 40835, 37635, 4596, 285, 47, 620, 52826, 5362, 28352, 92, 640, 634, 46, 627, 46, 628, 3496, 59, 650, 59, 627, 45, 630, 50, 59, 36603, 48, 23900, 48659, 94, 55839, 59, 44086, 51291, 4811, 4418, 2143, 5385, 49545, 5383, 65, 630, 47, 718, 47, 705, 47, 712, 59, 676, 692, 48, 49545, 59, 28352, 92, 5892, 2073, 3527, 94, 61328, 39498, 4450, 31645, 47, 28352, 40835, 37635, 4596, 287, 47, 620, 52826, 5362, 261, 60848, 38932, 31198, 52445, 3341, 19617, 52952, 31073, 4660, 32464, 308, 4418, 52243, 31077, 298, 261, 5585, 41693, 59, 308, 38547, 22590, 61328, 31871, 4450, 31645, 38392, 45, 21265, 50970, 44748, 1241, 22748, 22187, 59961, 4712, 22590, 31731, 4715, 39934, 22748, 332, 56242, 55638, 4601, 47, 29889, 22748, 22187, 22590, 30405, 47, 261, 48699, 44748, 1241, 4712, 45004, 3489, 47, 624, 47, 51, 3982, 84, 45, 53275, 59882, 4450, 21280, 59, 286, 47, 54, 47, 56, 46, 50, 42396, 49, 47, 636, 47, 624, 47, 50, 261, 6699, 46731, 4706, 274, 26309, 8782, 280, 98, 40, 21700, 44748, 460, 31731, 3529, 692, 46471, 31296, 59, 36208, 37, 31510, 8782, 280, 98, 125, 25529, 280, 106, 4418, 2143, 28352, 8786, 55, 43914, 49, 36216, 49, 4369, 737, 59, 1745, 2143, 65087, 1652, 737, 727, 65359, 23842, 978, 261, 48533, 45, 44748, 22748, 4596, 30808, 59961, 4712, 30322, 22590, 20097, 53, 50855, 21265, 22590, 20097, 55, 50855, 47, 308, 57525, 32234, 32487, 22166, 7211, 47, 3336, 11, 1500, 39446, 44748, 30917, 50919, 52167, 45, 22799, 31059, 4811, 22459, 4833, 332, 32355, 50845, 32487, 60522, 31296, 22590, 59725, 55786, 59, 36208, 27096, 46979, 2050, 7296, 21227, 96, 27370, 45439, 45439, 54228, 28352, 27096, 46979, 2050, 7296, 21227, 96, 2258, 8421, 45439, 96, 2258, 8421, 28352, 27096, 46979, 2050, 7296, 22459, 96, 27370, 96, 27108, 45439, 45439, 8774, 28352, 27096, 46979, 2050, 7296, 22459, 96, 63084, 280, 113, 45439, 96, 2258, 8421, 45439, 281, 43, 281, 43, 281, 43, 3328, 6699, 38809, 22748, 4677, 21053, 47, 29398, 46471, 31296, 59, 36208, 877, 1313, 982, 96, 6747, 296, 274, 1745, 2143, 5385, 24970, 2174, 59, 24970, 2174, 54228, 65, 630, 47, 49, 47, 52, 47, 52, 609, 40, 261, 32883, 32227, 22276, 27283, 26219, 51791, 39708, 4600, 22590, 31496, 4706, 22590, 353, 25617, 47, 36232, 45, 308, 30949, 4677, 353, 25617, 22590, 31496, 45439, 96, 2258, 8421, 47, 4255, 51291, 45, 308, 46650, 4811, 45450, 22590, 21053, 4811, 59, 36208, 877, 1313, 982, 96, 6747, 296, 274, 1745, 2143, 5385, 24970, 2174, 59, 24970, 2174, 54228, 65, 630, 47, 49, 47, 52, 47, 52, 48, 24970, 2174, 96, 2258, 8421, 40, 261, 74, 61885, 32234, 52157, 4424, 22590, 44748, 31321, 4596, 282, 9008, 48, 8110, 48, 42123, 2050, 47, 37451, 460, 332, 281, 8110, 30853, 32227, 59385, 22590, 59725, 38392, 32465, 308, 21795, 22590, 40229, 21053, 59, 36208, 124, 34536, 1914, 1988, 96, 34376, 45, 49476, 45, 49, 45, 28352, 124, 1745, 2143, 96, 34376, 45, 40891, 96, 8581, 27369, 45, 28352, 35, 40891, 4811, 353, 25617, 19287, 52678, 21700, 32355, 274, 24970, 2174, 5204, 28352, 40, 60964, 47, 26493, 463, 126, 11, 0, 261, 24281, 59, 3637, 61989, 4715, 47177, 52206, 269, 7373, 461, 59270, 22614, 38328, 35, 261, 74, 4418, 52157, 21700, 332, 47177, 4715, 45534, 61989, 4811, 22441, 32227, 4601, 4600, 22187, 37629, 24381, 4811, 61580, 4715, 59270, 60522, 45285, 22799, 31223, 22590, 45098, 52382, 47, 308, 4418, 57419, 37598, 60190, 59596, 4715, 56258, 38981, 52738, 45, 21400, 30135, 31458, 56155, 64388, 37598, 332, 56895, 56894, 52382, 47, 261, 5585, 41693, 59, 277, 5822, 461, 31979, 22590, 32039, 45285, 22799, 5231, 45431, 22590, 30255, 523, 261, 43, 33292, 31979, 22590, 30255, 460, 32039, 45285, 22226, 21800, 46368, 22590, 37824, 523, 261, 43, 6699, 21533, 31051, 461, 30259, 30410, 275, 9122, 495, 47, 261, 43, 6699, 38491, 48, 26102, 32497, 31051, 461, 30259, 31929, 48, 42293, 523, 261, 43, 1141, 31051, 461, 21256, 30259, 31929, 22797, 523, 261, 43, 1141, 21245, 461, 275, 6984, 42, 31601, 275, 7005, 30666, 42, 22797, 523, 261, 43, 6393, 31475, 22187, 4435, 22614, 31053, 122, 4596, 22226, 460, 31853, 34612, 523, 0, 82, 59, 50276, 36638, 50437, 19907, 4704, 106, 96, 49133, 472, 36638, 275, 6420, 42, 261, 74, 40219, 31296, 4811, 21751, 22590, 57271, 38392, 52221, 4588, 332, 39415, 38442, 4596, 44665, 3483, 104, 47, 267, 5936, 37059, 45, 20595, 21800, 22590, 39083, 96, 34376, 472, 56292, 32227, 21413, 47102, 51454, 37598, 22763, 332, 39415, 45953, 47, 267, 74, 38008, 22590, 22283, 47, 8235, 46509, 21700, 22590, 4704, 106, 96, 49133, 472, 267, 54014, 45, 21400, 30917, 32464, 308, 22449, 4601, 31581, 52743, 38448, 4712, 30810, 47, 261, 74, 40013, 40085, 4704, 96, 34376, 5266, 21400, 308, 4418, 22187, 51844, 55521, 30917, 4601, 47, 261, 23727, 4600, 332, 30469, 47169, 59, 19241, 37, 7499, 296, 37747, 5269, 3331, 37, 102, 296, 283, 60, 19241, 609, 33024, 53140, 21256, 38485, 21265, 46300, 22168, 52663, 31162, 55912, 3331, 7579, 467, 103, 62, 49, 60, 271, 103, 61, 42270, 467, 34449, 502, 271, 103, 5337, 3331, 124, 19242, 37, 49297, 96, 41900, 296, 271, 34449, 1645, 103, 6865, 49297, 96, 41900, 5224, 19242, 37, 27046, 96, 2215, 296, 271, 34449, 1645, 103, 6865, 27046, 96, 2215, 5224, 19242, 37, 25107, 296, 271, 34449, 1645, 103, 6865, 25107, 5224, 28350, 37, 8743, 296, 269, 41638, 31162, 52049, 275, 25112, 1857, 96, 1940, 45, 52049, 96, 41900, 45, 52049, 96, 25107, 42, 43913, 42482, 3420, 37, 27046, 96, 2215, 444, 3395, 49297, 96, 41900, 444, 3395, 25107, 441, 389, 36210, 19242, 37, 8753, 101, 296, 4704, 106, 96, 35358, 467, 25060, 45, 271, 8743, 502, 19242, 37, 42175, 296, 4704, 106, 96, 49133, 467, 8753, 101, 502, 19242, 19242, 609, 1130, 39415, 38442, 19242, 1942, 464, 37, 42175, 42, 19242, 124]]


    #student_input_ids = [[33155,0,33155,0,33155,0]]
    #student_input_ids = torch.tensor(student_input_ids, dtype=torch.long)

    teacher_seqidx, teacher_input_ids_list, input_text_batch = calc_student2teacher_seqidx(student_tokenizer, teacher_tokenizer, student_input_ids)
    teacher_input_ids = torch.tensor(teacher_input_ids_list).long()

    print("input text reformed")
    print(input_text_batch)

    for i in range(len(student_input_ids)):
    #     student_tokens = [student_tokenizer.decode([id]) for id in student_input_ids[i]]
    #     print("student tokens")
    #     print(student_tokens)
        print("teacher tokens")
        teacher_tokens = [teacher_tokenizer.decode([id]) for id in teacher_input_ids_list[i]]
        print(teacher_tokens)

    print("\nstudent2teacher_seqidx")
    print(teacher_seqidx)
    total_tok_count = sum(map(len, teacher_seqidx))
    print(total_tok_count - sum(map(lambda x: x.count(-1), teacher_seqidx)), '/', total_tok_count)

    teacher_seqidx = torch.tensor(teacher_seqidx).long()

    B,S = teacher_seqidx.shape
    print("TEACHER INPUT IDS", teacher_input_ids.tolist())
    #teacher_input_ids = teacher_input_ids.view(-1)[teacher_seqidx.view(-1)].view(B,S) # THIS DOESNT WORK BC THE INDICES ARE NOT OVER THE view(-1) SIZE, BUT OVER EACH BATCH!
    
    use_tok = teacher_seqidx >= 0 # bool for each input, indicates if there's a student<->teacher token text-offset match
    # using -1's won't work here..so we replace those with zeros via multiplication with use_tok
    teacher_seqidx = teacher_seqidx * use_tok

    teacher_input_ids = torch.gather(input=teacher_input_ids, dim=-1, index=teacher_seqidx)
    print("TEACHER INPUT IDS", teacher_input_ids.tolist())
    teacher2student_tok_idx = torch.Tensor(teacher2student_tok_idx).long()
    teacher_targets = teacher2student_tok_idx[teacher_input_ids]
    print('\nstudent input_ids')
    print(student_input_ids)
    print('\nteacher_targets')
    print(teacher_targets.tolist())

    print(teacher_targets.eq(torch.tensor(student_input_ids).long()))


    # print('\nstudent_input_ids')
    # print(student_input_ids)
    # print("\nteacher_input_ids")
    # print(teacher_input_ids)
    




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
