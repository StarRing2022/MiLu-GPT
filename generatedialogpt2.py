from transformers import BertTokenizerFast,GPT2LMHeadModel,GenerationConfig
from transformers import pipeline, set_seed 
import torch
import gradio as gr
from gptbot import GPTBot

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

#torch.set_default_tensor_type(torch.cuda.HalfTensor)


tokenizer = BertTokenizerFast.from_pretrained("StarRing2022/MiLu-GPT", add_special_tokens=True)
model = GPT2LMHeadModel.from_pretrained("StarRing2022/MiLu-GPT",device_map='auto')

# PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

# Instruction:
# {instruction}

# Response:"""

# # input_ids = tokenizer(PROMPT_FORMAT.format(instruction='你是我同学吗？'), return_tensors="pt").input_ids.to("cuda")
# # out = model.generate(input_ids=input_ids)
# # answer = tokenizer.decode(out[0])
# # print(answer)

# def evaluate(instruction):
#   with torch.no_grad():
#     input_ids = tokenizer(PROMPT_FORMAT.format(instruction=instruction), return_tensors="pt").input_ids.to("cuda")
#     out = model.generate(
#             input_ids=input_ids,
#             max_new_tokens=128,
#         )
#     answer = tokenizer.decode(out[0])
    
#     return answer.split("response : [SEP] ")[1].split("[SEP]")[0].strip()
# gr.Interface(
#     fn=evaluate,#接口函数
#     inputs=[
#         gr.components.Textbox(
#             lines=2, label="Instruction", placeholder="聊天内容"
#         ),
#     ],
#     outputs=[
#         gr.inputs.Textbox(
#             lines=5,
#             label="Output",
#         )
#     ],
#     title="ChatUni",
#     description="Chat,Your Own World",
# ).launch()

#----------------------------------

# def generate_prompt(instruction, input=None):
#     if input:
#         return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# ### Instruction:
# {instruction}

# ### Input:
# {input}

# ### Response:"""
#     else:
#         return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

# ### Instruction:
# {instruction}

# ### Response:"""

# def evaluate(
#     instruction,
#     input=None,
#     temperature=0.1,
#     top_p=0.75,
#     top_k=40,
#     num_beams=4,
#     max_new_tokens=128,
#     **kwargs,
# ):
#     prompt = generate_prompt(instruction, input)
#     inputs = tokenizer(prompt, return_tensors="pt")
#     input_ids = inputs["input_ids"].to(device)
#     generation_config = GenerationConfig(
#         temperature=temperature,
#         top_p=top_p,
#         top_k=top_k,
#         num_beams=num_beams,
#         **kwargs,
#     )
#     with torch.no_grad():
#         generation_output = model.generate(
#             input_ids=input_ids,
#             generation_config=generation_config,
#             return_dict_in_generate=True,
#             output_scores=True,
#             max_new_tokens=max_new_tokens,
#         )
#     s = generation_output.sequences[0]
#     output = tokenizer.decode(s)
#     return output.split("# # # response : [SEP]")[1].split("[SEP]")[0].strip()


# gr.Interface(
#     fn=evaluate,#接口函数
#     inputs=[
#         gr.components.Textbox(
#             lines=2, label="指令", placeholder="告诉我羊驼是什么."
#         ),
#         gr.components.Textbox(lines=2, label="输入内容", placeholder="输入文本"),
#         gr.components.Slider(minimum=0, maximum=1, value=0.1, label="随机性"),
#         gr.components.Slider(minimum=0, maximum=1, value=0.75, label="注意力参数P"),
#         gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="注意力参数K"),
#         gr.components.Slider(minimum=1, maximum=4, step=1, value=4, label="波束搜索"),
#         gr.components.Slider(
#             minimum=1, maximum=2000, step=1, value=128, label="最大长度"
#         ),
#     ],
#     outputs=[
#         gr.inputs.Textbox(
#             lines=5,
#             label="输出内容",
#         )
#     ],
#     title="ChatUni",
#     description="Chat,Your Own World",
# ).launch()

#----------------------------------

#指令格式1
def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

#指令格式2
PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

Instruction:
{instruction}

Response:"""

def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    repetition_penalty=1.0,
    max_new_tokens=128,
    **kwargs,
):
    #prompt = generate_prompt(instruction, input)
    #print(generate_prompt(instruction, input))

    gptbot = GPTBot(model_name_or_path="dialogpt2", max_history_len=3,max_len=max_new_tokens,temperature=temperature,topk=top_k,topp=top_p,repetition_penalty=repetition_penalty )
    
    #output = gptbot.answer(PROMPT_FORMAT.format(instruction=instruction))
    #print(PROMPT_FORMAT.format(instruction=instruction))
    
    output = gptbot.answer(instruction+input)

    return output.strip()


gr.Interface(
    fn=evaluate,#接口函数
    inputs=[
        gr.components.Textbox(
            lines=2, label="指令", placeholder="告诉我羊驼是什么."
        ),
        gr.components.Textbox(lines=2, label="输入内容", placeholder="输入文本"),
        gr.components.Slider(minimum=0, maximum=1, value=0.1, label="随机性"),
        gr.components.Slider(minimum=0, maximum=1, value=0.75, label="注意力参数P"),
        gr.components.Slider(minimum=0, maximum=100, step=1, value=50, label="注意力参数K"),
        gr.components.Slider(minimum=1, maximum=100, step=1, value=50, label="重复惩罚参数"),
        gr.components.Slider(
            minimum=1, maximum=500, step=1, value=128, label="最大长度"
        ),
    ],
    outputs=[
        gr.inputs.Textbox(
            lines=5,
            label="输出内容",
        )
    ],
    title="ChatUni",
    description="Chat,Your Own World",
).launch()
