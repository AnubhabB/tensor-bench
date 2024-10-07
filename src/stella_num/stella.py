import os

import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize
import safetensors
from safetensors.torch import save_file

with open('samples.txt', 'r') as file:
    data = file.read().split("\n")

model_dir = "stella-cache"
vector_dims = [256,1024,4096]

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to("cuda").to(torch.float32)

for v in vector_dims:
    vector_linear_directory = f"2_Dense_{v}"
    vector_linear = torch.nn.Linear(in_features=model.config.hidden_size, out_features=v).to("cuda")
    vector_linear_dict = {
        k.replace("linear.", ""): v for k, v in
        torch.load(os.path.join(model_dir, f"{vector_linear_directory}/pytorch_model.bin"), weights_only=True).items()
    }
    vector_linear.load_state_dict(vector_linear_dict)
    vector_linear.cuda()

    # Embed the queries
    with torch.no_grad():
        input_data = tokenizer(data, padding="longest", truncation=True, max_length=512, return_tensors="pt")
        input_data = {k: v.cuda() for k, v in input_data.items()}
        print(input_data["attention_mask"].shape, input_data["input_ids"].shape)
        attention_mask = input_data["attention_mask"]
        last_hidden_state = model(**input_data)[0]
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        query_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        out = torch.Tensor(normalize(vector_linear(query_vectors.to(torch.float32)).cpu().numpy()))
        print(f"{v}:\n", out)
        
        save_file({"w": out}, f"stella_{v}.safetensors")
        

    del vector_linear
    