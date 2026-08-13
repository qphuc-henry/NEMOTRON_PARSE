from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

t = AutoTokenizer.from_pretrained('/home/cqphuc/NVIDIA-Nemotron-Parse-2.0', trust_remote_code=True)
prompt1 = "<predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
prompt2 = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"

print("Prompt 1:", t.encode(prompt1))
print("Prompt 2:", t.encode(prompt2))
