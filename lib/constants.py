from dataclasses import dataclass


@dataclass
class Model:
    name: str
    id: str
    chat_template_file: str
    stop_tokens: str


LLAMA_2_BASE = 'llama-2-base'
LLAMA_2_ICL = 'llama-2-icl'
LLAMA_2_CHAT = 'llama-2-chat'

LLAMA_2_ID = 'meta-llama/Llama-2-7b-hf'
LLAMA_2_CHAT_ID = 'meta-llama/Llama-2-7b-chat-hf'

URIAL_0_FILE = './lib/template_library/urial_0.jinja'
URIAL_1K_FILE = './lib/template_library/urial_1k.jinja'

URIAL_STOP_TOKENS = ['```']
LLAMA_STOP_TOKENS = ['</s>']


MODELS_MAP = {
    LLAMA_2_BASE: Model(LLAMA_2_BASE, LLAMA_2_ID, URIAL_0_FILE, URIAL_STOP_TOKENS),
    LLAMA_2_ICL: Model(LLAMA_2_ICL, LLAMA_2_ID, URIAL_1K_FILE, URIAL_STOP_TOKENS),
    LLAMA_2_CHAT: Model(LLAMA_2_CHAT, LLAMA_2_CHAT_ID, None, LLAMA_STOP_TOKENS),
}
