
import pytz
import loguru

from rag.rag import RagApplication,ApplicationConfig
from rag.reranker.bge_reranker import BgeRerankerConfig
from rag.retrieval.dense_retriever import DenseRetrieverConfig
from prompt import templates
from infer.qwen.source_infer import QwenMoelRun

from tokenizers import Tokenizer


sys_prompt = templates.SYSTEM_PROMPT
question = "请给出一个关于如何使用RAG的例子"

llm_model_path = "D:/code/transformer_models/models--Qwen--Qwen2.5-3B-Instruct/"
llm_model_path = "D:/code/transformer_models/models--Qwen--Qwen2.5-Coder-0.5B-Instruct-GPTQ-Int8/"
llm_model_path = "D:/code/transformer_models/models--Qwen--Qwen2.5-0.5B-Instruct/"

model = QwenMoelRun(llm_model_path)
model.repetition_penalty = 1.3
tokenizers = Tokenizer.from_file(llm_model_path+"tokenizer.json")
model.set_tokenizer(tokenizers)

app_config = ApplicationConfig()
app_config.docs_path = "d:/code/python/TrustRAG/docs"
app_config.docs_path = "d:/code/py/TrustRAG-main/docs"
app_config.llm_model_path = llm_model_path

retriever_config = DenseRetrieverConfig(
    model_name_or_path="D:/code/transformer_models/bge-large-zh-v1.5",
    dim=1024,
    index_path='./examples/retrievers/dense_cache'
)
rerank_config = BgeRerankerConfig(
    model_name_or_path="D:/code/transformer_models/bge-reranker-large"
)

app_config.retriever_config = retriever_config
app_config.rerank_config = rerank_config
application = RagApplication(tokenizers,model,app_config)
application.init_vector_store()

beijing_tz = pytz.timezone("Asia/Shanghai")
IGNORE_FILE_LIST = [".DS_Store"]



def shorten_label(text, max_length=10):
    if len(text) > 2 * max_length:
        return text[:max_length] + "..." + text[-max_length:]
    return text


def predict(question,
            top_k = 5,
            use_web = 'Use',
            use_pattern = "Rag",
            history=None):
    loguru.logger.info("User Question：" + question)
    if history is None:
        history = []
    # Handle web content
    web_content = ''
    if use_web == 'Use':
        loguru.logger.info("Use Web Search")
        results = application.web_searcher.retrieve(query=question, top_k=5)
        for search_result in results:
            web_content += search_result['title'] + " " + search_result['body'] + "\n"
    search_text = ''
    if use_pattern == 'Only LLM':
        # Handle model Q&A mode
        loguru.logger.info('Only LLM Mode:')

        # result = application.llm.chat(query=question, web_content=web_content)
        system_prompt = "You are a helpful assistant."
        user_input = [
            {"role": "user", "content": question}
        ]
        # 调用 chat 方法进行对话
        result, total_tokens = application.llm.chat(system=system_prompt, history=user_input)
        history.append((question, result))
        search_text += web_content
    
        return history, search_text,''
    elif use_pattern == 'Rag':
        # Handle RAG mode
        loguru.logger.info('R.AG Mode:')
        response, _, contents, rewrite_query = application.chat(
            question=question,
            top_k=top_k,
        )
        history.append((question, response))
        # Format search results
        for idx, source in enumerate(contents):
            sep = f'----------【搜索结果{idx + 1}：】---------------\n'
            search_text += f'{sep}\n{source["text"]}\n分数：{source["score"]:.2f}\n\n'
        # Add web content if available
        if web_content:
            search_text += "----------【网络检索内容】-----------\n"
            search_text += web_content
    

        return history, search_text, rewrite_query
    else :
        return history, '', ''


history, search_text, rewrite_query = predict("请和我介绍一下有关rag技术的发展")

print('history:',history)
print('search_text:',search_text)
print('rewrite_query:',rewrite_query)
