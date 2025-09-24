import json
import base64
from typing import List, AsyncGenerator
from sqlalchemy.orm import Session
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from app.core.config import settings
from app.models.chat import Message
from app.models.knowledge import KnowledgeBase, Document
from langchain.globals import set_verbose, set_debug
from app.services.vector_store import VectorStoreFactory
from app.services.embedding.embedding_factory import EmbeddingsFactory
from app.services.llm.llm_factory import LLMFactory
from app.services.vector_store.chroma import StaticListRetriever
set_verbose(True)
set_debug(True)

async def generate_response(
    query: str,
    messages: dict,
    knowledge_base_ids: List[int],
    chat_id: int,
    db: Session
) -> AsyncGenerator[str, None]:
    try:
        # Create user message
        user_message = Message(
            content=query,
            role="user",
            chat_id=chat_id
        )
        db.add(user_message)
        db.commit()
        
        # Create bot message placeholder
        bot_message = Message(
            content="",
            role="assistant",
            chat_id=chat_id
        )
        db.add(bot_message)
        db.commit()
        
        # Get knowledge bases and their documents
        knowledge_bases = (
            db.query(KnowledgeBase)
            .filter(KnowledgeBase.id.in_(knowledge_base_ids))
            .all()
        )
        
        # Initialize embeddings
        embeddings = EmbeddingsFactory.create()

        # top_k = 3
        # score_threshold = 0.6

        # vector_stores = []
        # for kb in knowledge_bases:
        #     documents = db.query(Document).filter(Document.knowledge_base_id == kb.id).all()
        #     if documents:
        #         vector_store = VectorStoreFactory.create(
        #             store_type=settings.VECTOR_STORE_TYPE,
        #             collection_name=f"kb_{kb.id}",
        #             embedding_function=embeddings,
        #         )
        #         vector_stores.append(vector_store)

        # if not vector_stores:
        #     error_msg = "I don't have any knowledge base to help answer your question."
        #     yield f'0:"{error_msg}"\n'
        #     bot_message.content = error_msg
        #     db.commit()
        #     return

        # # Truy vấn và lọc score
        # vs = vector_stores[0]
        # print("Vector store:", vs)
        # results = vs.similarity_search_with_score(query, k=top_k)  # [(doc, score), ...]
        # print("Raw results:", results)
        # filtered_docs = [doc for doc, score in results if score >= score_threshold]
        # print("Filtered results:", filtered_docs)

        # if not filtered_docs:
        #     error_msg = "Information is missing on related topic."
        #     yield f'0:"{error_msg}"\n'
        #     bot_message.content = error_msg
        #     db.commit()
        #     return
        top_k = 5

        vector_stores = []
        for kb in knowledge_bases:
            documents = db.query(Document).filter(Document.knowledge_base_id == kb.id).all()
            if documents:
                vector_store = VectorStoreFactory.create(
                    store_type=settings.VECTOR_STORE_TYPE,
                    collection_name=f"kb_{kb.id}",
                    embedding_function=embeddings,
                )
                vector_stores.append(vector_store)

        if not vector_stores:
            error_msg = "I don't have any knowledge base to help answer your question."
            yield f'0:"{error_msg}"\n'
            bot_message.content = error_msg
            db.commit()
            return

        # Thay vì dùng similarity_search_with_score và lọc score thủ công
        # ta dùng phương thức hybrid_search đã tích hợp BM25 với vector
        vs = vector_stores[0]
        print("Vector store:", vs)

        # Gọi hybrid_search với top_k và trọng số weights để cân bằng vector và bm25
        results = vs.hybrid_search(query, k=top_k, weights=[0.4, 0.6])  # trả về danh sách Document, không có score

        print("Hybrid search results:", results)

        if not results:
            error_msg = "Information is missing on related topic."
            yield f'0:"{error_msg}"\n'
            bot_message.content = error_msg
            db.commit()
            return
        # Initialize the language model
        llm = LLMFactory.create()
        
        # Create contextualize question prompt
        contextualize_q_system_prompt = (
            "Dựa trên lịch sử hội thoại và câu hỏi mới nhất của người dùng "
            "có thể tham chiếu đến ngữ cảnh trong lịch sử hội thoại, "
            "hãy xây dựng lại câu hỏi sao cho nó có thể hiểu được độc lập "
            "mà không cần đến lịch sử hội thoại. KHÔNG trả lời câu hỏi, chỉ "
            "định dạng lại câu hỏi nếu cần, hoặc giữ nguyên nếu đã rõ. Bạn chỉ trả lời bằng tiếng việt "
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        filtered_retriever = StaticListRetriever(docs=results)

        
        # Create history aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm, 
            filtered_retriever,
            contextualize_q_prompt
        )

        # Create QA prompt
        # Prompt trả lời QA tối ưu (chuyên nghiệp, có trích dẫn)
        qa_system_prompt = (
            "Bạn là một trợ lý AI chuyên tra cứu QUY TRÌNH NỘI BỘ.\n"
            "Bạn chỉ được phép sử dụng thông tin trong QUY TRÌNH NỘI BỘ để trả lời câu hỏi.\n"
            "Nếu không tìm thấy thông tin hoặc không chắc chắn, hãy báo rõ.\n"
            "Trả lời ngắn gọn, chính xác, lịch sự, có trích dẫn đoạn tham chiếu theo định dạng [Trích dẫn: đoạn số X].\n"
            "Bạn chỉ trả lời bằng tiếng việt \n"
            "QUY TRÌNH NỘI BỘ:\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # 修改 create_stuff_documents_chain 来自定义 context 格式
        document_prompt = PromptTemplate.from_template("\n\n- {page_content}\n\n")

        # Create QA chain
        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt,
            document_variable_name="context",
            document_prompt=document_prompt
        )

        # Create retrieval chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain,
        )

        # Generate response
        chat_history = []
        for message in messages["messages"]:
            if message["role"] == "user":
                chat_history.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                # if include __LLM_RESPONSE__, only use the last part
                if "__LLM_RESPONSE__" in message["content"]:
                    message["content"] = message["content"].split("__LLM_RESPONSE__")[-1]
                chat_history.append(AIMessage(content=message["content"]))

        full_response = ""
        async for chunk in rag_chain.astream({
            "input": query,
            "chat_history": chat_history
        }):
            # if "context" in chunk:
            #     serializable_context = []
            #     for context in chunk["context"]:
            #         serializable_doc = {
            #             "page_content": context.page_content.replace('"', '\\"'),
            #             "metadata": context.metadata,
            #         }
            #         serializable_context.append(serializable_doc)
                
            #     # 先替换引号，再序列化
            #     escaped_context = json.dumps({
            #         "context": serializable_context
            #     })

            #     # 转成 base64
            #     base64_context = base64.b64encode(escaped_context.encode()).decode()

            #     # 连接符号
            #     separator = "__LLM_RESPONSE__"
                
            #     yield f'0:"{base64_context}{separator}"\n'
            #     full_response += base64_context + separator

            if "answer" in chunk:
                full_response += chunk["answer"]
        # Escape ký tự đặc biệt, xuống dòng
        escaped_response = full_response.replace('"', '\\"').replace('\n', '\\n')
        # Yield đúng format 1 lần duy nhất với toàn bộ response
        yield f'0:"{escaped_response}"\n'
        # Update bot message content
        bot_message.content = full_response
        db.commit()
            
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        print(error_message)
        yield '3:{text}\n'.format(text=error_message)
        
        # Update bot message with error
        if 'bot_message' in locals():
            bot_message.content = error_message
            db.commit()
    finally:
        db.close()