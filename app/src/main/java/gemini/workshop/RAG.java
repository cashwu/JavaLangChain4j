/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package gemini.workshop;

import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.parser.apache.pdfbox.ApachePdfBoxDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.vertexai.VertexAiEmbeddingModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.injector.DefaultContentInjector;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.model.vertexai.VertexAiGeminiChatModel;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.List;

public class RAG {

    interface LlmExpert {
        String ask(String question);
    }

    public static void main(String[] args) throws IOException {
        ApachePdfBoxDocumentParser pdfParser = new ApachePdfBoxDocumentParser();
        Document document = pdfParser.parse(new FileInputStream("/tmp/attention-is-all-you-need.pdf"));

        VertexAiEmbeddingModel embeddingModel = VertexAiEmbeddingModel.builder()
                                                                      .endpoint(System.getenv("LOCATION")
                                                                                        + "-aiplatform.googleapis.com:443")
                                                                      .project(System.getenv("PROJECT_ID"))
                                                                      .location(System.getenv("LOCATION"))
                                                                      .publisher("google")
                                                                      .modelName("textembedding-gecko@001")
                                                                      .maxRetries(3)
                                                                      .build();

        // 一個記憶體的 vector db，用於存儲 vector embeddings
        InMemoryEmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        // 拆分為 500 個字元的片段，重疊 100 個字元
        DocumentSplitter recursive = DocumentSplitters.recursive(500, 100);

        // 將 PDF 文件載入並拆分為 chunks
        // 為所有 chunks 創建 vector embeddings。
        var storeIngestor = EmbeddingStoreIngestor.builder()
                                                  .documentSplitter(recursive)
                                                  .embeddingModel(embeddingModel)
                                                  .embeddingStore(embeddingStore)
                                                  .build();
        storeIngestor.ingest(document);

        ChatLanguageModel model = VertexAiGeminiChatModel.builder()
                                                         .project(System.getenv("PROJECT_ID"))
                                                         .location(System.getenv("LOCATION"))
                                                         .modelName("gemini-1.0-pro")
                                                         .maxOutputTokens(1000)
                                                         .build();

        // 檢索器 - 計算使用者查詢的 vector embedding 來查詢 vector db，以在 db 中查找相似的 vector
        EmbeddingStoreContentRetriever retriever = new EmbeddingStoreContentRetriever(embeddingStore, embeddingModel);

        LlmExpert expert = AiServices.builder(LlmExpert.class)
                                     .chatLanguageModel(model)
                                     .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                                     .contentRetriever(retriever)
                                     //                                     .retrievalAugmentor(DefaultRetrievalAugmentor.builder()
                                     //                                                                                  .contentInjector(
                                     //                                                                                          DefaultContentInjector.builder()
                                     //                                                                                                                .promptTemplate(
                                     //                                                                                                                        PromptTemplate.from(
                                     //                                                                                                                                """
                                     //                                                                                                                                        你是大型語言模型的專家，\s
                                     //                                                                                                                                        你擅長簡單明瞭地解釋有關LLM的問題，並且可以使用正體(繁體)中文回答
                                     //
                                     //                                                                                                                                        這是問題: {{userMessage}}
                                     //
                                     //                                                                                                                                        使用以下資訊回答:
                                     //                                                                                                                                        {{contents}}
                                     //                                                                                                                                        """))
                                     //                                                                                                                .build())
                                     //                                                                                  .queryRouter(new DefaultQueryRouter(
                                     //                                                                                          retriever))
                                     //                                                                                  .build())
                                     .build();

        //        List.of(
        //            "What neural network architecture can be used for language models?",
        // "變壓器神經網路有哪些不同的組成部分？",
        //            "What are the different components of a transformer neural network?",
        // "大型語言模型中的注意力是什麼？",
        //            "What is attention in large language models?",
        //            "What is the name of the process that transforms text into vectors?"
        //        )

        List.of("What neural network architecture can be used for language models?")
            .forEach(query -> System.out.printf("%n=== %s === %n%n %s %n%n", query, expert.ask(query)));
    }
}
