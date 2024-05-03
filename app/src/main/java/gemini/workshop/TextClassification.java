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

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.vertexai.VertexAiGeminiChatModel;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.output.Response;

import java.util.Map;

public class TextClassification {
    public static void main(String[] args) {
        ChatLanguageModel model = VertexAiGeminiChatModel.builder()
            .project(System.getenv("PROJECT_ID"))
            .location(System.getenv("LOCATION"))
            .modelName("gemini-1.0-pro")
            .maxOutputTokens(10)
            .maxRetries(3)
            .build();

        PromptTemplate promptTemplate = PromptTemplate.from("""
            分析下面這段文字的情感。僅用一個字來描述這種情緒。
            
            INPUT: 這是個好消息！
            OUTPUT: POSITIVE 積極

            INPUT: Pi 大約等於 3.14
            OUTPUT: NEUTRAL 中性的

            INPUT: 我真的不喜歡披薩。誰會用鳳梨當披薩配料？
            OUTPUT: NEGATIVE 消極的

            INPUT: {{text}}
            OUTPUT:
            """);

//        Prompt prompt = promptTemplate.apply(
//            Map.of("text", "我喜歡草莓!"));

        Prompt prompt = promptTemplate.apply(
                Map.of("text", "去死吧 !"));

        Response<AiMessage> response = model.generate(prompt.toUserMessage());

        System.out.println(response.content().text());
    }
}
