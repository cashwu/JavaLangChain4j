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
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.input.PromptTemplate;

import java.util.HashMap;
import java.util.Map;

public class PromptTemplateClass {
    public static void main(String[] args) {
        ChatLanguageModel model = VertexAiGeminiChatModel.builder()
            .project(System.getenv("PROJECT_ID"))
            .location(System.getenv("LOCATION"))
            .modelName("gemini-1.0-pro")
            .maxOutputTokens(500)
            .temperature(0.8f)
            .topK(40)
            .topP(0.95f)
            .maxRetries(3)
            .build();

        PromptTemplate promptTemplate = PromptTemplate.from("""
            你是一位友善的廚師，擁有豐富的烹飪經驗。
            使用以下成分建立 {{dish}} 的食譜 :
            {{ingredients}}，並給它一個名字
            請使用正體(繁體)中文回答
            """
        );

        Map<String, Object> variables = new HashMap<>();
        variables.put("dish", "甜點");
        variables.put("ingredients", "草莓、巧克力和生奶油");

        Prompt prompt = promptTemplate.apply(variables);

        Response<AiMessage> response = model.generate(prompt.toUserMessage());

        System.out.println(response.content().text());
    }
}
