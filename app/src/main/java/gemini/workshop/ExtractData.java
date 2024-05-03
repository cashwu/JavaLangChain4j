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
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.UserMessage;

public class ExtractData {

    record Person(String name, int age) {}

    interface PersonExtractor {

        @UserMessage("""
            提取下述人員的姓名和年齡。傳回一個 JSON 文檔，其中包含 "name" 和 "age" property,
            請參照下面的結構 : {"name": "John Doe", "age": 34}
            僅返回 JSON，沒有加上任何 Markdown 標記。
            這是描述此人的文件：:
            ---
            {{it}}
            ---
            JSON:
            """)
        Person extractPerson(String text);
    }

    public static void main(String[] args) {
        ChatLanguageModel model = VertexAiGeminiChatModel.builder()
            .project(System.getenv("PROJECT_ID"))
            .location(System.getenv("LOCATION"))
            .modelName("gemini-1.0-pro-001")
            .temperature(0f)
            .topK(1)
            .build();

        PersonExtractor extractor = AiServices.create(PersonExtractor.class, model);

//        Person person = extractor.extractPerson("""
//            Anna is a 23 year old artist based in Brooklyn, New York. She was born and
//            raised in the suburbs of Chicago, where she developed a love for art at a
//            young age. She attended the School of the Art Institute of Chicago, where
//            she studied painting and drawing. After graduating, she moved to New York
//            City to pursue her art career. Anna's work is inspired by her personal
//            experiences and observations of the world around her. She often uses bright
//            colors and bold lines to create vibrant and energetic paintings. Her work
//            has been exhibited in galleries and museums in New York City and Chicago.
//            """
//        );
        Person person = extractor.extractPerson("""
            Anna 是一位 23 歲的藝術家，住在紐約布魯克林。她出生並且她在芝加哥郊區長大，在那裡的一所學校培養了對藝術的熱愛。
            她就讀於芝加哥藝術學院，她學習繪畫。
            畢業後，她搬到了紐約追求她的藝術事業。
            安娜的作品靈感來自於她的個人經歷、她對周遭世界的經驗和觀察。
            她經常使用明亮的色彩和大膽的線條創造出充滿活力和活力的畫作。
            她的工作曾在紐約和芝加哥的畫廊和博物館展出。
            """
        );

        System.out.println(person.name());  // Anna
        System.out.println(person.age());   // 23
    }
}
