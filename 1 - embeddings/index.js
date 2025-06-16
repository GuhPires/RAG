import { GoogleGenAI } from "@google/genai";
import * as dotenv from "dotenv";

dotenv.config({ path: "../.env" });

const { GEMINI_API_KEY } = process.env;

async function embedContent(content, { isQuery } = { isQuery: false }) {
	const ai = new GoogleGenAI({ vertexai: false, apiKey: GEMINI_API_KEY });

	const response = await ai.models.embedContent({
		model: "text-embedding-004",
		contents: content,
		config: {
			taskType: isQuery ? "RETRIEVAL_QUERY" : "RETRIEVAL_DOCUMENT",
		},
	});

	return response;
}

try {
	const embedded = await embedContent("First ever embedded content!");
	const embeddedQuery = await embedContent("First ever embedded content!", {
		isQuery: true,
	});
	// embedding SHOULD be different (https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/task-types#retrieve_information_from_texts)
	console.log(JSON.stringify(embedded), "\n\n");
	console.log(JSON.stringify(embeddedQuery));
	console.log("Embed size:", embedded.embeddings[0].values.length);
} catch (err) {
	console.error("Error:", err);
}
