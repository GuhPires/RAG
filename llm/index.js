import { GoogleGenAI } from "@google/genai";
import { Pinecone } from "@pinecone-database/pinecone";
import * as dotenv from "dotenv";

dotenv.config({ path: "../.env" });

const { GEMINI_API_KEY, PINECONE_API_KEY } = process.env;
const SHOULD_SEED = false;

const db = new Pinecone({ apiKey: PINECONE_API_KEY });
const collection = db.index("learn");
const knowledgeBase = [
	"O sol é uma estrela massiva e o centro do nosso sistema solar.",
	"A fotossíntese é o processo pelo qual as plantas usam a luz solar para criar alimentos.",
	"A água (H2O) é composta por dois átomos de hidrogênio e um átomo de oxigênio.",
	"A lua é o único satélite natural da Terra.",
	"JavaScript é uma linguagem de programação popular para desenvolvimento web.",
];

async function upsertVectors(embeddings) {
	const records = embeddings.map((embedding, i) => ({
		id: `${i + 1}`,
		values: embedding.values,
		metadata: { text: knowledgeBase[i] },
	}));

	console.log("INSERTING:", records);

	await collection.upsert(records);
}

async function similaritySearch(vector) {
	return await collection.query({
		vector,
		topK: 2,
		includeMetadata: true,
	});
}

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

async function useLLM(context, query) {
	const template = `
  Crie uma resposta curta mas humana e educada para a pergunta apenas utilizando o contexto abaixo:

  ### Contexto
  ${context.join("\n")}

  ### Pergunta
  ${query}
  `;

	console.log("TEMPLATE:", template);

	const ai = new GoogleGenAI({ vertexai: false, apiKey: GEMINI_API_KEY });

	const response = await ai.models.generateContent({
		model: "gemini-2.0-flash",
		contents: template,
	});

	return response.text;
}

try {
	if (SHOULD_SEED) {
		const vectors = await embedContent(knowledgeBase);
		console.log(JSON.stringify(vectors));
		await upsertVectors(vectors.embeddings);
	}

	// QUERY INFORMATION
	const input =
		"O que é água e quão importante esse elemento é para as plantas?";
	const query = await embedContent(input, { isQuery: true });
	const records = await similaritySearch(query.embeddings[0].values);

	const text = await useLLM(
		records.matches.map((r) => r.metadata.text),
		input
	);
	console.log("LLM RESPONSE:", text);
} catch (err) {
	console.error("Error:", err);
}
