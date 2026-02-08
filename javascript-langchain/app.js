import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import readline from "readline";
import dotenv from "dotenv";

// Load environment variables
dotenv.config();

/**
 * Calculate cosine similarity between two vectors
 * Cosine similarity = (A · B) / (||A|| * ||B||)
 */
function cosineSimilarity(vectorA, vectorB) {
  if (vectorA.length !== vectorB.length) {
    throw new Error("Vectors must have the same dimensions");
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vectorA.length; i++) {
    dotProduct += vectorA[i] * vectorB[i];
    normA += vectorA[i] * vectorA[i];
    normB += vectorB[i] * vectorB[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function main() {
  // Function to search sentences in the vector store
  async function searchSentences(vectorStore, query, k = 3) {
    const results = await vectorStore.similaritySearchWithScore(query, k);
    console.log(`\n🔍 Top ${k} results for query: "${query}"`);
    results.forEach(([doc, score], idx) => {
      console.log(
        `#${idx + 1} | Score: ${score.toFixed(4)} | Sentence: "${doc.pageContent}"`,
      );
    });
    return results;
  }
  console.log("🤖 JavaScript LangChain Agent Starting...\n");

  // Debug: Print GITHUB_TOKEN status
  if (!process.env.GITHUB_TOKEN) {
    console.error("❌ Error: GITHUB_TOKEN not found in environment variables.");
    console.log("Please create a .env file with your GitHub token:");
    console.log("GITHUB_TOKEN=your-github-token-here");
    console.log("\nGet your token from: https://github.com/settings/tokens");
    console.log("Or use GitHub Models: https://github.com/marketplace/models");
    process.exit(1);
  } else {
    console.log(
      "✅ GITHUB_TOKEN loaded. Length:",
      process.env.GITHUB_TOKEN.length,
    );
    // Uncomment the next line to print the token value (for debugging only, do not share this!)
    // console.log("Token:", process.env.GITHUB_TOKEN);
  }

  // Function to call GitHub Models Embedding API directly
  async function fetchGithubEmbedding(text) {
    const url = "https://models.github.ai/inference/embeddings";
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.GITHUB_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "text-embedding-3-small",
        input: text,
      }),
    });
    if (!response.ok) {
      const err = await response.text();
      throw new Error(`GitHub Models API error: ${response.status} ${err}`);
    }
    const data = await response.json();
    // The API returns { data: [{ embedding: [...] }] }
    return data.data[0].embedding;
  }

  // Use a plain object for embeddings compatible with MemoryVectorStore
  const embeddings = {
    embedQuery: fetchGithubEmbedding,
    embedDocuments: async (texts) =>
      Promise.all(texts.map(fetchGithubEmbedding)),
  };

  // Create a MemoryVectorStore instance with no initial texts
  const vectorStore = await MemoryVectorStore.fromTexts([], [], embeddings);

  console.log("=== Embedding Inspector Lab ===\n");
  console.log("Generating embeddings for three sentences...\n");

  // Define the three test sentences (same as Lab 1)
  const sentences = [
    "The canine barked loudly.",
    "The dog made a noise.",
    "The electron spins rapidly.",
  ];

  // Prepare documents with metadata
  const now = new Date().toISOString();
  const docs = sentences.map((sentence, idx) => ({
    pageContent: sentence,
    metadata: {
      createdAt: now,
      index: idx,
    },
  }));

  // Add all documents to the vector store at once
  await vectorStore.addDocuments(docs);

  // Print confirmation and each sentence
  console.log(`\n✅ Stored ${docs.length} sentences in the vector store.`);
  docs.forEach((doc, i) => {
    console.log(`Sentence ${i + 1}: "${doc.pageContent}"`);
  });

  // === Semantic Search Interactive Loop ===
  console.log("\n=== Semantic Search ===");

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  // Promisified question for async/await
  function askQuestion(query) {
    return new Promise((resolve) => rl.question(query, resolve));
  }

  while (true) {
    const userInput = (
      await askQuestion("Enter a search query (or 'quit' to exit): ")
    ).trim();
    if (
      userInput.toLowerCase() === "quit" ||
      userInput.toLowerCase() === "exit"
    ) {
      break;
    }
    if (userInput === "") {
      continue;
    }
    await searchSentences(vectorStore, userInput);
    console.log(""); // Blank line for readability
  }
  rl.close();
  console.log("👋 Goodbye! Thanks for using the Semantic Search Lab.");
}

main().catch(console.error);
