import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import fs from "fs";
import path from "path";
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
      if (response.status === 429) {
        const retryAfter = response.headers.get("Retry-After");
        let waitMsg = "";
        if (retryAfter) {
          // Retry-After can be seconds or a date string
          const seconds = Number(retryAfter);
          if (!isNaN(seconds) && seconds > 0) {
            waitMsg = `Please wait ${seconds} seconds before trying again.`;
          } else {
            // Try to parse as date
            const retryDate = new Date(retryAfter);
            const now = new Date();
            const diff = Math.ceil((retryDate - now) / 1000);
            if (!isNaN(diff) && diff > 0) {
              waitMsg = `Please wait ${diff} seconds (until ${retryDate.toLocaleString()}) before trying again.`;
            }
          }
        }
        if (waitMsg) {
          console.error(
            `⏳ Received 429 Too Many Requests from GitHub Models API. You are being rate limited. ${waitMsg}`,
          );
        } else {
          console.error(
            "⏳ Received 429 Too Many Requests from GitHub Models API. You are being rate limited. Please wait a few minutes before trying again.",
          );
        }
      }
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

  // === Loading Documents into Vector Database ===
  console.log("=== Loading Documents into Vector Database ===");
  // Use process.cwd() for workspace root
  const filePath = path.join(process.cwd(), "HealthInsuranceBrochure.md");
  const docId = await loadDocument(vectorStore, filePath);
  if (docId) {
    console.log(`Document '${filePath}' loaded successfully with ID: ${docId}`);
  }

  // Load EmployeeHandbook.md from workspace root
  const employeeHandbookPath = path.join(process.cwd(), "EmployeeHandbook.md");
  const handbookDocId = await loadDocument(vectorStore, employeeHandbookPath);
  if (handbookDocId) {
    console.log(
      `Document '${employeeHandbookPath}' loaded successfully with ID: ${handbookDocId}`,
    );
  }

  // Async function to load a document from file and add to vector store
  async function loadDocument(vectorStore, filePath) {
    try {
      const text = fs.readFileSync(filePath, "utf-8");
      const fileName = path.basename(filePath);
      const createdAt = new Date().toISOString();
      const document = new Document({
        pageContent: text,
        metadata: {
          fileName,
          createdAt,
        },
      });
      await vectorStore.addDocuments([document]);
      console.log(
        `✅ Loaded '${fileName}' (${text.length} chars) into vector store.`,
      );
      // Return document ID if available, else fileName
      return document.id || fileName;
    } catch (err) {
      const msg = err.message || String(err);
      if (
        msg.includes("maximum context length") ||
        msg.toLowerCase().includes("token")
      ) {
        console.error(
          `⚠️ This document is too large to embed as a single chunk.`,
        );
        console.error(
          `Token limit exceeded. The embedding model can only process up to 8,191 tokens at once.`,
        );
        console.error(
          `Solution: The document needs to be split into smaller chunks.`,
        );
      } else {
        console.error(`❌ Error loading file '${filePath}':`, msg);
      }
      return null;
    }
  }
}

main().catch(console.error);
