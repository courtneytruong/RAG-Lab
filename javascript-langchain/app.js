import { OpenAIEmbeddings } from "@langchain/openai";
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

  console.log("=== Embedding Inspector Lab ===\n");
  console.log("Generating embeddings for three sentences...\n");

  // Define the three test sentences
  const sentences = [
    "The canine barked loudly.",
    "The dog made a noise.",
    "The electron spins rapidly.",
  ];

  // Generate and display embeddings for each sentence using GitHub Models API
  const vectors = [];
  for (let i = 0; i < sentences.length; i++) {
    console.log(`Sentence ${i + 1}: "${sentences[i]}"`);
    const embedding = await fetchGithubEmbedding(sentences[i]);
    vectors.push(embedding);
  }

  // Show the cosine similarities between the embeddings
  console.log("\n=== Embedding Vectors ===\n");
  const sim12 = cosineSimilarity(vectors[0], vectors[1]);
  const sim23 = cosineSimilarity(vectors[1], vectors[2]);
  const sim31 = cosineSimilarity(vectors[2], vectors[0]);
  console.log(
    `Cosine similarity between Sentence 1 and Sentence 2: ${sim12.toFixed(4)}`,
  );
  console.log(
    `Cosine similarity between Sentence 2 and Sentence 3: ${sim23.toFixed(4)}`,
  );
  console.log(
    `Cosine similarity between Sentence 3 and Sentence 1: ${sim31.toFixed(4)}`,
  );

  console.log("\n📊 Observations:");
  console.log("- Each embedding is just an array of floating-point numbers");
  console.log(
    "- Sentences 1 and 2 (about dogs) will have similar values in many dimensions",
  );
  console.log(
    "- Sentence 3 (about electrons) will differ significantly from sentences 1 and 2",
  );
  console.log(
    "\nThis demonstrates that 'AI embeddings' are simply numerical vectors,",
  );
  console.log(
    "not magic—they represent semantic meaning as coordinates in high-dimensional space.",
  );
}

main().catch(console.error);
