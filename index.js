// simple LLM chain, which just relies on information in the prompt template to respond.

// Initialize OpenAI Chat Model class 
import { ChatOpenAI } from "@langchain/openai";

const chatModel = new ChatOpenAI({});
const invocation = await chatModel.invoke("what is LangSmith?");
console.log(invocation)

// Import ChatPromptTemplate to use prompt templates
import { ChatPromptTemplate } from "@langchain/core/prompts";

// create a prompt using template
const basicPrompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a world class technical documentation writer."],
  ["user", "{input}"],
]);

// create a simple chain
const chain = basicPrompt.pipe(chatModel);

// invoke the chain with a question
const invokeChain = await chain.invoke({
    input: "what is LangSmith?",
});

console.log(`\n\n############ Invoking via chain (returning a message) ############\n\n`, invokeChain)
// out is a message, next lets add a simple output parser to convert the chat
// message to a string

// Bring in the string output parser
import { StringOutputParser } from "@langchain/core/output_parsers";

const outputParser = new StringOutputParser();

const llmChain = basicPrompt.pipe(chatModel).pipe(outputParser);

const invokeChainForString = await llmChain.invoke({
  input: "what is LangSmith?",
});

console.log(`\n\n############ Invoking via chain (returning a string) ############\n\n`, invokeChainForString)

// To avoid hallucinations, provide context to the LLM via retrieval

// Load data we want to index using Cheerio document loader

import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";

const loader = new CheerioWebBaseLoader(
  "https://docs.smith.langchain.com/user_guide"
);

const docs = await loader.load();

console.log(docs.length);
console.log(docs[0].pageContent.length);

// use text_splitter to split document into managable chunks
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const splitter = new RecursiveCharacterTextSplitter();

const splitDocs = await splitter.splitDocuments(docs);

console.log(splitDocs.length);
console.log(splitDocs[0].pageContent.length);

// We now need to index the local documents into a vectorstore

// First we need an embedding model
import { OpenAIEmbeddings } from "@langchain/openai";

const embeddings = new OpenAIEmbeddings();

// Use an in-memory demo vectorstore
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// Create a vector store; use the split docs and the embedding model
const vectorstore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);

//  Set up the chain that takes a question and the retrieved documents and generates an answer.
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
// import { ChatPromptTemplate } from "@langchain/core/prompts";

const prompt =
  ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}`);

const documentChain = await createStuffDocumentsChain({
  llm: chatModel,
  prompt,
});

// Run this ourselves by passing in documents directly
import { Document } from "@langchain/core/documents";

const docChainInvocation = await documentChain.invoke({
  input: "what is LangSmith?",
  context: [
    new Document({
      pageContent:
        "LangSmith is a platform for building production-grade LLM applications.",
    }),
  ],
});

console.log(`Doc chain invocation:\n`,docChainInvocation)

// Documents to first come from the retriever we just set up
import { createRetrievalChain } from "langchain/chains/retrieval";

const retriever = vectorstore.asRetriever();

const retrievalChain = await createRetrievalChain({
  combineDocsChain: documentChain,
  retriever,
});

// Invoke the retrieval chain
const result = await retrievalChain.invoke({
    input: "what is LangSmith?",
  });
  
console.log(`Retrieval chain invocation:\n`, result.answer);

// Conversational retrieval chain

// Create a new chain that will take in the most recent input and convo history
// and use an LLM to generate a search query

import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { MessagesPlaceholder } from "@langchain/core/prompts";


// Basic prompt for comparison:
// const basicPrompt = ChatPromptTemplate.fromMessages([
//   ["system", "You are a world class technical documentation writer."],
//   ["user", "{input}"],
// ]);

const historyAwarePrompt = ChatPromptTemplate.fromMessages([
  new MessagesPlaceholder("chat_history"),
  ["user", "{input}"],
  [
    "user",
    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
  ],
]);

const historyAwareRetrieverChain = await createHistoryAwareRetriever({
  llm: chatModel,
  retriever,
  rephrasePrompt: historyAwarePrompt,
});

// simulate user asking a follow up question and invoke the retriever chain

import { HumanMessage, AIMessage } from "@langchain/core/messages";

const chatHistory = [
  new HumanMessage("Can LangSmith help test my LLM applications?"),
  new AIMessage("Yes!"),
];

await historyAwareRetrieverChain.invoke({
  chat_history: chatHistory,
  input: "Tell me how!",
}) // returns douments about testing in LangSmith 


// Create a new chat prompt, and use the chat history 

const historyAwareRetrievalPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "Answer the user's questions based on the below context:\n\n{context}",
  ],
  new MessagesPlaceholder("chat_history"),
  ["user", "{input}"],
]);

const historyAwareCombineDocsChain = await createStuffDocumentsChain({
  llm: chatModel,
  prompt: historyAwareRetrievalPrompt,
});

const conversationalRetrievalChain = await createRetrievalChain({
  retriever: historyAwareRetrieverChain,
  combineDocsChain: historyAwareCombineDocsChain,
});

// test end-to-end

const result2 = await conversationalRetrievalChain.invoke({
  chat_history: [
    new HumanMessage("Can LangSmith help test my LLM applications?"),
    new AIMessage("Yes!"),
  ],
  input: "tell me how",
});

console.log(result2.answer);