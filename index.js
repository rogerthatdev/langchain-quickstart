// simple LLM chain, which just relies on information in the prompt template to respond.

// Initialize OpenAI Chat Model class 
import { ChatOpenAI } from "@langchain/openai";

const chatModel = new ChatOpenAI({});
const invocation = await chatModel.invoke("what is LangSmith?");
console.log(invocation)

// Import ChatPromptTemplate to use prompt templates
import { ChatPromptTemplate } from "@langchain/core/prompts";

// create a prompt using template
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a world class technical documentation writer."],
  ["user", "{input}"],
]);

// create a simple chain
const chain = prompt.pipe(chatModel);

// invoke the chain with a question
const invokeChain = await chain.invoke({
    input: "what is LangSmith?",
});

console.log(`\n\n############ Invoking via chain ############\n\n`, invokeChain)