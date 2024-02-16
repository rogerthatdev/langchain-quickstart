// simple LLM chain, which just relies on information in the prompt template to respond.

// Initialize OpenAI Chat Model class 
import { ChatOpenAI } from "@langchain/openai";

const chatModel = new ChatOpenAI({});

const invocation = await chatModel.invoke("what is LangSmith?");

console.log(invocation)

